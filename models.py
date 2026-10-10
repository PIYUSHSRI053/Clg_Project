# Loads our trained models and finds the vehicles in a frame
import os
import cv2
import torch
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from sort import iou

CLASSES = ["car", "motorcycle", "bus", "truck"]
CONF = 0.15     # boxes below this score are ignored (lower = more vehicles are marked)
FILES = {
    "YOLO11": "models/yolo11.pt",
    "Faster R-CNN": "models/fasterrcnn.pth",
    "RetinaNet": "models/retinanet.pth",
}
VOTES = {"YOLO11": 1, "Faster R-CNN": 1, "RetinaNet": 1}   

def load_model(name):
    if name == "YOLO11":
        return YOLO(FILES[name])
    model = torch.load(FILES[name], map_location="cpu", weights_only=False)
    if isinstance(model, dict):
        # the file has only the trained weights, so make the empty model first and fill it
        weights = model
        build = fasterrcnn_resnet50_fpn if name == "Faster R-CNN" else retinanet_resnet50_fpn
        model = build(weights=None, weights_backbone=None, num_classes=5, min_size=640, max_size=640)
        model.load_state_dict(weights)
    return model.eval()

def load_accident_model():
    if os.path.exists("models/accident.pt"):
        return YOLO("models/accident.pt")

# returns a list of [x1, y1, x2, y2, label]
def detect(name, model, frame):
    found = []
    if name == "Combine All":
        return detect_all(model, frame)
    if name == "YOLO11":
        r = model(frame, conf=CONF, imgsz=960, verbose=False)[0]
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            found.append([int(v) for v in box.xyxy[0]] + [label])
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            out = model([img])[0]
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if score >= CONF:
                found.append([int(v) for v in box] + [CLASSES[int(label) - 1]])
    return found

# Combine All: every model looks at the frame. Boxes that overlap are the same vehicle,
# and the votes of the models that found it are added. A vehicle needs 2 votes to be kept.
def detect_all(models, frame):
    vehicles = []                               # each item is [box, votes]
    for name in FILES:
        for d in detect(name, models[name], frame):
            same = [v for v in vehicles if iou(v[0], d) > 0.5]
            if same:
                same[0][1] += VOTES[name]
            else:
                vehicles.append([d, VOTES[name]])
    return [v[0] for v in vehicles if v[1] >= 2]

# Small and far vehicles are missed on a small picture, so we detect on the big frame.
# Then we make the frame small for the screen and move the boxes to fit it.
def detect_small(name, model, big):
    found = detect(name, model, big)
    sx, sy = 480 / big.shape[1], 270 / big.shape[0]
    found = [[int(d[0] * sx), int(d[1] * sy), int(d[2] * sx), int(d[3] * sy), d[4]] for d in found]
    return cv2.resize(big, (480, 270)), found

# how many vehicles of each type were found
def count_vehicles(found):
    counts = {c: 0 for c in CLASSES}
    for d in found:
        counts[d[4]] += 1
    return counts

# how sure the accident model is that the frame has an accident (0 to 1)
def accident_probability(model, frame):
    if model is None:
        return 0
    r = model(frame, verbose=False)[0]
    for i, name in r.names.items():
        if not name.lower().startswith("no"):       # the class that is not "no accident"
            return float(r.probs.data[i])
    return 0

# True if the probability is more than 80% AND some vehicle is not moving
def accident_seen(probability, tracks):
    stopped = any(t.still >= 6 for t in tracks)
    return probability > 0.6 and stopped