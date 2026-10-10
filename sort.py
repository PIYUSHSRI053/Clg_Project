# Simple SORT tracker
# It gives an ID to every vehicle and keeps the same ID in the next frames
# by matching the boxes that overlap the most (IOU).

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class Track:
    def __init__(self, track_id, box):
        self.id = track_id
        self.box = box
        self.hits = 1       # frames in which this vehicle was seen
        self.age = 0        # frames since it was last seen
        self.still = 0      # frames in a row in which it did not move


class Sort:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.total = 0      # total vehicles seen so far

    def update(self, boxes):
        used = []
        for track in self.tracks:
            # find the new box that overlaps this track the most
            best_iou, best = 0, -1
            for i, box in enumerate(boxes):
                if i not in used and iou(track.box, box) > best_iou:
                    best_iou, best = iou(track.box, box), i

            if best_iou >= self.iou_threshold:
                moved = abs(boxes[best][0] - track.box[0]) + abs(boxes[best][1] - track.box[1])
                track.still = track.still + 1 if moved < 3 else 0
                track.box = boxes[best]
                track.hits += 1
                track.age = 0
                used.append(best)
                if track.hits == self.min_hits:
                    self.total += 1
            else:
                track.age += 1

        # boxes that did not match are new vehicles
        for i, box in enumerate(boxes):
            if i not in used:
                self.tracks.append(Track(self.next_id, box))
                self.next_id += 1

        # remove vehicles that are gone
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.age == 0]
