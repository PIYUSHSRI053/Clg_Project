# Smart Traffic Control System        run:  streamlit run app.py
import time, tempfile
import cv2
import streamlit as st
from sort import Sort
from models import CLASSES, FILES, load_model, load_accident_model, detect_small, count_vehicles, accident_probability, accident_seen
from traffic import *

st.set_page_config(page_title="Smart Traffic Control System", layout="wide")
s = st.session_state
get_model = st.cache_resource(load_model)                 # models are loaded only once
get_accident_model = st.cache_resource(load_accident_model)
full_screen = s.get("full", False)

page_style(full_screen)
if not full_screen:
    center(st, "Smart Traffic Control System", "h1")

# ---------- first page: upload the videos ----------
if "caps" not in s:
    videos = st.columns([1, 2, 1])[1].file_uploader("Upload 4 lane videos", type=["mp4", "avi"], accept_multiple_files=True)
    if len(videos) != 4:
        st.stop()
    s.caps = []
    for v in videos:
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.write(v.read())
        s.caps.append(cv2.VideoCapture(f.name))
    s.trackers = [Sort() for i in range(4)]
    s.counts = [{c: 0 for c in CLASSES} for i in range(4)]
    s.frames = [None] * 4
    s.accident, s.crash, s.prob = [False] * 4, [0] * 4, [0] * 4
    s.history = []
    s.current, s.state, s.green_time, s.switch = 0, "GREEN", MIN_GREEN, time.time()
    st.rerun()

# ---------- controls ----------
s.prob = s.get("prob", [0] * 4)           # accident probability of each lane
col = st.columns([1, 2, 1, 1, 1, 1, 1])
model_name = col[1].selectbox("Model", list(FILES) + ["Combine All"], label_visibility="collapsed")
if col[2].button("Next Lane ➤", width="stretch"):
    s.state, s.switch = "YELLOW", 0         # yellow time is already over, so the next lane turns green at once
if col[3].button("Force Red 🔴", width="stretch"):
    s.state = "ALL RED"
if col[4].button("Stop Alert ⚠️", width="stretch"):
    s.accident = [False] * 4
col[5].toggle("Full screen", key="full")    

model = {n: get_model(n) for n in FILES} if model_name == "Combine All" else get_model(model_name)
accident_model = get_accident_model()

# ---------- page ----------
status = st.empty()
if full_screen:                             # only the 4 videos
    video_box, chart_box = st.empty(), None
else:
    live_tab, dash_tab = st.tabs(["LIVE 🎥","DASHBOARD 📊"])
    video_box = live_tab.columns([1, 5, 1])[1].empty()
    chart_box = dash_tab.columns([1, 4, 1])[1].empty()

# ---------- main loop ----------
step = 0
while True:
    # signal timing: green -> yellow -> next lane green
    passed = time.time() - s.switch
    if s.state == "GREEN" and passed > s.green_time:
        s.state, s.switch = "YELLOW", time.time()
    elif s.state == "YELLOW" and passed > YELLOW_TIME:
        s.current = (s.current + 1) % 4
        s.state, s.switch = "GREEN", time.time()
        s.green_time = green_time(s.counts[s.current])

    left = max(0, int(s.green_time - passed)) if s.state == "GREEN" else 0
    text = "All lanes RED" if s.state == "ALL RED" else f"Lane {s.current + 1} is {s.state} | Time Left: {left}s"
    center(status, text + (" | ACCIDENT CHANCE" if True in s.accident else ""), "h3")

    pictures = []
    for i in range(4):
        light = s.state if (i == s.current and s.state != "ALL RED") else "RED"
        # only the green lane plays, the other lanes stay on their last frame
        if light == "GREEN" or s.frames[i] is None:
            frame, found = detect_small(model_name, model, read_frame(s.caps[i]))
            tracks = s.trackers[i].update([d[:4] for d in found])
            s.counts[i] = count_vehicles(found)
            # accident alert when an accident is seen 3 times in a row
            s.prob[i] = accident_probability(accident_model, frame)
            s.crash[i] = s.crash[i] + 1 if accident_seen(s.prob[i], tracks) else 0
            s.accident[i] = s.accident[i] or s.crash[i] >= 3
            draw_boxes(frame, found)
            s.frames[i] = frame
        pictures.append(lane_picture(i, s.frames[i], light, left, s.counts[i], s.trackers[i].total, s.accident[i]))
    video_box.image(make_grid(pictures), channels="BGR", width="stretch")

    # dashboard charts, drawn every 5 steps
    s.history = (s.history + [[sum(c.values()) for c in s.counts]])[-30:]
    if chart_box is not None and step % 5 == 0:
        show_dashboard(chart_box, s.counts, s.history, s.prob)
    step += 1
    time.sleep(0.01)