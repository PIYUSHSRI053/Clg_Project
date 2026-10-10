# Signal timing, video frames and the dashboard charts
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from models import CLASSES

MIN_GREEN = 5
MAX_GREEN = 30
YELLOW_TIME = 1.5
SKIP_FRAMES = 10       # read every 5th frame so the video plays faster
COLORS = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 200, 0)}


# green time of a lane. Big vehicles need more time to cross than bikes,
# so every type has its own weight.
def green_time(c):
    four_wheelers = c["car"] + c["bus"] + c["truck"]
    seconds = MIN_GREEN + 0.5 * c["motorcycle"] + 1 * c["car"] + 2 * c["bus"] + 3 * c["truck"]
    if four_wheelers >= 8 and four_wheelers > c["motorcycle"]:
        seconds = max(seconds, 16)          # mostly 4 wheelers: give more than 15 seconds
    return int(max(MIN_GREEN, min(MAX_GREEN, seconds)))


def read_frame(cap):
    for i in range(SKIP_FRAMES):
        ok, frame = cap.read()
        if not ok:                          # video is over, start it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
    return cv2.resize(frame, (960, 540))     # big frame for detection, it is made small later


def draw_boxes(frame, found):
    for d in found:
        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)
        cv2.putText(frame, d[4], (d[0], max(12, d[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# one lane picture: traffic light and timer on the left, video, and a line of text below.
# The timer shows the seconds left for the green lane and -- for the others.
def lane_picture(i, frame, light, left, c, passed, accident):
    text = f"Lane {i + 1}: {light}  Total {sum(c.values())}  Car {c['car']}  Bike {c['motorcycle']}  Bus {c['bus']}  Truck {c['truck']}  Passed {passed}"
    out = cv2.copyMakeBorder(frame, 0, 34, 70, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for k, name in enumerate(["RED", "YELLOW", "GREEN"]):
        color = COLORS[name] if name == light else (90, 90, 90)
        cv2.circle(out, (35, 60 + k * 56), 24, color, -1)
    timer = f"{left}s" if light == "GREEN" else "--"
    cv2.rectangle(out, (6, 228), (64, 262), (255, 255, 255), -1)
    cv2.putText(out, timer, (35 - 7 * len(timer), 252), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # warning sign on the video: white outline normally, filled yellow when there is an accident
    sign = np.array([[500, 44], [540, 44], [520, 10]])
    if accident:
        cv2.rectangle(out, (0, 270), (550, 304), (0, 0, 255), -1)
        cv2.fillPoly(out, [sign], (0, 255, 255))
        text += "  ACCIDENT"
    cv2.polylines(out, [sign], True, (0, 0, 0) if accident else (255, 255, 255), 2)
    cv2.putText(out, "!", (514, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0) if accident else (255, 255, 255), 2)
    cv2.putText(out, text, (8, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out

# shows a text in the middle of the page
def center(place, text, size):
    place.markdown(f"<{size} style='text-align:center'>{text}</{size}>", unsafe_allow_html=True)

# tabs in the middle of the page. In full screen the videos fill the window.
def page_style(full_screen):
    css = "div[role='tablist'] {justify-content: center}"
    if full_screen:
        css += ".block-container {padding: 3rem 1rem 0 1rem} img {max-height: 74vh; object-fit: contain,hover: scale-down}"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# joins the 4 lane pictures into one big picture (2 x 2)
def make_grid(pictures):
    gap_v = np.full((304, 10, 3), 255, np.uint8)
    top = np.hstack([pictures[0], gap_v, pictures[1]])
    bottom = np.hstack([pictures[2], gap_v, pictures[3]])
    gap_h = np.full((10, top.shape[1], 3), 255, np.uint8)
    return np.vstack([top, gap_h, bottom])


# the 4 charts of the dashboard. The accident probability bar turns red above 60%.
def show_dashboard(place, counts, history, prob):
    lanes = ["L1", "L2", "L3", "L4"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    for i in range(4):
        ax[0][0].plot([h[i] for h in history], label=lanes[i])
    ax[0][0].set_title("Traffic Trend")
    ax[0][0].legend()
    ax[0][1].bar(lanes, [sum(c.values()) for c in counts])
    ax[0][1].set_title("Lane Load")
    ax[1][0].bar(CLASSES, [sum(c[v] for c in counts) for v in CLASSES])
    ax[1][0].set_title("Vehicle Types")
    ax[1][1].bar(lanes, [p * 100 for p in prob], color=["red" if p > 0.6 else "tab:blue" for p in prob])
    ax[1][1].axhline(60, color="gray", linestyle="--")
    ax[1][1].set(ylim=(0, 100), title="Accident Probability (%)")
    fig.tight_layout()
    place.pyplot(fig)
    plt.close(fig)