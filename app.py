import cv2
import torch
import numpy as np
import tempfile
import streamlit as st
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection, draw_working_areas


# ---------------- DEVICE & MODEL ----------------
@st.cache_resource
def setup_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolov8n.pt")
    model.to(device)
    model.nms = 0.5
    return model, device


# ---------------- UTILS ----------------
def calculate_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def track_time(obj_id, index, entry_time, time_in_area, frame_duration):
    if obj_id not in entry_time:
        entry_time[obj_id] = index
    else:
        prev_index = entry_time[obj_id]
        time_in_area[prev_index] += frame_duration
        entry_time[obj_id] = index


def process_frame(model, frame, working_area, time_in_area, entry_time, frame_duration):
    boxes, classes, names, confs, ids = YOLO_Detection(
        model, frame, conf=0.05, mode="track"
    )

    polygon_detections = [False] * len(working_area)

    for box, cls, obj_id in zip(boxes, classes, ids):
        center = calculate_center(box)

        label_detection(
            frame=frame,
            text=f"{names[int(cls)]}, ID:{int(obj_id)}",
            tbox_color=(255, 144, 30),
            left=box[0], top=box[1], bottom=box[2], right=box[3]
        )

        for idx, area in enumerate(working_area):
            if cv2.pointPolygonTest(np.array(area, np.int32), center, False) >= 0:
                polygon_detections[idx] = True
                track_time(obj_id, idx, entry_time, time_in_area, frame_duration)

    for idx, area in enumerate(working_area):
        color = (0, 255, 0) if polygon_detections[idx] else (0, 0, 255)
        draw_working_areas(frame, area, idx, color)

    return frame


# ---------------- STREAMLIT APP ----------------
def main():
    st.set_page_config(page_title="YOLO Working Area Tracker", layout="wide")
    st.title("🎯 YOLOv8 Working Area Time Tracking")

    model, device = setup_model()
    st.success(f"Using device: {device}")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        working_area = [
            [(499, 41), (384, 74), (377, 136), (414, 193), (417, 112), (548, 91)],
            [(547, 91), (419, 113), (414, 189), (452, 289), (453, 223), (615, 164)],
            [(158, 84), (294, 85), (299, 157), (151, 137)],
            [(151, 139), (300, 155), (321, 251), (143, 225)],
            [(143, 225), (327, 248), (351, 398), (142, 363)],
            [(618, 166), (457, 225), (454, 289), (522, 396), (557, 331), (698, 262)]
        ]

        time_in_area = {i: 0 for i in range(len(working_area))}
        entry_time = {}
        frame_duration = 1 / cap.get(cv2.CAP_PROP_FPS)

        stframe = st.empty()
        stat_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(
                model, frame, working_area,
                time_in_area, entry_time, frame_duration
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            stat_box.markdown("### ⏱ Time Spent Per Area")
            for k, v in time_in_area.items():
                stat_box.write(f"Area {k+1}: {round(v, 2)} seconds")

        cap.release()
        st.success("Video processing completed!")


if __name__ == "__main__":
    main()
