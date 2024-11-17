import streamlit as st
import cv2
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import os

# Title
st.title("Video Analysis and Object Detection")

# Step 1: Upload Video
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        input_video_path = temp_video.name

    # Step 2: Process the Video
    def process_video(input_path):
        # Initialize YOLO model
        model = YOLO('yolov10m.pt')  # Ensure the correct model file path
        class_list = model.names

        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError("Error opening video file")

        # Video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Video Properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

        # Output video writer
        output_video_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Ensure compatible codec
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Tracking data
        tracking_timestamps = {}
        disappearance_counts = {}
        debounce_limit = 10
        tracking_data_json = []

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            timestamp = frame_number / fps  # Calculate timestamp

            # YOLO Inference
            results = model.predict(frame)
            detected_ids = []

            if results and results[0].boxes is not None:
                for box in results[0].boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = box
                    if conf > 0.5 and int(cls) < len(class_list):
                        label = class_list[int(cls)]
                        object_id = int(cls)

                        detected_ids.append(object_id)
                        if object_id not in tracking_timestamps:
                            tracking_timestamps[object_id] = {
                                "class": label,
                                "start_time": timestamp,
                                "last_time": timestamp
                            }
                            disappearance_counts[object_id] = 0
                            # Add object immediately to JSON
                            tracking_data_json.append({
                                "name": label,
                                "trackId": object_id,
                                "startTime": f"{timestamp:.2f}s",
                                "endTime": f"{timestamp:.2f}s"
                            })
                        else:
                            tracking_timestamps[object_id]["last_time"] = timestamp
                            disappearance_counts[object_id] = 0

                            # Update JSON entry for existing object
                            for obj in tracking_data_json:
                                if obj["trackId"] == object_id:
                                    obj["endTime"] = f"{timestamp:.2f}s"

                        # Draw bounding box and label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Handle undetected objects
            for obj_id in list(tracking_timestamps.keys()):
                if obj_id not in detected_ids:
                    disappearance_counts[obj_id] += 1
                    if disappearance_counts[obj_id] > debounce_limit:
                        # Finalize object tracking
                        tracking_timestamps.pop(obj_id, None)
                        disappearance_counts.pop(obj_id, None)

            out.write(frame)

        # Save JSON
        json_path = os.path.join(tempfile.gettempdir(), "tracking_data_output.json")
        with open(json_path, "w") as f:
            json.dump(tracking_data_json, f, indent=4)

        # Release resources
        cap.release()
        out.release()

        return output_video_path, json_path

    output_video_path, json_file_path = process_video(input_video_path)

    # Step 3: Display Results
    st.subheader("Output Video")
    if os.path.exists(output_video_path):
        st.write("Video file successfully created!")
        with open(output_video_path, "rb") as f:
            st.video(f)
    else:
        st.write("Video file not found at:", output_video_path)

    st.subheader("Timeline Graph")
    with open(json_file_path, "r") as f:
        tracking_data = json.load(f)

    # Generate Timeline Graph
    unique_objects = sorted(set([item["name"] for item in tracking_data]))
    object_intervals = []
    object_names = []

    for obj in tracking_data:
        start_time = float(obj["startTime"][:-1])  # Strip "s" and convert to float
        end_time = float(obj["endTime"][:-1])  # Strip "s" and convert to float
        object_intervals.append((start_time, end_time))
        object_names.append(obj["name"])

    name_to_idx = {name: idx for idx, name in enumerate(unique_objects)}
    colors = plt.cm.get_cmap("tab20", len(unique_objects))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (start, end) in enumerate(object_intervals):
        idx = name_to_idx[object_names[i]]
        ax.broken_barh([(start, end - start)], (idx * 2 - 0.4, 0.8), facecolors=colors(idx))

    ax.set_yticks([name_to_idx[name] * 2 for name in unique_objects])
    ax.set_yticklabels(unique_objects)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Objects")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_title("Object Detection Timeline")
    st.pyplot(fig)
