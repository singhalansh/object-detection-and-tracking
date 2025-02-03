import streamlit as st
import cv2
import json
import matplotlib.pyplot as plt
import tempfile
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# Title
st.title("Video Analysis and Object Detection")

def load_model():
    """Load pre-trained SSD model from torchvision"""
    # Load pre-trained model
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # COCO class labels
    CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    return model, CLASSES, device

# Image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Step 1: Upload Video
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        input_video_path = temp_video.name

    # Step 2: Process the Video
    def process_video(input_path):
        try:
            # Initialize model
            model, class_list, device = load_model()
            
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
            output_video_path = os.path.join(os.getcwd(), "output_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            # Tracking data structures
            tracking_timestamps = {}
            disappearance_counts = {}
            debounce_limit = 10
            tracking_data_json = []

            frame_number = 0
            confidence_threshold = 0.7

            # Progress bar
            progress_bar = st.progress(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                timestamp = frame_number / fps

                # Update progress bar
                progress = frame_number / total_frames
                progress_bar.progress(progress)

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                # Transform image
                img_tensor = transform(pil_image)
                # Add batch dimension
                img_tensor = img_tensor.unsqueeze(0)
                # Move to device
                img_tensor = img_tensor.to(device)

                # Get predictions
                with torch.no_grad():
                    predictions = model(img_tensor)[0]

                detected_ids = []

                # Process detected objects
                for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
                    if score > confidence_threshold:
                        box = box.cpu().numpy()
                        label = label.cpu().item()
                        score = score.cpu().item()
                        
                        if label < len(class_list):
                            x1, y1, x2, y2 = box.astype(int)
                            class_name = class_list[label]
                            
                            if class_name != 'N/A':  # Skip undefined classes
                                detected_ids.append(label)
                                if label not in tracking_timestamps:
                                    tracking_timestamps[label] = {
                                        "class": class_name,
                                        "trackId": label,
                                        "startTime": timestamp,
                                        "lastSeenTime": timestamp,
                                        "detectedFrames": 1
                                    }
                                    disappearance_counts[label] = 0
                                else:
                                    tracking_timestamps[label]["lastSeenTime"] = timestamp
                                    tracking_timestamps[label]["detectedFrames"] += 1
                                    disappearance_counts[label] = 0

                                # Draw bounding box and label
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Handle objects not detected in the current frame
                for obj_id in list(tracking_timestamps.keys()):
                    if obj_id not in detected_ids:
                        disappearance_counts[obj_id] += 1
                        if disappearance_counts[obj_id] > debounce_limit:
                            tracking_data_json.append({
                                "name": tracking_timestamps[obj_id]["class"],
                                "trackId": tracking_timestamps[obj_id]["trackId"],
                                "startTime": f"{tracking_timestamps[obj_id]['startTime']:.2f}s",
                                "endTime": f"{tracking_timestamps[obj_id]['lastSeenTime']:.2f}s",
                                "duration": f"{(tracking_timestamps[obj_id]['lastSeenTime'] - tracking_timestamps[obj_id]['startTime']):.2f}s",
                                "totalFramesDetected": tracking_timestamps[obj_id]["detectedFrames"]
                            })
                            del tracking_timestamps[obj_id]
                            del disappearance_counts[obj_id]

                out.write(frame)

            # Finalize tracking data for remaining objects
            for obj_id in tracking_timestamps:
                tracking_data_json.append({
                    "name": tracking_timestamps[obj_id]["class"],
                    "trackId": tracking_timestamps[obj_id]["trackId"],
                    "startTime": f"{tracking_timestamps[obj_id]['startTime']:.2f}s",
                    "endTime": f"{tracking_timestamps[obj_id]['lastSeenTime']:.2f}s",
                    "duration": f"{(tracking_timestamps[obj_id]['lastSeenTime'] - tracking_timestamps[obj_id]['startTime']):.2f}s",
                    "totalFramesDetected": tracking_timestamps[obj_id]["detectedFrames"]
                })

            # Save JSON
            json_path = os.path.join(os.getcwd(), "tracking_data_output.json")
            with open(json_path, "w") as f:
                json.dump(tracking_data_json, f, indent=4)

            # Release resources
            cap.release()
            out.release()
            
            # Clear progress bar
            progress_bar.empty()

            return output_video_path, json_path
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            raise e

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
        start_time = float(obj["startTime"][:-1])
        end_time = float(obj["endTime"][:-1])
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