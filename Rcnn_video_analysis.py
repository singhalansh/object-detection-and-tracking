import streamlit as st
import cv2
import json
import tempfile
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# Title
st.title("Video Analysis and Object Detection with Faster R-CNN")

# Load pre-trained Faster R-CNN model
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device

# Image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
])

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        input_video_path = temp_video.name

    def process_video(input_path):
        try:
            model, device = load_model()
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise IOError("Error opening video file")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.write(f"Video Properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

            output_video_path = os.path.join(os.getcwd(), "output_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            frame_number = 0
            confidence_threshold = 0.5
            progress_bar = st.progress(0)
            
            tracking_data_json = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                timestamp = frame_number / fps
                progress_bar.progress(frame_number / total_frames)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    predictions = model(img_tensor)[0]
                
                detected_objects = []
                for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
                    if score > confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.cpu().numpy())
                        class_name = f"Object {label.item()}"
                        detected_objects.append({
                            "name": class_name,
                            "startTime": f"{timestamp:.2f}s",
                            "confidence": f"{score:.2f}"
                        })
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                tracking_data_json.extend(detected_objects)
                out.write(frame)
            
            cap.release()
            out.release()
            progress_bar.empty()
            
            json_path = os.path.join(os.getcwd(), "tracking_data_output.json")
            with open(json_path, "w") as f:
                json.dump(tracking_data_json, f, indent=4)
            
            return output_video_path, json_path
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            raise e

    output_video_path, json_file_path = process_video(input_video_path)

    st.subheader("Output Video")
    if os.path.exists(output_video_path):
        st.video(output_video_path)
    
    st.subheader("Detected Objects")
    with open(json_file_path, "r") as f:
        tracking_data = json.load(f)
    st.json(tracking_data)
