import streamlit as st
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tempfile
import os

# Load models
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")
    try:
        lstm_model = load_model("lstm_crowd_behavior.h5")
    except Exception as e:
        st.error(f"Error: Could not load lstm_crowd_behavior.h5: {e}. Please train the model first.")
        return None, None
    return yolo_model, lstm_model

yolo_model, lstm_model = load_models()
if yolo_model is None or lstm_model is None:
    st.stop()

# Load label encoder classes
try:
    label_encoder_classes = np.load("label_encoder_classes.npy", allow_pickle=True)
except Exception as e:
    st.error(f"Error: Could not load label_encoder_classes.npy: {e}")
    st.stop()

tracker = sv.ByteTrack()
sequence_length = 10

# Streamlit UI
st.title("AI-Powered Crowd Behavior Predictor")
st.write("Analyze crowd behavior (Calm, Aggressive, Dispersing, Stampede) using YOLOv8 and LSTM with real-time drone/CCTV footage or pre-recorded video.")

# Input selection
input_option = st.radio("Select Input Source:", ("Real-Time Drone/CCTV Feed", "Pre-Recorded Video"))

# Function to process frame
def process_frame(frame, prev_positions, density_history, speed_history, time_history, frame_count):
    # YOLOv8 inference
    results = yolo_model(frame)
    filtered_boxes = [box for box in results[0].boxes if int(box.cls) == 0]  # Class 0 = person

    # Handle case where no detections are found
    if len(filtered_boxes) == 0:
        detections = sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),  # Empty 2D array with shape (0, 4)
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=np.int32)
        )
    else:
        detections = sv.Detections(
            xyxy=np.array([box.xyxy[0].cpu().numpy() for box in filtered_boxes]),
            confidence=np.array([box.conf[0].cpu().numpy() for box in filtered_boxes]),
            class_id=np.array([0] * len(filtered_boxes))
        )

    # Track detections
    tracked_detections = tracker.update_with_detections(detections)
    annotated_frame = frame.copy()

    # Calculate density and movement
    num_people = len(tracked_detections)
    frame_area = frame.shape[0] * frame.shape[1]
    density = num_people / frame_area * 10000 if frame_area > 0 else 0

    speeds = []
    for box, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
        x1, y1, x2, y2 = map(int, box)
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        if prev_positions[track_id] is not None:
            prev_x, prev_y = prev_positions[track_id]
            speed = np.sqrt((centroid[0] - prev_x)**2 + (centroid[1] - prev_y)**2)
            speeds.append(speed)
        prev_positions[track_id] = centroid
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0

    # Rule-based behavior classification
    if density > 0.015 and avg_speed > 25 and speed_variance > 50:
        rule_behavior = "Stampede"
    elif density > 0.01 and avg_speed > 20:
        rule_behavior = "Aggressive"
    elif density < 0.005 and avg_speed > 10:
        rule_behavior = "Dispersing"
    else:
        rule_behavior = "Calm"

    # LSTM prediction
    lstm_behavior = "Unknown"
    if len(density_history) >= sequence_length - 1:
        density_history.append(density)
        speed_history.append(avg_speed)
        sequence = np.array(list(zip(density_history[-sequence_length:], 
                                   speed_history[-sequence_length:])))
        sequence = sequence.reshape(1, sequence_length, 2)
        pred = lstm_model.predict(sequence, verbose=0)
        lstm_behavior = label_encoder_classes[np.argmax(pred)]
    else:
        density_history.append(density)
        speed_history.append(avg_speed)

    # Store data
    frame_count += 1
    time_history.append(frame_count)

    # Annotate frame
    cv2.putText(annotated_frame, f"Rule-Based: {rule_behavior}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"LSTM Pred: {lstm_behavior}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(annotated_frame, f"People: {num_people}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Density: {density:.4f}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Speed Variance: {speed_variance:.2f}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Alerts
    if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
        cv2.putText(annotated_frame, "ALERT: Aggressive Behavior Detected", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
        cv2.putText(annotated_frame, "ALERT: Stampede Detected!", (10, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Generate plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('Crowd Analysis Trends')
    ax1.plot(time_history, density_history, 'b-', label='Density')
    ax1.set_title('Density Over Time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('People/10k pixels')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_history, speed_history, 'r-', label='Speed')
    ax2.set_title('Average Speed Over Time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Pixels/Frame')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    return annotated_frame, num_people, density, avg_speed, rule_behavior, lstm_behavior, fig, frame_count

# Real-Time Drone/CCTV Feed
if input_option == "Real-Time Drone/CCTV Feed":
    st.write("Using real-time feed from camera (default: webcam). Enter an RTSP URL for drone/CCTV if needed.")
    rtsp_url = st.text_input("RTSP URL (leave blank for webcam)", "")
    video_source = rtsp_url if rtsp_url else 0  # Default to webcam (0) if no RTSP URL

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Error: Could not open real-time feed. Check your camera or RTSP URL.")
        st.stop()

    # Initialize data storage
    prev_positions = defaultdict(lambda: None)
    density_history = []
    speed_history = []
    time_history = []
    frame_count = 0

    # Placeholders
    video_placeholder = st.empty()
    plot_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Start/Stop button
    if 'running' not in st.session_state:
        st.session_state.running = False

    if st.button("Start Real-Time Analysis"):
        st.session_state.running = True
    if st.button("Stop Real-Time Analysis"):
        st.session_state.running = False

    if st.session_state.running:
        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("End of feed or error reading frame.")
                break

            annotated_frame, num_people, density, avg_speed, rule_behavior, lstm_behavior, fig, frame_count = process_frame(
                frame, prev_positions, density_history, speed_history, time_history, frame_count
            )

            # Display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, caption=f"Frame {frame_count}", use_container_width=True)
            plot_placeholder.pyplot(fig)
            metrics_placeholder.write(f"Frame {frame_count}: People: {num_people}, Density: {density:.4f}, "
                                     f"Avg Speed: {avg_speed:.2f}, Rule-Based: {rule_behavior}, LSTM: {lstm_behavior}")

            # Show warnings
            if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
                st.warning("Aggressive behavior detected!")
            if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
                st.error("Stampede detected! Immediate action recommended.")

            plt.close(fig)  # Close figure to free memory

    cap.release()

# Pre-Recorded Video
elif input_option == "Pre-Recorded Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video.")
            st.stop()

        # Initialize data storage
        prev_positions = defaultdict(lambda: None)
        density_history = []
        speed_history = []
        time_history = []
        frame_count = 0

        # Placeholders
        video_placeholder = st.empty()
        plot_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # Process video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, num_people, density, avg_speed, rule_behavior, lstm_behavior, fig, frame_count = process_frame(
                frame, prev_positions, density_history, speed_history, time_history, frame_count
            )

            # Display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, caption=f"Frame {frame_count}", use_container_width=True)
            plot_placeholder.pyplot(fig)
            metrics_placeholder.write(f"Frame {frame_count}: People: {num_people}, Density: {density:.4f}, "
                                     f"Avg Speed: {avg_speed:.2f}, Rule-Based: {rule_behavior}, LSTM: {lstm_behavior}")

            # Show warnings
            if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
                st.warning("Aggressive behavior detected!")
            if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
                st.error("Stampede detected! Immediate action recommended.")

            plt.close(fig)

        cap.release()
        os.unlink(video_path)  # Clean up temporary file
        st.success("Video processing completed!")
    else:
        st.write("Please upload a video file to begin analysis.")