# AI-Powered Crowd Behavior Predictor

## Overview
This project is an AI-powered crowd behavior prediction system that analyzes pre-recorded video footage to detect potential panic situations before they occur. It utilizes OpenCV, TensorFlow, YOLOv8, and an LSTM-based model to analyze crowd movement patterns and generate alerts.

## Features
- **Crowd Behavior Analysis**: Uses YOLOv8 for object detection and LSTM for time-series prediction.
- **Dataset Handling**: Prepares and encodes crowd behavior data.
- **Model Training**: LSTM-based deep learning model for predicting behavior trends.
- **Web Interface**: Interactive Streamlit dashboard for visualizing predictions.
- **Pre-trained YOLO Model**: Uses `yolov8n.pt` for object detection.

## Project Structure
```
📂 AI-Powered Crowd Behavior Predictor
│── analyze_crowd.py         # Streamlit web interface for analysis
│── crowd_data.csv           # Dataset containing crowd behavior data
│── label_encoder_classes.npy # Encoded labels for classification
│── lstm_crowd_behavior.py   # LSTM model for predicting crowd behavior
│── prepare_lstm_data.py     # Data preprocessing and feature engineering
│── requirements.txt         # List of required Python packages
│── train_lstm.py            # Training script for LSTM model
│── X_train.npy, X_test.npy  # Processed training and test datasets
│── y_train.npy, y_test.npy  # Processed labels for training/testing
│── yolov8n.pt               # YOLOv8 pre-trained weights
```

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed. Then, install dependencies:
```sh
pip install -r requirements.txt
```

### Running the Project
1. **Prepare Data**:
   ```sh
   python prepare_lstm_data.py
   ```
2. **Train the LSTM Model**:
   ```sh
   python train_lstm.py
   ```
3. **Run the Streamlit Web App**:
   ```sh
   streamlit run analyze_crowd.py
   ```
4. **Access the Dashboard**:
   Open [http://localhost:8501](http://localhost:8501) in your browser.

## Model Details
- **YOLOv8**: Used for object detection in video frames.
- **LSTM Model**: Predicts crowd behavior based on time-series movement patterns.

![image alt](https://github.com/suman2807/crowd_behavior_analysis/blob/main/Screenshot%202025-03-26%20230719.png?raw=true)
![image alt](https://github.com/suman2807/crowd_behavior_analysis/blob/main/Screenshot%202025-03-26%20230838.png?raw=true)


