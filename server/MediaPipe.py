import pickle
import cv2
import base64
import time
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import warnings
import csv
import os
from flask_cors import CORS

warnings.filterwarnings(
    "ignore", 
    message="X does not have valid feature names, but RobustScaler was fitted with feature names"
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Create Flask and WebSocket (SocketIO) server
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin requests

with open(r'C:\Users\rhima\HackathonCS6\server\models\deadlift.pkl', 'rb') as f:
    model = pickle.load(f, encoding='bytes')

# Landmarking
feature_columns = [f'{coord}{i}' for i in range(1, 34) for coord in ['x', 'y', 'z', 'v']]

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Global flag to ensure only one streaming task runs
is_streaming = False
show_angles = False

# Limited Starting
@app.route('/start_camera/<variant>', methods=['GET'])
def start_camera(variant):
    global is_streaming, show_angles
    if variant == 'angle':
        show_angles = True
    else:
        show_angles = False
    if not is_streaming:
        socketio.start_background_task(stream_video)
        is_streaming = True
        return jsonify({'status': 'Camera streaming started'}), 200
    else:
        return jsonify({'status': 'Camera streaming already running'}), 200

# Global Stopping
@app.route('/stop_camera', methods=['GET'])
def stop_camera():
    global is_streaming
    is_streaming = False
    return jsonify({'status': 'Camera streaming stopped'}), 200

def generate_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    predicted_stage = None 
    global is_streaming
    predicted_class = None

    with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            if not is_streaming:
                print("is_streaming is False, breaking out of the loop")
                break

            ret, frame = cap.read()

            # Convert frame from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Improve performance.
            results = pose.process(image)
            image.flags.writeable = True   # Allow drawing.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            if results.pose_landmarks:
                # Extract keypoints for 33 landmarks (x, y, z, visibility) -> 132 features.
                keypoints = np.array([
                    [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in results.pose_landmarks.landmark
                ]).flatten()

                if keypoints.shape[0] != 132:
                    print("Unexpected keypoint vector length:", keypoints.shape[0])
                    continue

                keypoints = keypoints.reshape(1, -1) # Reshape to (1, 33 * 4)
                keypoints = pd.DataFrame(keypoints, columns=feature_columns)

                # Get prediction from the model.
                try:
                    predicted_class = model.predict(keypoints)[0]
                except Exception as e:
                    print("Error during prediction:", e)
                    continue

                # Get confidence value if available.
                confidence_value = 0.0
                probabilities = model.predict_proba(keypoints)
                confidence_value = np.max(probabilities)

                # Count a rep when transitioning from 'up' to 'down'
                if predicted_class == 'deadlift: down' and predicted_stage == 'deadlift: up':
                    counter += 1
                    print("Rep counted! Total reps:", counter)

                if predicted_class == 'deadlift: lean left' and predicted_stage == 'deadlift: lean right':
                    counter += 1
                    print("Re-center yourself:", counter)

                predicted_stage  = predicted_class

                if show_angles:
                    landmarks = results.pose_landmarks.landmark
            
                    # Extract the x,y coordinates for the left hip, knee, and ankle
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    # Calculate the angle at the left knee
                    angle = calculate_angle(left_hip, left_knee, left_ankle)
                        
                    knee_coords = tuple(np.multiply(left_knee, [640, 480]).astype(int))
                    cv2.putText(
                        image,
                        str(int(angle)),              
                        knee_coords,                    
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,                       
                        (255, 255, 255),             
                        2,                       
                        cv2.LINE_AA
                    )
            
            # encode frame as jpeg to be transmitted to frontend
            ret, buffer = cv2.imencode('.jpg', image)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')

            # Yield a dictionary with all the information.
            yield {
                'image': frame_encoded,
                'counter': counter,
                'predicted_class': predicted_class,
                'confidence': confidence_value
            }         

    cap.release()
    cv2.destroyAllWindows()

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('message', {'data': 'Connected to MediaPipe backend'})

def stream_video():
    for frame_data in generate_frames():
        socketio.emit('frame', frame_data)
        socketio.sleep(0.03)  # roughly 33 fps

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

