"""
Realtime Squat Counter & Quality Feedback
Final Year Project - Stable Version
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from collections import deque
import joblib
import os

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../models/squat_model.pkl")
OUTPUT_LOG_DIR = os.path.join(BASE_DIR, "../data")

FEATURE_NAMES = [
    "knee_angle_r", "knee_angle_l",
    "elbow_angle_r", "elbow_angle_l",
    "hip_angle_r", "hip_angle_l",
    "shoulder_angle_r", "shoulder_angle_l"
]

DOWN_THRESHOLD = 100
UP_THRESHOLD = 165
DEPTH_THRESHOLD = 90   # 🔥 key fix

SMOOTHING_WINDOW = 7
HOLD_TIME = 0.4
MIN_VISIBILITY = 0.4
FPS_LIMIT = 0.02
# -----------------------------------------

# Load model
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️ Model not found at {MODEL_PATH}")

# --- Helpers ---
# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def speak(text):
    pass

# --- State ---
rep_count = 0
state = "up"
last_state_change = time.time()
knee_buffer = deque(maxlen=SMOOTHING_WINDOW)
rep_log = []
feedback_msg = ""
min_knee_angle = 180   # 🔥 track depth

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Camera not available")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Visibility check
            vis_ids = [23,24,25,26,27,28]
            if np.mean([lm[i].visibility for i in vis_ids]) < MIN_VISIBILITY:
                cv2.putText(frame, "Adjust camera",
                            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,160,255), 2)
                cv2.imshow("Squat AI", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Joints
            hip_r, knee_r, ankle_r = (lm[24].x, lm[24].y), (lm[26].x, lm[26].y), (lm[28].x, lm[28].y)
            hip_l, knee_l, ankle_l = (lm[23].x, lm[23].y), (lm[25].x, lm[25].y), (lm[27].x, lm[27].y)

            shoulder_r, elbow_r, wrist_r = (lm[12].x, lm[12].y), (lm[14].x, lm[14].y), (lm[16].x, lm[16].y)
            shoulder_l, elbow_l, wrist_l = (lm[11].x, lm[11].y), (lm[13].x, lm[13].y), (lm[15].x, lm[15].y)

            # Angles
            knee_angle_r = calculate_angle(hip_r, knee_r, ankle_r)
            knee_angle_l = calculate_angle(hip_l, knee_l, ankle_l)
            elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            hip_angle_r = calculate_angle(shoulder_r, hip_r, knee_r)
            hip_angle_l = calculate_angle(shoulder_l, hip_l, knee_l)
            shoulder_angle_r = calculate_angle(elbow_r, shoulder_r, hip_r)
            shoulder_angle_l = calculate_angle(elbow_l, shoulder_l, hip_l)

            # Smoothing
            avg_knee = (knee_angle_r + knee_angle_l) / 2
            knee_buffer.append(avg_knee)
            smooth_knee = float(np.median(knee_buffer))

            # 🔥 Track minimum depth
            min_knee_angle = min(min_knee_angle, smooth_knee)

            # ML Prediction + Confidence
            quality = "GOOD"
            confidence = 1.0

            if model:
                features = pd.DataFrame([[
                    knee_angle_r, knee_angle_l,
                    elbow_angle_r, elbow_angle_l,
                    hip_angle_r, hip_angle_l,
                    shoulder_angle_r, shoulder_angle_l
                ]], columns=FEATURE_NAMES)

                pred = model.predict(features)[0]

                # Try getting confidence if model supports it
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features)[0]
                    confidence = float(max(proba))
                else:
                    confidence = 0.7  # fallback assumption

                quality = "GOOD" if int(pred) == 1 else "BAD"

            # Rep logic
            now = time.time()

            if state == "up" and smooth_knee < DOWN_THRESHOLD:
                if now - last_state_change > HOLD_TIME:
                    state = "down"
                    last_state_change = now

            elif state == "down" and smooth_knee > UP_THRESHOLD:
                if now - last_state_change > HOLD_TIME:

                    if rep_log and (now - rep_log[-1]["time"]) < 0.8:
                        pass
                    else:
                        state = "up"
                        last_state_change = now

                        # 🔥 STRICT VALIDATION (depth + posture + symmetry)
                        valid_depth = min_knee_angle < DEPTH_THRESHOLD
                        good_symmetry = abs(knee_angle_r - knee_angle_l) < 20
                        good_back = hip_angle_r > 60 and hip_angle_l > 60
                        confident = confidence > 0.6

                        if valid_depth and good_symmetry and good_back and confident:

                            issues = []

                            if not valid_depth:
                                issues.append("Go deeper")

                            if not good_symmetry:
                                issues.append("Balance your legs")

                            if not good_back:
                                issues.append("Keep back straight")

                            if quality == "BAD" and not issues:
                                issues.append("Form needs improvement")

                            feedback_msg = issues[0] if issues else "Good form"

                            rep_count += 1

                            rep_log.append({
                                "time": now,
                                "exercise": "squat",
                                "rep": rep_count,
                                "quality": quality,
                                "is_good": quality == "GOOD",
                                "feedback": feedback_msg,
                                "confidence": round(confidence, 2),
                                "knee_angle": min_knee_angle
                            })
                        else:
                            feedback_msg = "Go deeper"

                        # Reset depth tracker
                        min_knee_angle = 180

            # Draw
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (frame.shape[1]-350, 20), (frame.shape[1]-20, 230), (50,50,50), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, f"REPS: {rep_count}", (frame.shape[1]-300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)

            cv2.putText(frame, f"FORM: {quality}", (frame.shape[1]-300, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0,255,0) if quality=="GOOD" else (0,0,255), 2)

            text_size = cv2.getTextSize(feedback_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = int((frame.shape[1] - text_size[0]) / 2)

            cv2.putText(frame, feedback_msg,
                        (text_x, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0,255,0) if feedback_msg == "Good form" else (0,0,255),
                        3)

        cv2.imshow("Squat AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(FPS_LIMIT)

finally:
    cap.release()
    cv2.destroyAllWindows()

    if rep_log:
        os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_LOG_DIR, f"squat_log_{int(time.time())}.csv")
        pd.DataFrame(rep_log).to_csv(out_path, index=False)
        print(f"SESSION_LOG:{out_path}")

    print(f"✅ Session ended. Total reps: {rep_count}")