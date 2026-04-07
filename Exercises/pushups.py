import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import pandas as pd
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Thresholds
DOWN_THRESHOLD = 80
UP_THRESHOLD = 160
DEPTH_THRESHOLD = 85

SMOOTHING = 5

# State
state = "up"
rep_count = 0
angle_buffer = deque(maxlen=SMOOTHING)
min_elbow_angle = 180
feedback_msg = ""
rep_log = []

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Right side joints
        shoulder_r = (lm[12].x, lm[12].y)
        elbow_r = (lm[14].x, lm[14].y)
        wrist_r = (lm[16].x, lm[16].y)

        hip_r = (lm[24].x, lm[24].y)
        knee_r = (lm[26].x, lm[26].y)

        # Left side joints
        shoulder_l = (lm[11].x, lm[11].y)
        elbow_l = (lm[13].x, lm[13].y)
        wrist_l = (lm[15].x, lm[15].y)

        hip_l = (lm[23].x, lm[23].y)
        knee_l = (lm[25].x, lm[25].y)

        # Angles
        elbow_r_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)
        elbow_l_angle = calculate_angle(shoulder_l, elbow_l, wrist_l)

        hip_angle_r = calculate_angle(shoulder_r, hip_r, knee_r)
        hip_angle_l = calculate_angle(shoulder_l, hip_l, knee_l)

        # Smoothing
        avg_elbow = (elbow_r_angle + elbow_l_angle) / 2
        angle_buffer.append(avg_elbow)
        smooth_angle = np.mean(angle_buffer)

        min_elbow_angle = min(min_elbow_angle, smooth_angle)

        # Rep logic
        if state == "up" and smooth_angle < DOWN_THRESHOLD:
            state = "down"

        elif state == "down" and smooth_angle > UP_THRESHOLD:
            state = "up"

            # Strict validation
            valid_depth = min_elbow_angle < DEPTH_THRESHOLD
            good_balance = abs(elbow_r_angle - elbow_l_angle) < 20
            good_body = hip_angle_r > 150 and hip_angle_l > 150

            if valid_depth and good_balance and good_body:
                rep_count += 1
                feedback_msg = "Good form"

                rep_log.append({
                    "time": time.time(),
                    "exercise": "pushup",
                    "rep": rep_count,
                    "quality": "GOOD",
                    "is_good": True,
                    "feedback": feedback_msg
                })
            else:
                if not valid_depth:
                    feedback_msg = "Go lower"
                elif not good_body:
                    feedback_msg = "Keep body straight"
                else:
                    feedback_msg = "Balance arms"

                rep_log.append({
                    "time": time.time(),
                    "exercise": "pushup",
                    "rep": rep_count,
                    "quality": "BAD",
                    "is_good": False,
                    "feedback": feedback_msg
                })

            min_elbow_angle = 180

        # Draw
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f"Reps: {rep_count}", (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2)

        cv2.putText(frame, feedback_msg, (30,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0,255,0) if feedback_msg=="Good form" else (0,0,255), 3)

    cv2.imshow("Push-up AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if rep_log:
    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/pushup_{int(time.time())}.csv"
    pd.DataFrame(rep_log).to_csv(out_path, index=False)
    print(f"SESSION_LOG:{out_path}")

cap.release()
cv2.destroyAllWindows()