import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pandas as pd
import os
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Thresholds
DOWN_THRESHOLD = 100
UP_THRESHOLD = 160
DEPTH_THRESHOLD = 110

SMOOTHING = 5

# State
state = "down"
rep_count = 0
angle_buffer = deque(maxlen=SMOOTHING)
min_angle = 180
max_angle = 0
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

        # Right arm
        shoulder_r = (lm[12].x, lm[12].y)
        elbow_r = (lm[14].x, lm[14].y)
        wrist_r = (lm[16].x, lm[16].y)

        # Left arm
        shoulder_l = (lm[11].x, lm[11].y)
        elbow_l = (lm[13].x, lm[13].y)
        wrist_l = (lm[15].x, lm[15].y)

        # Angles
        angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
        angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

        angle = (angle_r + angle_l) / 2

        # Smoothing
        angle_buffer.append(angle)
        smooth_angle = np.mean(angle_buffer)

        min_angle = min(min_angle, smooth_angle)
        max_angle = max(max_angle, smooth_angle)

        # Rep logic
        if state == "down" and smooth_angle > UP_THRESHOLD:
            state = "up"

        elif state == "up" and smooth_angle < DOWN_THRESHOLD:
            state = "down"

            # Strict validation
            full_extension = max_angle > UP_THRESHOLD
            good_depth = min_angle < DEPTH_THRESHOLD
            good_balance = abs(angle_r - angle_l) < 20

            if full_extension and good_depth and good_balance:
                rep_count += 1
                feedback_msg = "Good form"

                rep_log.append({
                    "time": time.time(),
                    "exercise": "shoulder",
                    "rep": rep_count,
                    "quality": "GOOD",
                    "is_good": True,
                    "feedback": feedback_msg
                })
            else:
                if not full_extension:
                    feedback_msg = "Extend arms fully"
                elif not good_depth:
                    feedback_msg = "Lower elbows more"
                elif not good_balance:
                    feedback_msg = "Balance both arms"
                else:
                    feedback_msg = "Control movement"

                rep_log.append({
                    "time": time.time(),
                    "exercise": "shoulder",
                    "rep": rep_count,
                    "quality": "BAD",
                    "is_good": False,
                    "feedback": feedback_msg
                })

            # Reset
            min_angle = 180
            max_angle = 0

        # Draw
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f"REPS: {rep_count}", (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,0), 3)

        cv2.putText(frame, feedback_msg, (30,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0,255,0) if feedback_msg=="Good form" else (0,0,255), 3)

    cv2.imshow("Shoulder Press AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if rep_log:
    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/shoulder_{int(time.time())}.csv"
    pd.DataFrame(rep_log).to_csv(out_path, index=False)
    print(f"SESSION_LOG:{out_path}")

cap.release()
cv2.destroyAllWindows()