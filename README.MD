# 🏋️ AI Workout Detection & Form Correction System

Final Year Engineering Project

---

# 📌 Project Overview

AI Workout Detection & Form Correction System is a real-time fitness coaching application that uses Computer Vision, Pose Estimation, and Machine Learning to monitor workout exercises through a webcam.

The system detects human body posture using MediaPipe Pose Estimation, calculates joint angles, validates exercise form, counts repetitions, detects incorrect posture, and generates workout session summaries.

The project is designed as a lightweight real-time AI fitness assistant.

---

# 🎯 Main Objectives

- Detect workout exercises in real time
- Count exercise repetitions accurately
- Validate exercise posture and form
- Detect incorrect movements
- Provide visual feedback to the user
- Generate workout session summaries

---

# 🧠 Technologies Used

## Programming Language
- Python

## Computer Vision & AI
- OpenCV
- MediaPipe Pose
- NumPy
- Scikit-learn

## Machine Learning
- Random Forest Classifier

## Data Processing
- Pandas

## Version Control
- Git
- GitHub

---

# 🏗️ Project Architecture

text Camera Input       ↓ MediaPipe Pose Estimation       ↓ Keypoint Extraction       ↓ Joint Angle Calculation       ↓ Exercise Validation Logic       ↓ ML-Based Quality Prediction       ↓ Rep Counting       ↓ Session Logging       ↓ Workout Summary Generation 

---

# 📂 Project Structure

text ai-workout-detection/ │ ├── app.py │ ├── Exercises/ │   ├── squat.py │   ├── pushups.py │   ├── bicep.py │   └── shoulder.py │ ├── notebooks/ │   └── AI_Workout_train.ipynb │ ├── models/ │   └── squat_model.pkl │ ├── logs/ │ ├── data/ │ ├── requirements.txt │ └── README.md 

---

# ✅ Features Implemented

## 1. Real-Time Pose Detection

- Detects body posture using webcam
- Uses MediaPipe Pose landmarks
- Tracks key body joints in real time

---

## 2. Exercise Detection

Implemented exercises:

- Squats
- Push-Ups
- Bicep Curls
- Shoulder Press

---

## 3. Joint Angle Calculation

The system calculates:

- Knee angles
- Elbow angles
- Hip angles
- Shoulder angles

These angles are used for:

- Rep detection
- Form correction
- Posture validation

---

## 4. AI-Based Form Classification

A Random Forest ML model was trained using:

- Good exercise samples
- Bad exercise samples

Dataset includes:

- Extracted joint angles
- Labels:
  - GOOD
  - BAD

The model predicts:

- GOOD form
- BAD form

---

## 5. Real-Time Rep Counting

The system:

- Detects movement states
- Tracks motion transitions
- Counts valid repetitions only
- Prevents false multiple counts

---

## 6. Visual Feedback System

The application displays:

- Total reps
- Exercise state
- Form quality
- Mistake indicators

Example feedback:

- Go Lower
- Extend Arms Fully
- Balance Both Arms
- Keep Body Straight

---

## 7. Session Logging

Each workout session is saved into CSV logs.

Logged data includes:

- Exercise name
- Rep count
- Good/Bad quality
- Feedback message
- Joint angle data

---

## 8. Workout Summary System

After workout completion:

- Session logs are analyzed automatically
- Summary statistics are generated

Summary includes:

- Total reps
- Good reps
- Bad reps
- Most common mistake

---

# 🧪 Machine Learning Pipeline

## Dataset Generation

Videos used:

- squat_good.mp4
- squat_bad.mp4

Pipeline:

1. Extract pose landmarks
2. Calculate joint angles
3. Generate CSV datasets
4. Train ML model

---

## Features Used

text knee_angle_r knee_angle_l elbow_angle_r elbow_angle_l hip_angle_r hip_angle_l shoulder_angle_r shoulder_angle_l 

---

## Model Used

text RandomForestClassifier 

---

# 🔥 Engineering Improvements

During development, several real-world engineering problems were solved.

## Fixed Issues

- False rep counting
- Double counting
- Incorrect state transitions
- Missing model paths
- Directory handling issues
- Low visibility detection
- Visual UI scaling
- Logging consistency
- MediaPipe compatibility issues

---

# 🧠 Why This is an AI Project

The project combines:

## Computer Vision
Pose estimation using MediaPipe

## Machine Learning
Random Forest model for posture quality prediction

## Real-Time Analytics
Workout tracking and feedback generation

This makes the system:

- Intelligent
- Interactive
- Real-time
- AI-assisted

---

# 🚀 Future Scope

- Additional exercise support
- Workout performance analytics
- Personalized workout recommendations
- Mobile application integration

---

# ⚙️ Installation & Setup

## 1. Clone Repository

bash git clone https://github.com/your-username/ai-workout-detection.git 

---

## 2. Create Virtual Environment

bash python3.11 -m venv env 

---

## 3. Activate Environment

### macOS/Linux

bash source env/bin/activate 

### Windows

bash env\Scripts\activate 

---

## 4. Install Dependencies

bash pip install -r requirements.txt 

---

# ▶️ Run Application

bash python app.py 

---

# 🖥️ Application Flow

1. Launch application
2. Select exercise
3. Perform exercise in front of webcam
4. System tracks posture
5. Reps are counted
6. Feedback is shown
7. Session summary is generated

---

# 📊 Current Project Status

## Completed

- Real-time pose detection
- Multi-exercise support
- ML-based form validation
- Rep counting system
- Logging system
- Workout summary analytics
- Git-based project management

---

# 👨‍💻 Developed By

Final Year Engineering Project

AI Workout Detection & Form Correction