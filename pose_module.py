import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # Stats
        self.count = 0
        self.direction = 0
        self.max_low_angle = 180
        self.feedback = "Ready"
        self.form_feedback = "Good Form"
        
        # Buffers
        self.arm_buffer = deque(maxlen=5)
        self.body_buffer = deque(maxlen=5)
        self.flare_buffer = deque(maxlen=5)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def process_frame(self, frame):
        # 1. Clear text from previous frame (We don't draw text on the image anymore!)
        # We only draw the skeleton.
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            def get_xy(idx):
                return [landmarks[idx].x * w, landmarks[idx].y * h]

            # Landmarks
            l_shldr, r_shldr = get_xy(11), get_xy(12)
            l_elbow = get_xy(13)
            l_wrist = get_xy(15)
            l_hip = get_xy(23)
            l_knee = get_xy(25)
            nose = get_xy(0)

            # Angles
            arm_angle = self.calculate_angle(l_shldr, l_elbow, l_wrist)
            body_angle = self.calculate_angle(l_shldr, l_hip, l_knee)
            flare_angle = self.calculate_angle(l_hip, l_shldr, l_elbow)
            neck_angle = self.calculate_angle(nose, l_shldr, l_hip)

            # Smoothing
            self.arm_buffer.append(arm_angle)
            self.body_buffer.append(body_angle)
            self.flare_buffer.append(flare_angle)
            
            # Use smoothed values
            s_arm = np.mean(self.arm_buffer)
            s_body = np.mean(self.body_buffer)
            s_flare = np.mean(self.flare_buffer)

            # --- LOGIC ---
            current_errors = []
            
            if s_body < 160: current_errors.append("Lower Hips")
            elif s_body > 200: current_errors.append("Hips Sagging")
            
            if s_flare > 75: current_errors.append("Tuck Elbows")
            
            if abs(l_wrist[0] - l_shldr[0]) > 70: current_errors.append("Hands Under Shoulders")
            
            if neck_angle < 150: current_errors.append("Head Too Low")

            # Set Feedback
            if not current_errors:
                self.form_feedback = "Good Form"
            else:
                self.form_feedback = current_errors[0] # Show the most important error first

            # Counting
            if s_arm > 160:
                self.feedback = "Up"
                if self.direction == 1:
                    if self.max_low_angle <= 100:
                        if self.form_feedback == "Good Form":
                            self.count += 1
                    self.direction = 0
                    self.max_low_angle = 180
            
            if s_arm < 100:
                self.feedback = "Down"
                self.direction = 1
                if s_arm < self.max_low_angle:
                    self.max_low_angle = s_arm

            # Draw ONLY the Skeleton (No text)
            self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame