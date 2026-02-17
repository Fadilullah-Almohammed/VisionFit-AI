import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class YoloPoseDetector:
    def __init__(self):
        # Load the "Nano" Pose model (smallest & fastest)
        self.model = YOLO("yolo11n-pose.pt") 
        
        # Stats
        self.count = 0
        self.direction = 0
        self.max_low_angle = 180
        self.feedback = "Ready"
        self.form_feedback = "Good Form"
        
        # Buffers
        self.arm_buffer = deque(maxlen=5)
        self.body_buffer = deque(maxlen=5)

    def calculate_angle(self, a, b, c):
        # Same math as before
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def process_frame(self, frame):
        # YOLO works best on original frames, no need to flip or convert color manually
        # verbose=False keeps your terminal clean
        results = self.model(frame, verbose=False)
        
        # Did we find a person?
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # Get the first person detected (Index 0)
            # data[0] contains 17 points: [x, y, confidence]
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            h, w, _ = frame.shape
            
            # --- YOLO COCO KEYPOINT MAPPING ---
            # 5: L-Shoulder, 7: L-Elbow, 9: L-Wrist
            # 11: L-Hip, 13: L-Knee
            
            # Helper to extract (x, y) checking confidence
            def get_xy(idx):
                # keypoints[idx] is [x, y, conf]
                x, y, conf = keypoints[idx]
                return [int(x), int(y)]

            # Get Coordinates using YOLO indices
            l_shldr = get_xy(5)
            l_elbow = get_xy(7)
            l_wrist = get_xy(9)
            l_hip   = get_xy(11)
            l_knee  = get_xy(13)
            
            # (Optional) Right side for symmetry if needed
            r_shldr = get_xy(6)

            # Calculate Angles
            arm_angle = self.calculate_angle(l_shldr, l_elbow, l_wrist)
            body_angle = self.calculate_angle(l_shldr, l_hip, l_knee)

            # Smoothing
            self.arm_buffer.append(arm_angle)
            self.body_buffer.append(body_angle)
            s_arm = np.mean(self.arm_buffer)
            s_body = np.mean(self.body_buffer)

            # --- LOGIC (Same as your logic, just using YOLO angles) ---
            self.form_feedback = "Good Form"
            
            if s_body < 160: self.form_feedback = "Lower Hips"
            elif s_body > 200: self.form_feedback = "Hips Sagging"
            
            if abs(l_wrist[0] - l_shldr[0]) > 70: self.form_feedback = "Hands Under Shoulders"
            if abs(l_shldr[1] - r_shldr[1]) > 40: self.form_feedback = "Keep Shoulders Level"

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

            # --- DRAWING ---
            # YOLO has a built-in "plot()" function, but it draws bounding boxes too.
            # We will manually draw the skeleton for a cleaner look.
            
            # Draw Arm (5->7->9)
            cv2.line(frame, tuple(l_shldr), tuple(l_elbow), (255, 255, 255), 3)
            cv2.line(frame, tuple(l_elbow), tuple(l_wrist), (255, 255, 255), 3)
            
            # Draw Body (5->11->13)
            cv2.line(frame, tuple(l_shldr), tuple(l_hip), (255, 255, 255), 3)
            cv2.line(frame, tuple(l_hip), tuple(l_knee), (255, 255, 255), 3)

            # Draw Points
            for point in [l_shldr, l_elbow, l_wrist, l_hip, l_knee]:
                cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)

        return frame