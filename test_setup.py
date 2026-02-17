import mediapipe as mp
import cv2
import flask

print(f"✅ Flask Version: {flask.__version__}")
print(f"✅ OpenCV Version: {cv2.__version__}")

try:
    mp_pose = mp.solutions.pose
    print("✅ SUCCESS: MediaPipe Solutions loaded successfully!")
except AttributeError:
    print("❌ ERROR: MediaPipe is still broken.")