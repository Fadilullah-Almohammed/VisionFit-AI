import cv2
import os

video_path = "pushup.mp4"

print(f"📂 Current Working Directory: {os.getcwd()}")
print(f"🔍 Looking for: {video_path}")

# Check 1: Does the file exist?
if os.path.exists(video_path):
    print("✅ File exists!")
else:
    print("❌ ERROR: File NOT found.")
    print("   👉 Check if the file is named 'pushup.mp4.mp4' by mistake.")
    print("   👉 Check if it is inside the 'VisionFit' folder.")
    exit()

# Check 2: Can OpenCV open it?
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ERROR: File exists, but OpenCV cannot open it.")
    print("   👉 The video codec might be unsupported. Try converting it.")
else:
    ret, frame = cap.read()
    if ret:
        print("✅ SUCCESS: OpenCV can read the video!")
        print(f"   📏 Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("❌ ERROR: Video opened, but the first frame is empty.")