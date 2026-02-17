import cv2
import yt_dlp
from flask import Flask, render_template, Response, jsonify
from pose_module import PoseDetector
from yolo_pose_module import YoloPoseDetector


app = Flask(__name__)

# --- 1. CONFIGURATION ---
YOUTUBE_URL = "https://youtu.be/IODxDxX7oi4?si=U_ffgfHIppsMj2BA" # Perfect Pushup
START_TIME = 100   # Start at 45 seconds
END_TIME = 300     # End at 55 seconds (Loop back to 45)

# --- 2. YOUTUBE EXTRACTOR ---
def get_youtube_stream_url(url):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f"❌ Error getting YouTube URL: {e}")
        return None

# Get video stream
print(f"🚀 Extracting YouTube Stream...")
video_path = get_youtube_stream_url(YOUTUBE_URL)

if video_path is None:
    print("⚠️ YouTube failed. Using Webcam.")
    video_path = 0 

# --- 3. GLOBAL OBJECTS ---
detector = PoseDetector()

# detector = YoloPoseDetector()

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    
    # JUMP TO START immediately
    cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

    while True:
        success, frame = cap.read()
        
        if not success:
            # If video ends naturally, loop back to start time
            cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)
            continue

        # --- TIME CHECK LOOP ---
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if current_time >= END_TIME:
            # Reached end of clip? Rewind!
            cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)
            continue
        
        # --- AI PROCESSING ---
        # No flip needed for YouTube
        frame = detector.process_frame(frame)
        
        # --- ENCODE ---
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        'reps': detector.count,
        'state': detector.feedback,
        'form': detector.form_feedback
    })

if __name__ == "__main__":
    app.run(debug=True)