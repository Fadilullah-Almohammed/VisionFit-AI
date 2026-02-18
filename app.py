import cv2
import yt_dlp
from flask import Flask, render_template, Response, jsonify, request
# Select your preferred module: 'pose_module' OR 'yolo_pose_module'
from pose_module import PoseDetector 

app = Flask(__name__)

# --- GLOBAL SETTINGS ---
current_config = {
    "mode": "webcam",  # 'webcam' or 'youtube'
    "url": None,
    "start": 0,
    "end": 0
}

detector = PoseDetector()

def get_youtube_stream(url):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except:
        return None

def generate_frames():
    # 1. SETUP SOURCE
    if current_config["mode"] == "youtube" and current_config["url"]:
        src = get_youtube_stream(current_config["url"])
        cap = cv2.VideoCapture(src if src else 0)
        # Jump to start
        cap.set(cv2.CAP_PROP_POS_MSEC, current_config["start"] * 1000)
    else:
        cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            if current_config["mode"] == "youtube":
                # Loop video
                cap.set(cv2.CAP_PROP_POS_MSEC, current_config["start"] * 1000)
                continue
            else:
                break
        
        # Check End Time (for YouTube)
        if current_config["mode"] == "youtube" and current_config["end"] > 0:
            current_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if current_pos >= current_config["end"]:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_config["start"] * 1000)

        # 2. AI PROCESSING
        # Flip only if webcam
        if current_config["mode"] == "webcam":
            frame = cv2.flip(frame, 1)
            
        frame = detector.process_frame(frame)

        # 3. STREAM
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    current_config["mode"] = data.get("mode")
    current_config["url"] = data.get("url")
    current_config["start"] = int(data.get("start", 0))
    current_config["end"] = int(data.get("end", 0))
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)