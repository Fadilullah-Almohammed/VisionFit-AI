# 🏋️‍♂️ VisionFit Professional — AI Fitness Trainer

**VisionFit** is a professional AI-powered fitness dashboard that analyzes exercise form in real time using computer vision.

It tracks body movements, counts repetitions accurately, and provides instant corrective feedback such as:
- “Lower Hips”
- “Tuck Elbows”
- “Align Head”


---

## 🛠️ Prerequisites

Make sure you have the following installed:
- **Python 3.8** or higher
- **Git**

---

## 🚀 Installation Guide

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/YOUR_USERNAME/VisionFit.git](https://github.com/YOUR_USERNAME/VisionFit.git)
cd VisionFit
2. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies isolated.

For Windows:

Bash
# Create the environment
python -m venv .venv

# Activate the environment
.venv\Scripts\activate
For macOS / Linux:

Bash
# Create the environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate
You will know it worked if you see (.venv) appear at the start of your terminal line.

3. Install Dependencies
Install all required libraries (Flask, OpenCV, MediaPipe, etc.) using pip:

Bash
pip install -r requirements.txt
🎮 How to Run
Start the Application:
Make sure your virtual environment is activated, then run:

Bash
python app.py
Open the Dashboard:
You will see a message saying Running on http://127.0.0.1:5000.
Open your web browser and navigate to:
http://127.0.0.1:5000