from flask import Flask, render_template, jsonify
import threading
from face_recog import FaceRecognitionSystem, surveillance_logs
import time 
app = Flask(__name__)

# Initialize the face recognition system
face_recognition_system = FaceRecognitionSystem()

def background_log_generator():
    log_path = './logs/jaywalking_log.txt'
    while True:
        face_recognition_system.recognize_images_from_log(log_path)
        time.sleep(1)  # Process logs every second

# Start background thread for log generation
log_thread = threading.Thread(target=background_log_generator, daemon=True)
log_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_logs')
def get_logs():
    return jsonify(surveillance_logs[-20:])  # Return last 20 logs

if __name__ == '__main__':
    app.run(debug=True)