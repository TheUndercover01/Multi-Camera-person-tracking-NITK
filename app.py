from flask import Flask, render_template, jsonify
import threading
from face_recog import FaceRecognitionSystem, surveillance_logs
import time 
app = Flask(__name__)

# Initialize the face recognition system
face_recognition_system = FaceRecognitionSystem()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_logs')
def get_logs():
    return jsonify(surveillance_logs[-20:])  # Return last 20 logs

if __name__ == '__main__':
    app.run(debug=True)