from flask import Flask, render_template, jsonify
import threading
from face_recog import LOG_FILE_PATH
import time 
import json
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_logs')
def get_logs():
    logs = []
    try:
        with open(LOG_FILE_PATH, 'r') as log_file:
            logs = [json.loads(line) for line in log_file.readlines()]
    except FileNotFoundError:
        pass
    return jsonify(logs[-10:])

if __name__ == '__main__':
    app.run(debug=True)