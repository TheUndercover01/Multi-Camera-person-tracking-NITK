from flask import Flask, render_template, jsonify
import random
import time
from datetime import datetime
import threading

app = Flask(__name__)

# Global variable to store logs
surveillance_logs = []

def generate_dummy_log():
    entities = ["human", "vehicle", "bicycle", "animal"]
    certainty = round(random.uniform(0.1, 1.0), 2)
    is_jaywalking = random.choice([True, False])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "timestamp": timestamp,
        "entity": random.choice(entities),
        "certainty": certainty,
        "is_jaywalking": is_jaywalking
    }

def background_log_generator():
    while True:
        new_log = generate_dummy_log()
        surveillance_logs.append(new_log)
        # Keep only last 100 logs
        if len(surveillance_logs) > 100:
            surveillance_logs.pop(0)
        time.sleep(1)  # Generate a new log every second

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