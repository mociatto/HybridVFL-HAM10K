from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import webbrowser
import subprocess
import threading
import time
import os
import sys

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

latest_metrics = {}
training_process = None

@app.route('/')
def index():
    return render_template('dashboard.html')

# Example: receive metrics from training process and broadcast to dashboard
@socketio.on('metrics_update')
def handle_metrics_update(data):
    global latest_metrics
    latest_metrics = data
    emit('metrics_update', data, broadcast=True)

@socketio.on('connect')
def handle_connect():
    if latest_metrics:
        emit('metrics_update', latest_metrics)

def launch_training():
    """Launch main.py training process after dashboard is ready"""
    global training_process
    
    # Wait a bit for dashboard to be fully ready
    print("Waiting for dashboard to initialize...")
    time.sleep(3)
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("Error: main.py not found in current directory")
        return
    
    try:
        print("Launching training process (main.py)...")
        print("Training metrics will appear in the dashboard shortly")
        print("-" * 60)
        
        # Launch main.py with the same Python interpreter
        training_process = subprocess.Popen(
            [sys.executable, 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print training output in real-time
        for line in iter(training_process.stdout.readline, ''):
            if line:
                print(f"[TRAINING] {line.rstrip()}")
        
        training_process.wait()
        
        if training_process.returncode == 0:
            print("Training completed successfully!")
        else:
            print(f"Training ended with error code: {training_process.returncode}")
            
    except Exception as e:
        print(f"Error launching training: {e}")

def cleanup_training():
    """Clean up training process on dashboard shutdown"""
    global training_process
    if training_process and training_process.poll() is None:
        print("Stopping training process...")
        training_process.terminate()

if __name__ == '__main__':
    port = 5050
    url = f'http://localhost:{port}'
    
    print("HYBRIDVFL Dashboard & Training Launcher")
    print("=" * 50)
    print(f"Dashboard starting at: {url}")
    print("Training (main.py) will launch automatically")
    print("=" * 50)
    
    # Launch training in a separate thread after dashboard starts
    training_thread = threading.Thread(target=launch_training, daemon=True)
    training_thread.start()
    
    try:
        # Open browser
        try:
            webbrowser.open(url)
        except Exception:
            pass
        
        # Start dashboard server
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
        
    except KeyboardInterrupt:
        print("\nShutting down dashboard and training...")
        cleanup_training()
    except Exception as e:
        print(f"Dashboard error: {e}")
        cleanup_training()