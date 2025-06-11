from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import webbrowser
import subprocess
import threading
import time
import os
import sys
import signal
import re

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

latest_metrics = {}
training_process = None
training_thread = None

@app.route('/')
def index():
    return render_template('dashboard.html')

# Receive metrics from training process and broadcast to dashboard
@socketio.on('metrics_update')
def handle_metrics_update(data):
    global latest_metrics
    latest_metrics = data
    emit('metrics_update', data, broadcast=True)

@socketio.on('connect')
def handle_connect():
    # Send initial status when client connects
    initial_status = {
        'current_status': 'Choose Model Mode...',
        'batch_size': '-',
        'round': 0,
        'rounds_total': '-',
        'epochs_per_round': '-',
        'feature_class': '-',
        'running_time': '00:00'
    }
    emit('metrics_update', initial_status)

@socketio.on('switch_mode')
def handle_mode_switch(data):
    """Handle mode switching from frontend"""
    global latest_metrics
    
    mode = data.get('mode', 'VanillaFL')
    privacy_vfl = data.get('secureVFL', False)
    
    print(f"\n=== MODE SWITCH REQUEST ===")
    print(f"Requested mode: {mode}")
    print(f"SecureVFL value: {privacy_vfl}")
    print(f"===========================")
    
    try:
        # Stop current training if running
        stop_training()
        
        preserved_final_metrics = {
            'test_accuracy': latest_metrics.get('test_accuracy', 0),
            'f1_score': latest_metrics.get('f1_score', 0), 
            'leak_gender_fused': latest_metrics.get('leak_gender_fused', 0),
            'leak_age_fused': latest_metrics.get('leak_age_fused', 0),
            'leak_gender_image': latest_metrics.get('leak_gender_image', 0),
            'leak_age_image': latest_metrics.get('leak_age_image', 0),
            'leak_gender_tabular': latest_metrics.get('leak_gender_tabular', 0),
            'leak_age_tabular': latest_metrics.get('leak_age_tabular', 0),
            'final_running_time': latest_metrics.get('final_running_time', '00:00'),
            'training_completed': latest_metrics.get('training_completed', False)
        }
        
        # Store previous mode results for comparison
        previous_mode = "SecureVFL" if latest_metrics.get('current_status', '').lower().find('secure') != -1 else "VanillaFL"
        if preserved_final_metrics['training_completed'] and preserved_final_metrics['test_accuracy'] > 0:

            mode_comparison_key = f'previous_{previous_mode.lower()}_results'
            preserved_final_metrics[mode_comparison_key] = {
                'test_accuracy': preserved_final_metrics['test_accuracy'],
                'f1_score': preserved_final_metrics['f1_score'],
                'leak_gender': preserved_final_metrics['leak_gender_fused'],
                'leak_age': preserved_final_metrics['leak_age_fused'],
                'mode': previous_mode
            }
        
        latest_metrics = {}
        
        reset_metrics = {
            'current_status': f'Switching to {mode} mode...',
            'batch_size': '-',
            'round': 0,
            'rounds_total': '-',
            'epochs_per_round': '-',
            'feature_class': '-',
            'running_time': '00:00',
            'train_acc_history': [],
            'val_acc_history': [],
            'train_loss_history': [],
            'gender_acc_history': [],
            'age_acc_history': [],
            'adv_loss_history': [],
            'test_accuracy': preserved_final_metrics['test_accuracy'],
            'f1_score': preserved_final_metrics['f1_score'],
            'leak_gender_fused': preserved_final_metrics['leak_gender_fused'],
            'leak_age_fused': preserved_final_metrics['leak_age_fused'],
            'leak_gender_image': preserved_final_metrics['leak_gender_image'],
            'leak_age_image': preserved_final_metrics['leak_age_image'],
            'leak_gender_tabular': preserved_final_metrics['leak_gender_tabular'],
            'leak_age_tabular': preserved_final_metrics['leak_age_tabular'],
            'final_running_time': preserved_final_metrics['final_running_time'],
            'mode_switching': True,
            'previous_mode_results': preserved_final_metrics.get(f'previous_{previous_mode.lower()}_results', {}),
            'switching_to_mode': mode
        }
        emit('metrics_update', reset_metrics, broadcast=True)
        
        if update_securevfl_setting(privacy_vfl):
            # Start new training process
            start_training()
            
            emit('mode_switched', {
                'mode': mode,
                'secureVFL': privacy_vfl,
                'message': f'Successfully switched to {mode} mode',
                'previous_results': preserved_final_metrics
            })
            
            print(f"Successfully switched to {mode} mode")
        else:
            emit('mode_switch_error', {
                'message': 'Failed to update main.py configuration'
            })
            print("Failed to update main.py configuration")
            
    except Exception as e:
        print(f"Error during mode switch: {e}")
        emit('mode_switch_error', {
            'message': f'Error switching mode: {str(e)}'
        })

def update_securevfl_setting(secure_vfl_value):
    """Update the SecureVFL setting in main.py"""
    try:
        
        with open('main.py', 'r') as file:
            content = file.read()
        

        pattern = r'SecureVFL\s*=\s*(True|False)'
        replacement = f'SecureVFL = {secure_vfl_value}'
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            
            with open('main.py', 'w') as file:
                file.write(new_content)
            
            print(f"Updated main.py: SecureVFL = {secure_vfl_value}")
            return True
        else:
            print("Could not find SecureVFL setting in main.py")
            return False
            
    except Exception as e:
        print(f"Error updating main.py: {e}")
        return False

def stop_training():
    """Stop the current training process"""
    global training_process, training_thread
    
    if training_process and training_process.poll() is None:
        print("Stopping current training process...")
        try:

            training_process.terminate()
            
            try:
                training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated
                training_process.kill()
                training_process.wait()
            
            print("Training process stopped")
        except Exception as e:
            print(f"Error stopping training: {e}")
    
    training_process = None
    
    socketio.emit('training_stopped', {'message': 'Training stopped'})

def start_training():
    """Start the training process"""
    global training_process, training_thread
    
    # Stop any existing training first
    stop_training()
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=launch_training, daemon=True)
    training_thread.start()

def launch_training():
    """Launch main.py training process"""
    global training_process
    
    # Wait for file system
    time.sleep(1)
    
    if not os.path.exists('main.py'):
        print("Error: main.py not found in current directory")
        socketio.emit('mode_switch_error', {'message': 'main.py not found'})
        return
    
    try:
        print("Launching training process (main.py)...")
        
        # Launch main.py with the same Python interpreter
        training_process = subprocess.Popen(
            [sys.executable, 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid if os.name != 'nt' else None  # For proper process group handling
        )
        
        # Print training output
        for line in iter(training_process.stdout.readline, ''):
            if line and training_process.poll() is None:
                print(f"[TRAINING] {line.rstrip()}")
        
        training_process.wait()
        
        if training_process.returncode == 0:
            print("Training completed successfully!")
            socketio.emit('training_completed', {'message': 'Training completed successfully'})
        else:
            print(f"Training ended with error code: {training_process.returncode}")
            socketio.emit('training_stopped', {'message': f'Training ended with error code: {training_process.returncode}'})
            
    except Exception as e:
        print(f"Error launching training: {e}")
        socketio.emit('mode_switch_error', {'message': f'Error launching training: {str(e)}'})

def cleanup_training():
    """Clean up training process on dashboard shutdown"""
    global training_process
    if training_process and training_process.poll() is None:
        print("Stopping training process...")
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(training_process.pid), signal.SIGTERM)
            else:  # Windows
                training_process.terminate()
            training_process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            if os.name != 'nt':
                os.killpg(os.getpgid(training_process.pid), signal.SIGKILL)
            else:
                training_process.kill()

if __name__ == '__main__':
    port = 5050
    url = f'http://localhost:{port}'
    
    print("HYBRIDVFL Dashboard with Mode Selection")
    print("=" * 50)
    print(f"Dashboard starting at: {url}")
    print("Select training mode in the browser to begin")
    print("=" * 50)
    
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
