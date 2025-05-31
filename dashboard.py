from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import webbrowser

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

latest_metrics = {}

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

if __name__ == '__main__':
    port = 5050
    url = f'http://localhost:{port}'
    print(f'\nDashboard running at: {url}\n')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    socketio.run(app, host='0.0.0.0', port=port)