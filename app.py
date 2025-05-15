try:
    eventlet.monkey_patch()
    print("✅ Eventlet monkey patching successful")
except Exception as e:
    print(f"⚠️ Eventlet patching failed: {e}")
    import threading

    print("Falling back to threading")

import os
import time
import base64
import eventlet
import numpy as np
import cv2
import pickle
from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp

# ==================== INITIALIZATION ====================
# Critical for WebSocket support


# Initialize Flask with static files support
app = Flask(__name__, static_folder='static')
app.config.update({
    'SECRET_KEY': os.urandom(24),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'TEMPLATES_AUTO_RELOAD': True
})

# Socket.IO Configuration
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=300,
    ping_interval=60,
    engineio_logger=True,
    logger=True
)

# ==================== ML MODEL SETUP ====================
MODEL = None
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 27: 'Done',
    28: 'Thank You', 29: 'I Love You', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome'
}

try:
    with open('model.p', 'rb') as f:
        MODEL = pickle.load(f).get('model')
        print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")

# ==================== MEDIAPIPE CONFIG ====================
mp_hands = mp.solutions.hands
HANDS = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

# ==================== CORE FUNCTIONALITY ====================
executor = ThreadPoolExecutor(max_workers=4)


def process_frame(image_data):
    """Process a single frame through MediaPipe and ML model"""
    try:
        # Decode image
        img_bytes = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image data")

        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = HANDS.process(frame_rgb)
        annotated_frame = frame.copy()
        result = {'text': '', 'confidence': 0}

        if results.multi_hand_landmarks:
            # Draw landmarks
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

            # Make prediction
            if MODEL:
                hand = results.multi_hand_landmarks[0]
                data_aux = [coord for landmark in hand.landmark
                            for coord in [landmark.x, landmark.y]]

                prediction = MODEL.predict([np.asarray(data_aux)])
                proba = MODEL.predict_proba([np.asarray(data_aux)])
                result = {
                    'text': LABELS.get(int(prediction[0]), 'Unknown'),
                    'confidence': float(np.max(proba[0]))
                }

        # Encode result
        _, buffer = cv2.imencode('.jpg', annotated_frame, [
            int(cv2.IMWRITE_JPEG_QUALITY), 85
        ])
        return {
            **result,
            'annotated_frame': base64.b64encode(buffer).decode('utf-8')
        }

    except Exception as e:
        print(f"⚠️ Frame processing error: {e}")
        raise


# ==================== ROUTES ====================
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "endpoints": {
            "websocket": "/socket.io",
            "test_page": "/test",
            "health_check": "/health"
        }
    })


@app.route('/health')
def health():
    """Render.com health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_ready": bool(MODEL),
        "timestamp": time.time()
    })


@app.route('/test')
def test_interface():
    """Serve HTML test page"""
    return send_from_directory('static', 'test.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files (for Socket.IO client)"""
    return send_from_directory('static', filename)


# ==================== SOCKET.IO HANDLERS ====================
@socketio.on('connect')
def handle_connect():
    """New client connection"""
    emit('connection_status', {
        'status': 'connected',
        'model_ready': bool(MODEL),
        'server_time': time.time()
    })
    print(f"🚀 New connection: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f"🔌 Disconnected: {request.sid}")


@socketio.on('frame')
def handle_frame(data):
    """Process incoming video frames"""
    if 'image' not in data:
        emit('error', {'message': 'No image data'}, room=request.sid)
        return

    def callback(future):
        try:
            result = future.result()
            emit('prediction', {
                **result,
                'timestamp': time.time()
            }, room=request.sid)
        except Exception as e:
            emit('error', {
                'message': 'Processing failed',
                'details': str(e)
            }, room=request.sid)

    executor.submit(process_frame, data['image']).add_done_callback(callback)


# ==================== START SERVER ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 Starting server on port {port}")

    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )