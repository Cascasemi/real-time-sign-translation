import os
import time
import base64
import eventlet
import numpy as np
import cv2
import pickle
from flask import Flask, jsonify, send_from_directory, request  # Added request import
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
import logging

# Initialize eventlet for WebSocket support
eventlet.monkey_patch()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask Application Setup
app = Flask(__name__, static_folder='static', static_url_path='')
app.config.update({
    'SECRET_KEY': os.urandom(24),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB
})

# Socket.IO Configuration
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=300,
    ping_interval=60,
    logger=True,
    engineio_logger=True
)

# Load ML Model
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
        logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Model loading error: {e}")

# MediaPipe Configuration
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Thread Pool for Parallel Processing
executor = ThreadPoolExecutor(max_workers=4)

# Routes
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "endpoints": {
            "test_interface": "/test",
            "websocket": "/socket.io",
            "health_check": "/health"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_ready": bool(MODEL),
        "timestamp": time.time()
    })

@app.route('/test')
def test_interface():
    """Serve the test interface HTML page"""
    return send_from_directory('static', 'test_local.html')

# Socket.IO Handlers
@socketio.on('connect')
def handle_connect():
    try:
        logger.info(f"🚀 Client connected: {request.sid}")
        emit('connection_status', {
            'status': 'connected',
            'model_ready': bool(MODEL)
        })
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return False

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"🔌 Client disconnected: {request.sid}")

@socketio.on('frame')
def handle_frame(data):
    try:
        if 'image' not in data:
            raise ValueError("No image data provided")

        def callback(future):
            try:
                result = future.result()
                emit('prediction', {
                    'text': result.get('text', ''),
                    'confidence': result.get('confidence', 0),
                    'annotated_frame': result.get('annotated_frame', ''),
                    'processing_ms': result.get('processing_ms', 0)
                }, room=request.sid)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                emit('error', {
                    'message': 'Processing failed',
                    'details': str(e)
                }, room=request.sid)

        executor.submit(process_frame, data['image'], request.sid).add_done_callback(callback)
    except Exception as e:
        logger.error(f"Frame handling error: {e}")
        emit('error', {
            'message': 'Frame processing error',
            'details': str(e)
        }, room=request.sid)

def process_frame(image_data, client_id):
    start_time = time.perf_counter()
    try:
        # Decode image
        img_bytes = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image data")

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
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
                data_aux = []
                for landmark in hand.landmark:
                    data_aux.extend([landmark.x, landmark.y])

                prediction = MODEL.predict([np.asarray(data_aux)])
                proba = MODEL.predict_proba([np.asarray(data_aux)])
                result = {
                    'text': LABELS.get(int(prediction[0]), 'Unknown'),
                    'confidence': float(np.max(proba[0]))
                }

        # Encode result
        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        processing_ms = (time.perf_counter() - start_time) * 1000

        return {
            **result,
            'annotated_frame': base64.b64encode(buffer).decode('utf-8'),
            'processing_ms': processing_ms
        }

    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        raise

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Starting server on port {port}")
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )