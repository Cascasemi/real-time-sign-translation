try:
    import eventlet
    eventlet.monkey_patch()  # Must be first import!
except ImportError:
    print("⚠️ eventlet not found, falling back to threading")

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
import os
import time
import eventlet
from concurrent.futures import ThreadPoolExecutor

# Required for Render.com WebSocket support
eventlet.monkey_patch()

# Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Socket.IO with Render optimization
socketio = SocketIO(app,
                    cors_allowed_origins="*",
                    async_mode='eventlet',
                    max_http_buffer_size=10 * 1024 * 1024,
                    ping_timeout=60,
                    ping_interval=25)

# Thread pool for parallel processing (matches Render's worker count)
executor = ThreadPoolExecutor(max_workers=4)

# Load ML model with error handling
model = None
try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
        print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")

# Sign Language Labels
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 27: 'Done',
    28: 'Thank You', 29: 'I Love You', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome'
}

# MediaPipe with Render-optimized settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,  # Better for video streams
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,  # Lower for better continuity
    model_complexity=1  # Balanced performance
)

# Performance tracking (for debugging)
processing_times = []


@app.route('/health')
def health():
    """Render.com health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_ready": bool(model),
        "avg_processing_ms": np.mean(processing_times[-10:] or 0)
    })


@app.route('/')
def index():
    return jsonify({
        "message": "Sign Language API",
        "websocket_endpoint": "wss://" + request.host + "/socket.io"
    })


@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connections"""
    print(f'🚀 Client connected: {request.sid}')
    emit('connection_status', {
        'status': 'connected',
        'model_ready': bool(model),
        'max_fps': 30  # Inform client of recommended rate
    })


@socketio.on('disconnect')
def handle_disconnect():
    print(f'🔌 Client disconnected: {request.sid}')


@socketio.on('frame')
def handle_frame(data):
    """Process incoming frames from Flutter"""
    if 'image' not in data:
        emit('error', {'message': 'No image data'}, room=request.sid)
        return

    # Offload processing to thread pool
    executor.submit(process_frame_async, data['image'], request.sid)


def process_frame_async(base64_img, client_id):
    """Background frame processing"""
    try:
        start_time = time.perf_counter()

        # Decode with error handling
        try:
            img_bytes = base64.b64decode(base64_img.split(',')[-1])
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid image data")
        except Exception as e:
            emit('error', {'message': f'Image decode error: {str(e)}'}, room=client_id)
            return

        # Process and annotate frame
        annotated_frame, result = process_hand_landmarks(frame)

        # Optimized JPEG encoding for Web
        _, buffer = cv2.imencode('.jpg', annotated_frame, [
            int(cv2.IMWRITE_JPEG_QUALITY), 85  # Balanced quality/size
        ])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send response
        emit('prediction', {
            'text': result['text'],
            'confidence': result['confidence'],
            'annotated_frame': frame_base64,
            'processing_ms': int((time.perf_counter() - start_time) * 1000)
        }, room=client_id)

        # Track performance (keep last 100 samples)
        processing_times.append((time.perf_counter() - start_time) * 1000)
        if len(processing_times) > 100:
            processing_times.pop(0)

    except Exception as e:
        print(f"⚠️ Processing error: {e}")
        emit('error', {
            'message': 'Processing failed',
            'details': str(e)
        }, room=client_id)


def process_hand_landmarks(frame):
    """Detect hands and make predictions"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    annotated_frame = frame.copy()
    result = {'text': '', 'confidence': 0}

    if results.multi_hand_landmarks:
        # Draw all detected hands
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

        # Use dominant hand for prediction
        if model and results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]  # Primary hand
            data_aux = []
            for landmark in hand.landmark:
                data_aux.extend([landmark.x, landmark.y])

            try:
                prediction = model.predict([np.asarray(data_aux)])
                proba = model.predict_proba([np.asarray(data_aux)])
                result = {
                    'text': LABELS.get(int(prediction[0]), 'Unknown'),
                    'confidence': float(np.max(proba[0]))
                }
            except Exception as e:
                print(f"Prediction error: {e}")

    return annotated_frame, result


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render uses 10000
    print(f"🚀 Starting server on port {port}")

    # Production-ready server
    socketio.run(app,
                 host='0.0.0.0',
                 port=port,
                 debug=False,
                 use_reloader=False,
                 log_output=True)