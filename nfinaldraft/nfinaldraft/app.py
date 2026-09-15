from flask import Flask, jsonify, render_template, Response, request
import serial
import time
import cv2
import mediapipe as mp
from improved_ml_model import ImprovedNPKAnalyzer
from database import NPKDatabase

# --- Global State Variable ---
sensor_active = True

# --- Serial Connection ---
try:
    ser = serial.Serial('COM7', 4800, timeout=3)
    time.sleep(2)
    print("✅ Connected to Arduino on COM7")
except Exception as e:
    print(f"⚠ Could not open serial port: {e}")
    ser = None

# --- NPK Reading Function ---
def get_npk_values():
    """Requests and reads NPK data from the Arduino."""
    if ser and ser.is_open:
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Clear buffers completely
                ser.flushInput()
                ser.flushOutput()
                time.sleep(0.1)
                
                # Send request
                ser.write(b'R')
                
                # Wait for response with timeout
                timeout = time.time() + 2  # 2 second timeout
                while time.time() < timeout:
                    if ser.in_waiting > 0:
                        time.sleep(0.2)  # Let more data arrive
                        break
                    time.sleep(0.1)
                
                # Read all available data
                if ser.in_waiting > 0:
                    raw_data = ser.read(ser.in_waiting)
                    try:
                        decoded_data = raw_data.decode('utf-8', errors='ignore')
                        lines = [line.strip() for line in decoded_data.split('\n') if line.strip()]
                        
                        # Find START marker
                        start_index = -1
                        for i, line in enumerate(lines):
                            if line == "START":
                                start_index = i
                                break
                        
                        if start_index >= 0 and start_index + 3 < len(lines):
                            # Extract NPK values
                            n_line = lines[start_index + 1]
                            p_line = lines[start_index + 2]
                            k_line = lines[start_index + 3]
                            
                            # Parse values (format: "Nitrogen: 207 mg/kg")
                            if ':' in n_line and ':' in p_line and ':' in k_line:
                                n = int(n_line.split(':')[1].strip().split()[0])
                                p = int(p_line.split(':')[1].strip().split()[0])
                                k = int(k_line.split(':')[1].strip().split()[0])
                                
                                # Only return valid readings (not -1 error values)
                                if n >= 0 and p >= 0 and k >= 0:
                                    print(f"📥 NPK Values: N={n}, P={p}, K={k}")
                                    return {"nitrogen": n, "phosphorus": p, "potassium": k}
                                else:
                                    print("⚠ Arduino sensor error (values = -1)")
                                    
                    except UnicodeDecodeError:
                        print("⚠ Serial data decode error")
                        
            except Exception as e:
                print(f"⚠ Serial communication error (attempt {retry + 1}): {e}")
                # Reset serial connection on error
                try:
                    ser.flushInput()
                    ser.flushOutput()
                except:
                    pass
                
                if retry < max_retries - 1:
                    time.sleep(0.5)  # Wait before retry

    print("❌ No valid NPK data received after retries")
    return None

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Initialize ML Model and Database ---
npk_analyzer = ImprovedNPKAnalyzer()
npk_db = NPKDatabase()

# --- Camera Logic ---
try:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise IOError("Cannot open webcam")
    print("📷 Camera initialized successfully.")
except Exception as e:
    print(f"⚠ Camera Error: {e}")
    camera = None

# --- MediaPipe Hand Tracking Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ar')
def ar_view():
    return render_template('ar_clean.html')

@app.route('/toggle_sensor', methods=['POST'])
def toggle_sensor():
    """Toggle sensor active state"""
    global sensor_active
    try:
        data = request.get_json()
        sensor_active = data.get('active', True)
        print(f"🔄 Sensor toggled: {'ACTIVE' if sensor_active else 'INACTIVE'}")
        return jsonify({"success": True, "sensor_active": sensor_active})
    except Exception as e:
        print(f"❌ Toggle sensor error: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data')
def data():
    """API endpoint that provides sensor data and ML analysis."""
    # Check if test mode is requested
    TEST_MODE = request.args.get('test', 'false').lower() == 'true'
    
    if TEST_MODE:
        # Fixed fake NPK values for AR demonstration - but still respect sensor_active state
        if sensor_active:
            npk = {"nitrogen": 134, "phosphorus": 41, "potassium": 140}
            analysis = npk_analyzer.analyze_soil(npk['nitrogen'], npk['phosphorus'], npk['potassium'])
            return jsonify({"npk": npk, "status": "on", "analysis": analysis})
        else:
            return jsonify({"npk": None, "status": "off"})
    
    if sensor_active:
        npk = get_npk_values()
        
        if npk is None:
            # Sensor offline - get recent value from MongoDB
            recent_readings = npk_db.get_recent_readings(1)
            if recent_readings:
                recent = recent_readings[0]
                fallback_npk = {
                    "nitrogen": recent['nitrogen'],
                    "phosphorus": recent['phosphorus'], 
                    "potassium": recent['potassium']
                }
                
                # Use recent analysis or generate new one
                if 'analysis' in recent and recent['analysis']:
                    analysis = recent['analysis']
                else:
                    analysis = npk_analyzer.analyze_soil(
                        fallback_npk['nitrogen'], 
                        fallback_npk['phosphorus'], 
                        fallback_npk['potassium']
                    )
                
                response_data = {
                    "npk": fallback_npk,
                    "status": "recent", 
                    "analysis": analysis,
                    "timestamp": recent['timestamp'].isoformat() if 'timestamp' in recent else None,
                    "source": "database"
                }
                return jsonify(response_data)
            else:
                return jsonify({"npk": None, "status": "no_data", "message": "No sensor data available"})
        else:
            # Live sensor data
            n, p, k = npk['nitrogen'], npk['phosphorus'], npk['potassium']
            analysis = npk_analyzer.analyze_soil(n, p, k)
            
            # Store data in MongoDB
            reading_id = npk_db.store_npk_reading(n, p, k, analysis)
            
            response_data = {
                "npk": npk, 
                "status": "live",
                "analysis": analysis,
                "reading_id": reading_id,
                "source": "sensor"
            }
            return jsonify(response_data)
    else:
        return jsonify({"npk": None, "status": "off"})

def generate_frames():
    """Generator function for streaming video frames with hand gesture controls."""
    global sensor_active
    if not camera: return
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # --- Define Gesture Control Boxes ---
        # OFF Box (left side)
        off_box_pt1 = (int(width * 0.05), int(height * 0.1))
        off_box_pt2 = (int(width * 0.25), int(height * 0.3))
        
        # ON Box (right side)
        on_box_pt1 = (int(width * 0.75), int(height * 0.1))
        on_box_pt2 = (int(width * 0.95), int(height * 0.3))
        
        finger_in_off_box = False
        finger_in_on_box = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get coordinates of the index finger tip (Landmark #8)
                tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                tip_x = int(tip.x * width)
                tip_y = int(tip.y * height)
                
                # Draw a circle on the fingertip for visual feedback
                cv2.circle(frame, (tip_x, tip_y), 10, (255, 0, 255), -1)
                
                # Check if fingertip is inside the OFF box
                if off_box_pt1[0] < tip_x < off_box_pt2[0] and off_box_pt1[1] < tip_y < off_box_pt2[1]:
                    sensor_active = False
                    finger_in_off_box = True
                
                # Check if fingertip is inside the ON box
                if on_box_pt1[0] < tip_x < on_box_pt2[0] and on_box_pt1[1] < tip_y < on_box_pt2[1]:
                    sensor_active = True
                    finger_in_on_box = True
        
        # --- Draw the Boxes and Status Text on the Frame ---
        # Draw OFF box (change color if finger is inside)
        off_color = (0, 0, 200) if not finger_in_off_box else (0, 0, 255)
        cv2.rectangle(frame, off_box_pt1, off_box_pt2, off_color, 3)
        cv2.putText(frame, "OFF", (off_box_pt1[0] + 20, off_box_pt1[1] + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, off_color, 2)
        
        # Draw ON box (change color if finger is inside)
        on_color = (0, 200, 0) if not finger_in_on_box else (0, 255, 0)
        cv2.rectangle(frame, on_box_pt1, on_box_pt2, on_color, 3)
        cv2.putText(frame, "ON", (on_box_pt1[0] + 25, on_box_pt1[1] + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, on_color, 2)
        
        # Display the overall sensor status
        status_text = f"Sensor: {'ACTIVE' if sensor_active else 'INACTIVE'}"
        status_color = (0, 255, 0) if sensor_active else (0, 0, 255)
        cv2.putText(frame, status_text, (int(width*0.4), height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def get_history():
    """Get recent NPK readings from database"""
    limit = request.args.get('limit', 10, type=int)
    readings = npk_db.get_recent_readings(limit)
    return jsonify({"readings": readings, "count": len(readings)})

@app.route('/averages')
def get_averages():
    """Get average NPK values for the last 24 hours"""
    hours = request.args.get('hours', 24, type=int)
    averages = npk_db.get_average_values(hours)
    return jsonify({"averages": averages})

if __name__ == '__main__':
    # use_reloader=False is important to prevent serial port conflicts
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
