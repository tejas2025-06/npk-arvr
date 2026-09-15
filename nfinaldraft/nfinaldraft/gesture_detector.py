import cv2
import mediapipe as mp
import numpy as np
import threading
import time

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.is_running = False
        self.current_gesture = "unknown"
        self.gesture_callback = None
        self.cap = None
        
    def start_detection(self, callback=None):
        """Start hand gesture detection"""
        self.gesture_callback = callback
        self.is_running = True
        
        # Start detection in a separate thread
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        return True
        
    def stop_detection(self):
        """Stop hand gesture detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
    def _detection_loop(self):
        """Main detection loop"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = self.hands.process(rgb_frame)
                
                gesture = "no_hand"
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Detect gesture based on hand landmarks
                        gesture = self._classify_gesture(hand_landmarks)
                        break  # Only process first hand
                
                # Update current gesture
                if gesture != self.current_gesture:
                    self.current_gesture = gesture
                    if self.gesture_callback:
                        self.gesture_callback(gesture)
                
                time.sleep(0.1)  # Reduce CPU usage
                
        except Exception as e:
            print(f"❌ Gesture detection error: {e}")
        finally:
            if self.cap:
                self.cap.release()
    
    def _classify_gesture(self, landmarks):
        """Classify hand gesture and detect box interaction"""
        try:
            # Get landmark positions
            landmark_list = []
            for lm in landmarks.landmark:
                landmark_list.append([lm.x, lm.y])
            
            # Convert to numpy array
            points = np.array(landmark_list)
            
            # Get key points - use index finger tip for box detection
            index_tip = points[8]
            index_pip = points[6]
            middle_tip = points[12]
            middle_pip = points[10]
            ring_tip = points[16]
            ring_pip = points[14]
            pinky_tip = points[20]
            pinky_pip = points[18]
            
            # Count extended fingers
            fingers_up = []
            
            # Check if fingers are extended
            finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
            
            for tip, pip in zip(finger_tips, finger_pips):
                if tip[1] < pip[1]:  # Y coordinate decreases upward
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            
            # Get hand center (palm area) for box detection
            wrist = points[0]
            hand_center_x = (index_tip[0] + wrist[0]) / 2
            hand_center_y = (index_tip[1] + wrist[1]) / 2
            
            # Define box areas (normalized coordinates)
            # OFF box: left side (0.1-0.45, 0.2-0.8)
            # ON box: right side (0.55-0.9, 0.2-0.8)
            
            off_box = {
                'x_min': 0.1, 'x_max': 0.45,
                'y_min': 0.2, 'y_max': 0.8
            }
            
            on_box = {
                'x_min': 0.55, 'x_max': 0.9,
                'y_min': 0.2, 'y_max': 0.8
            }
            
            # Check if hand is in OFF box
            if (off_box['x_min'] <= hand_center_x <= off_box['x_max'] and 
                off_box['y_min'] <= hand_center_y <= off_box['y_max']):
                return f"in_off_box|{hand_center_x:.3f},{hand_center_y:.3f}"
            
            # Check if hand is in ON box
            elif (on_box['x_min'] <= hand_center_x <= on_box['x_max'] and 
                  on_box['y_min'] <= hand_center_y <= on_box['y_max']):
                return f"in_on_box|{hand_center_x:.3f},{hand_center_y:.3f}"
            
            # Hand detected but not in any box
            else:
                return f"hand_detected|{hand_center_x:.3f},{hand_center_y:.3f}"
                
        except Exception as e:
            print(f"❌ Gesture classification error: {e}")
            return "unknown"
    
    def get_current_gesture(self):
        """Get the current detected gesture"""
        return self.current_gesture

# Global gesture detector instance
gesture_detector = HandGestureDetector()