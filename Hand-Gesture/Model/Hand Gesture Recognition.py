import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def detect_gesture(landmarks):
    # Extract landmark positions
    thumb_tip   = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_cmc   = landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    index_tip   = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp   = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip  = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp  = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip    = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp    = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip   = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp   = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # Determine finger status (True if extended)
    thumb_extended = thumb_tip.y < thumb_cmc.y
    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    ring_extended = ring_tip.y < ring_mcp.y
    pinky_extended = pinky_tip.y < pinky_mcp.y

    # Count extended fingers
    extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])

    # Determine gesture
    if extended_count == 5:
        gesture = "Palm"
    elif extended_count == 0:
        gesture = "Closed Fist"
    elif thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
        gesture = "Gun"
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        gesture = "Peace Sign"
    elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        gesture = "Thumbs Up"
    elif index_extended and not thumb_extended and not middle_extended and not ring_extended and not pinky_extended:
        gesture = "Pointing Finger"
    elif middle_extended and thumb_extended and not index_extended and not ring_extended and not pinky_extended:
        gesture = "Middle Finger"
    elif pinky_extended and thumb_extended and index_extended and not middle_extended and not ring_extended:
        gesture = "Rock and Roll"
    elif pinky_extended and thumb_extended and not index_extended and not middle_extended and not ring_extended:
        gesture = "Surf Ups"
    else:
        gesture = "Unknown"
    
    return gesture

def draw_hand_overlay(frame, hand_landmarks, gesture, color=(255, 0, 0)):
    h, w, _ = frame.shape
    # Get bounding box from landmarks
    x_vals = [lm.x for lm in hand_landmarks.landmark]
    y_vals = [lm.y for lm in hand_landmarks.landmark]
    x_min = int(min(x_vals) * w) - 10
    x_max = int(max(x_vals) * w) + 10
    y_min = int(min(y_vals) * h) - 10
    y_max = int(max(y_vals) * h) + 10

    # Draw bounding box with blue color
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # Background for text for better readability using the same blue color as the box
    (text_width, text_height), baseline = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x_min, y_min - text_height - baseline - 10), 
                  (x_min + text_width + 10, y_min), color, -1)

    # Put gesture text above the bounding box in white
    cv2.putText(frame, gesture, (x_min + 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define custom drawing specifications for blue color (BGR: (255, 0, 0))
    landmark_spec = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
    connection_spec = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip for mirror view and convert to RGB for processing
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame with custom blue specs
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         landmark_spec, connection_spec)
                gesture = detect_gesture(hand_landmarks)
                draw_hand_overlay(frame, hand_landmarks, gesture, color=(255, 0, 0))

                # Trigger action for Middle Finger gesture
                if gesture == "Middle Finger":
                    cv2.putText(frame, "Warning: Middle Finger Detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 3)
                    #os.system("shutdown /s /t 1")
        
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
