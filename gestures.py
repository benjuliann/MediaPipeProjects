import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions

import time

# Define hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize HandLandmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

pTime = cv2.getTickCount()
fps_buffer = []
frame_count = 0
fps_display = 0

# Initialize Gesture Recognizer
gesture_options = GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path='gesture_recognizer.task'),
    num_hands=2
)
gesture_recognizer = GestureRecognizer.create_from_options(gesture_options)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    # Convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image and process
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    gesture_results = None
    results = landmarker.detect(mp_image)
    frame_count += 1

    if results.hand_landmarks:
        gesture_results = gesture_recognizer.recognize(mp_image)
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Get handedness (Left or Right) - reverse it because camera is flipped
            hand_label = results.handedness[idx][0].category_name
            hand_label = "Left" if hand_label == "Right" else "Right"

            xs = [lm.x for lm in hand_landmarks]
            ys = [lm.y for lm in hand_landmarks]
            x1, y1 = int(min(xs)*w), int(min(ys)*h)
            x2, y2 = int(max(xs)*w), int(max(ys)*h)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            
            # Display hand label
            x_pos = int(hand_landmarks[0].x * w)
            y_pos = int(hand_landmarks[0].y * h) - 20
            cv2.putText(img, hand_label, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            # Draw gesture 
            if gesture_results.gestures and len(gesture_results.gestures) > idx: 
                gesture = gesture_results.gestures[idx][0].category_name 
                cv2.putText(img, gesture,
                            (x_pos - 40, y_pos + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)
                            
    else:
        gesture_results = None

    # FPS
    cTime = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (cTime - pTime)
    pTime = cTime
    fps_buffer.append(fps)
    if frame_count % 10 == 0:
        fps_display = int(sum(fps_buffer) / len(fps_buffer)) 
        fps_buffer = [] # reset buffer

    cv2.putText(img, f'FPS: {fps_display}', (10, 60),
    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()