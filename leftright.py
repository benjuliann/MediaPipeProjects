import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize HandLandmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options)
landmarker = vision.HandLandmarker.create_from_options(options)

pTime = 0
cTime = 0

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
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Get handedness (Left or Right) - reverse it because camera is flipped
            hand_label = results.handedness[idx][0].category_name
            hand_label = "Left" if hand_label == "Right" else "Right"
            
            # Choose color based on hand (Right = Blue, Left = Green)
            line_color = (255, 0, 0) if hand_label == "Right" else (0, 255, 0)
            circle_color = (255, 0, 0) if hand_label == "Right" else (0, 255, 0)
            
            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(img, (x1, y1), (x2, y2), line_color, 2)

            # Draw landmarks and highlight fingertips
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Circles on fingertips
                if id in [4, 8, 12, 16, 20]:
                    cv2.circle(img, (cx, cy), 10, circle_color, cv2.FILLED)
                else:
                    cv2.circle(img, (cx, cy), 5, circle_color, cv2.FILLED)
            
            # Display hand label
            x_pos = int(hand_landmarks[0].x * w)
            y_pos = int(hand_landmarks[0].y * h) - 20
            cv2.putText(img, hand_label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        circle_color, 2)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'fps:' + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()