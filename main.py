# Tests whether opening webcam with MediaPipe works
import cv2
import time
import os
import mediapipe as mp

# setting up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# access webcam with OpenCV
# cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

print("Webcam opened! Press 'q' to quit or ESC to close.")

frame_saved = False

while True:
    ret, frame = cap.read()
    print("Captured frame:", ret)

    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    if not frame_saved:
        cv2.imwrite("test_frame.png", frame)
        print("Saved a test frame as test_frame.png")
        frame_saved = True

    cv2.imshow('Hand Tracking', frame)

    time.sleep(0.05)

    key = cv2.waitKey(1) & 0xFF
    if (key == ord('q') or key == 27):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Closed webcam and exited.")