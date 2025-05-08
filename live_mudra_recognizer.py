# import cv2
# import mediapipe as mp
# import numpy as np

# # from mudra_rules import is_pataka, is_tripataka
# # from mudra_rules import *
# import mudra_rules

# # Setup MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
#                        min_detection_confidence=0.7, min_tracking_confidence=0.7)

# cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
# if not cap.isOpened():
#     print("Webcam error")
#     exit()

# print("🧠 Live Mudra Recognizer Started — press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)

#     label = "No hand detected"

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

#             if mudra_rules.is_pataka(landmarks):
#                 label = "Detected: Pataka ✅"
#             elif mudra_rules.is_tripataka(landmarks):
#                 label = "Detected: Tripataka ✅"
#             else:
#                 label = "Unknown ❌"

#     # Draw label on frame
#     cv2.putText(frame, label, (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if "✅" in label else (0, 0, 255), 2)

#     cv2.imshow("Live Mudra Recognizer", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# hands.close()
# print("👋 Live recognizer exited.")



import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
clf = joblib.load("gesture_classifier.pkl")

# Feature extraction (same as training)
def vector(a, b):
    return np.array(b) - np.array(a)

def angle_between(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def extract_features(landmarks):
    features = []
    pip_pairs = [(6, 5, 8), (10, 9, 12), (14, 13, 16), (18, 17, 20)]
    for pip, mcp, tip in pip_pairs:
        v1 = vector(landmarks[mcp], landmarks[pip])
        v2 = vector(landmarks[tip], landmarks[pip])
        features.append(angle_between(v1, v2))
    wrist = landmarks[0]
    thumb_base = landmarks[2]
    thumb_tip = landmarks[4]
    v1 = vector(wrist, thumb_base)
    v2 = vector(thumb_base, thumb_tip)
    features.append(angle_between(v1, v2))
    return features

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
print("🧠 Live Mudra Recognizer Started — press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label = "No hand detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            features = extract_features(landmarks)

            # Predict with classifier
            prediction = clf.predict([features])[0]
            label = f"Detected: {prediction} ✅"

    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if "✅" in label else (0, 0, 255), 2)

    cv2.imshow("Live Mudra Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("👋 Live recognizer exited.")

