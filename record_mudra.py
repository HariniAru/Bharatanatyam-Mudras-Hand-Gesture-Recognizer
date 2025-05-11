# Record mudra samples for sample data
import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# check for mudra input parameter
if len(sys.argv) < 2:
    print("Usage: python record_mudra.py <MUDRA_LABEL>\n")
    print("Please provide the name of the mudra you want to record samples for after the command.")
    print("(Example: python record_mudra.py mushti)\n")
    sys.exit(1)

if (sys.argv[1]):
    MUDRA_LABEL = sys.argv[1]
else:
    MUDRA_LABEL = "mushti"

print(f"Recording gesture: {MUDRA_LABEL}")

SAVE_DIR = f"mudra_data/{MUDRA_LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

# set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(1) # you may need to switch out 1 for whatever camera input you have
# cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Webcam error")
    exit()

sample_count = len(os.listdir(SAVE_DIR))
print(f"Recording for mudra: {MUDRA_LABEL}")
print("Hold your hand in position, then press 's' to save, or 'q' to quit.")

# record mudra samples
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # convert landmarks to numpy array
            landmark_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32
            )

            # save when 's' is pressed
            if key == ord('s'):
                filename = f"{MUDRA_LABEL}_{sample_count:03d}.npy"
                np.save(os.path.join(SAVE_DIR, filename), landmark_array)
                print(f"✅ Saved: {filename}")
                sample_count += 1

    cv2.putText(frame, f"Mudra: {MUDRA_LABEL} | Samples: {sample_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Record Mudra: Pataka", frame)

    if key == ord('q') or key == 27:
        break

# for exiting
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done recording.")
