import os
import numpy as np
import cv2

# Set the folder you want to preview
MUDRA_FOLDER = "mudra_data/mushti"  # <-- change if needed

# Helper to draw 2D skeleton
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),      # Thumb
    (0,5), (5,6), (6,7), (7,8),      # Index
    (5,9), (9,10), (10,11), (11,12),# Middle
    (9,13), (13,14), (14,15), (15,16),  # Ring
    (13,17), (17,18), (18,19), (19,20), # Pinky
    (0,17)  # Palm arc
]

def draw_hand(image, landmarks):
    h, w = image.shape[:2]
    for i, (x, y, z) in enumerate(landmarks):
        cx, cy = int(x * w), int(y * h)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        x0, y0 = int(landmarks[start][0] * w), int(landmarks[start][1] * h)
        x1, y1 = int(landmarks[end][0] * w), int(landmarks[end][1] * h)
        cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return image

# Load and preview each file
for filename in sorted(os.listdir(MUDRA_FOLDER)):
    if not filename.endswith(".npy"):
        continue

    filepath = os.path.join(MUDRA_FOLDER, filename)
    landmarks = np.load(filepath)

    # Create a blank image
    canvas = np.ones((480, 480, 3), dtype=np.uint8) * 255
    canvas = draw_hand(canvas, landmarks)

    cv2.putText(canvas, filename, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("Preview", canvas)
    # key = cv2.waitKey(0) # you gotta manually press a key to transition for this
    key = cv2.waitKey(1000)  # 1000 ms = 1 second per frame
    if key == ord('q') or key == 27:  # ESC or 'q' to quit
        break

cv2.destroyAllWindows()
