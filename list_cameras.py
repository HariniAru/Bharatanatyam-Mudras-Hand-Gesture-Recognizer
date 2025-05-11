# Lists available camera indices 
import cv2

print("Scanning camera indices...")

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ret, _ = cap.read()
        print(f"Index {i} -> Opened: {ret}")
        cap.release()
    else:
        print(f"Index {i} -> Not available")
