# Prints finger landmarks for a specific mudra
import os
import numpy as np

MUDRA_NAME = "ardhapataka" # mudra to print landmarks for, modify here

# path to mudra data
GESTURE_PATH = os.path.join("mudra_data", MUDRA_NAME)

# finger landmark indices
FINGER_INDICES = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20],
}

print(f"=== Gesture: {MUDRA_NAME} ===")

# go through each .npy sample file for mudra
for filename in sorted(os.listdir(GESTURE_PATH)):
    if not filename.endswith(".npy"):
        continue

    file_path = os.path.join(GESTURE_PATH, filename)
    sample = np.load(file_path)

    print(f"\nSample: {filename}")
    print(f"Shape: {sample.shape} (should be 21, 3)")

    for finger_name, indices in FINGER_INDICES.items():
        print(f"\n{finger_name} landmarks:")
        for idx in indices:
            x, y, z = sample[idx]
            print(f"  L{idx}: ({x:.4f}, {y:.4f}, {z:.4f})")

    print("-" * 40)