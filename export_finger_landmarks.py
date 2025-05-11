# Exports finger landmarks to a separate .txt file for easier readability
import os
import numpy as np

# provide mudra name
MUDRA_NAME = "mushti"
GESTURE_PATH = os.path.join("mudra_data", MUDRA_NAME)
OUTPUT_FILE = f"{MUDRA_NAME}_landmarks.txt"

FINGER_INDICES = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20],
}

# create output file and write landmark info to output file
with open(OUTPUT_FILE, "w") as f:
    f.write(f"=== Gesture: {MUDRA_NAME} ===\n")

    for filename in sorted(os.listdir(GESTURE_PATH)):
        if not filename.endswith(".npy"):
            continue

        file_path = os.path.join(GESTURE_PATH, filename)
        sample = np.load(file_path)

        f.write(f"\nSample: {filename}\n")
        f.write(f"Shape: {sample.shape} (should be 21, 3)\n")

        for finger_name, indices in FINGER_INDICES.items():
            f.write(f"\n{finger_name} landmarks:\n")
            for idx in indices:
                x, y, z = sample[idx]
                f.write(f"  L{idx}: ({x:.4f}, {y:.4f}, {z:.4f})\n")

        f.write("-" * 40 + "\n")

# verification
print(f"Landmarks written to '{OUTPUT_FILE}'")
