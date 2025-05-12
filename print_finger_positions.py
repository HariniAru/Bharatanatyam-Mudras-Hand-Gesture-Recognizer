# Prints finger positions for each sample in mudra_data
import os
import numpy as np

# finger landmark indices
FINGER_INDICES = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20],
    "Wrist": [0]
}

data_dir = "mudra_data"

# loop through each mudra sample and print finger positions
num_samples = 0
for mudra_folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, mudra_folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n=== Mudra: {mudra_folder} ===")

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".npy"):
            num_samples += 1
            path = os.path.join(folder_path, file)
            landmarks = np.load(path)

            print(f"\nSample: {file}")
            print(f"Shape: {landmarks.shape}")

            for finger, indices in FINGER_INDICES.items():
                print(f"  {finger}:")
                for idx in indices:
                    x, y, z = landmarks[idx]
                    print(f"    [{idx}] x={x:.4f}, y={y:.4f}, z={z:.4f}")

print("\n")
print("Evaluated for a total of", num_samples, "samples in mudra_data")