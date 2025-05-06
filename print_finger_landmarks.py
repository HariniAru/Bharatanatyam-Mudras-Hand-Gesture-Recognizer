import os
import numpy as np

# ✅ Replace this with the mudra you want to inspect
MUDRA_NAME = "ardhapataka"

# Path to the mudra folder
GESTURE_PATH = os.path.join("mudra_data", MUDRA_NAME)

# Map of landmark indices to finger names
FINGER_INDICES = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20],
}

print(f"=== Gesture: {MUDRA_NAME} ===")

# Loop through each .npy file in the gesture folder
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




# ********** THIS PRINTS ALL OF THE SAMPLES EVER **************
# import os
# import numpy as np

# # Path to your data folder
# DATA_DIR = "mudra_data"

# # Map of landmark indices to finger names
# FINGER_INDICES = {
#     "Thumb": [1, 2, 3, 4],
#     "Index": [5, 6, 7, 8],
#     "Middle": [9, 10, 11, 12],
#     "Ring": [13, 14, 15, 16],
#     "Pinky": [17, 18, 19, 20],
# }

# # Loop through all gesture folders
# for gesture_folder in os.listdir(DATA_DIR):
#     gesture_path = os.path.join(DATA_DIR, gesture_folder)
#     if not os.path.isdir(gesture_path):
#         continue

#     print(f"\n=== Gesture: {gesture_folder} ===")

#     # Loop through each .npy file in the gesture folder
#     for filename in sorted(os.listdir(gesture_path)):
#         if not filename.endswith(".npy"):
#             continue

#         file_path = os.path.join(gesture_path, filename)
#         sample = np.load(file_path)

#         print(f"\nSample: {filename}")
#         print(f"Shape: {sample.shape} (should be 21, 3)")

#         for finger_name, indices in FINGER_INDICES.items():
#             print(f"\n{finger_name} landmarks:")
#             for idx in indices:
#                 x, y, z = sample[idx]
#                 print(f"  L{idx}: ({x:.4f}, {y:.4f}, {z:.4f})")

#         print("-" * 40)
