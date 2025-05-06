import os
import numpy as np
# from mudra_rules import is_pataka, is_tripataka, is_ardhapataka
import mudra_rules

# Set which mudra you want to test
MUDRA_NAME = "ardhapataka"  # change to "tripataka" or "ardhapataka"

# Mapping of mudra name to function
mudra_functions = {
    "pataka": mudra_rules.is_pataka,
    "tripataka": mudra_rules.is_tripataka,
    "ardhapataka": mudra_rules.is_ardhapataka
}

# Select the function
check_fn = mudra_functions[MUDRA_NAME]

# Folder path
folder_path = f"mudra_data/{MUDRA_NAME}"

# Loop through files
for filename in sorted(os.listdir(folder_path)):
    if not filename.endswith(".npy"):
        continue

    path = os.path.join(folder_path, filename)
    sample = np.load(path)

    print(f"Sample: {path}")
    print(f"Sample shape: {sample.shape}")
    print(f"Is {MUDRA_NAME.capitalize()}? {check_fn(sample)}")
    print("----------")
