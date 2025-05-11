# Prints accuracy of mudra rules for a specified mudra
import os
import numpy as np
import mudra_rules

# mudra to test rule for 
MUDRA_NAME = "mushti" # modify this for different mudras

# map mudra name to respective function from mudra_rules
mudra_functions = {
    "pataka": mudra_rules.is_pataka,
    "tripataka": mudra_rules.is_tripataka,
    "ardhapataka": mudra_rules.is_ardhapataka,
    "mushti": mudra_rules.is_mushti
    # rest of mudras
}

check_fn = mudra_functions[MUDRA_NAME]
folder_path = f"mudra_data/{MUDRA_NAME}"

# loop through mudra samples
for filename in sorted(os.listdir(folder_path)):
    if not filename.endswith(".npy"):
        continue

    path = os.path.join(folder_path, filename)
    sample = np.load(path)

    print(f"Sample: {path}")
    print(f"Sample shape: {sample.shape}")
    print(f"Is {MUDRA_NAME.capitalize()}? {check_fn(sample)}")
    print("----------")