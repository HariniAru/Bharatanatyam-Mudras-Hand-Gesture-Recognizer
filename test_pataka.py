import numpy as np
from mudra_rules import is_pataka

# path = "mudra_data/pataka/pataka_001.npy"
# sample = np.load(path)
# print("Sample: ", path)
# print("Sample shape:", sample.shape)
# print("Is Pataka?", is_pataka(sample))

for i in range(1, 16):
    # Format the sample number to be 3 digits (e.g., 001, 002, ..., 015)
    sample_number = f"{i:03d}"
    
    # Construct the file path dynamically
    path = f"mudra_data/pataka/pataka_{sample_number}.npy"
    
    # Load the sample
    sample = np.load(path)
    
    # Print the details for each sample
    print(f"Sample: {path}")
    print(f"Sample shape: {sample.shape}")
    print(f"Is Pataka? {is_pataka(sample)}")
    print("----------")  # Separator for readability