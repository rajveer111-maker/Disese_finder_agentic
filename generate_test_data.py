import numpy as np
import pandas as pd
import os

# Create sample directory
os.makedirs('sample_data', exist_ok=True)

# 1. Synthetic Motor Imagery (BCI) - 1000 timepoints x 22 channels
bci_data = np.random.normal(0, 1, (1000, 22))
# Inject a 10Hz alpha rhythm in some channels
t = np.linspace(0, 4, 1000)
for i in [8, 9, 10]: # C3, Cz, C4 analog
    bci_data[:, i] += np.sin(2 * np.pi * 10 * t) * 5
pd.DataFrame(bci_data, columns=[f"Ch{i}" for i in range(1, 23)]).to_csv('sample_data/test_motor_imagery_bci.csv', index=False)
print("Generated test_motor_imagery_bci.csv")

# 2. Synthetic Parkinson's EEG - 1000 timepoints x 22 channels
pd_data = np.random.normal(0, 1, (1000, 22))
# Inject a slower 4-6Hz theta rhythm often seen in PD
for i in range(22):
    pd_data[:, i] += np.sin(2 * np.pi * 5 * t) * 3
pd.DataFrame(pd_data, columns=[f"Ch{i}" for i in range(1, 23)]).to_csv('sample_data/test_parkinsons_eeg.csv', index=False)
print("Generated test_parkinsons_eeg.csv")

# 3. Synthetic Alzheimer's / Dementia EEG - 1000 timepoints x 22 channels
ad_data = np.random.normal(0, 1, (1000, 22))
# Inject diffuse slow-wave activity (delta/theta) common in severe cognitive decline
for i in range(22):
    ad_data[:, i] += np.sin(2 * np.pi * 2 * t) * 6
pd.DataFrame(ad_data, columns=[f"Ch{i}" for i in range(1, 23)]).to_csv('sample_data/test_alzheimers_eeg.csv', index=False)
print("Generated test_alzheimers_eeg.csv")

# 4. Synthetic Brain MRI - 128x128 structural matrix
# Generate a noisy image with a bright 'tumor-like' blob in the center
mri_img = np.random.normal(50, 10, (128, 128))
x, y = np.ogrid[:128, :128]
mask = ((x - 64)**2 + (y - 64)**2) < 20**2
mri_img[mask] += 150 # Bright blob
# Clip and normalize
mri_img = np.clip(mri_img, 0, 255).astype(np.float32)
np.save('sample_data/test_brain_tumor_mri.npy', mri_img)
print("Generated test_brain_tumor_mri.npy")

print("\nAll synthetic test files successfully generated in the 'sample_data/' directory!")
