import nibabel as nib
import numpy as np

img = nib.load("dataset/subject04_crisp.mnc.gz")
data = img.get_fdata()

print("Shape:", data.shape)

print("Unique labels:")
print(np.unique(data))