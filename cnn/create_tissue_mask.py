import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

img = nib.load("dataset/subject04_crisp.mnc.gz")
labels = img.get_fdata()

mask = np.zeros_like(labels)

mask[labels == 1] = 1  # CSF
mask[labels == 2] = 2  # GM
mask[labels == 3] = 3  # WM

print("Unique classes:", np.unique(mask))

slice_idx = mask.shape[2] // 2

plt.imshow(mask[:, :, slice_idx], cmap="jet")
plt.colorbar()
plt.title("WM-GM-CSF Mask")
plt.show()