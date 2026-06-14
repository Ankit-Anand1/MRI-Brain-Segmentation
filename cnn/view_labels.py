import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load("dataset/subject04_crisp.mnc.gz")
data = img.get_fdata()

slice_idx = data.shape[2] // 2

plt.imshow(data[:, :, slice_idx], cmap="jet")
plt.colorbar()
plt.title("Ground Truth Labels")
plt.show()