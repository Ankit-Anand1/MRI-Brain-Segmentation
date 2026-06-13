import nibabel as nib
import matplotlib.pyplot as plt

# Load MRI image
img = nib.load("dataset/subject04_t1.mnc.gz")

# Convert to numpy array
data = img.get_fdata()

# Print shape
print("Shape:", data.shape)

# Middle slice
slice_idx = data.shape[2] // 2

# Display image
plt.imshow(data[:, :, slice_idx], cmap="gray")
plt.title("Brain MRI")
plt.axis("off")
plt.show()