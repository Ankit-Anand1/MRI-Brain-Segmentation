import nibabel as nib
import pywt
import matplotlib.pyplot as plt

img = nib.load("dataset/subject04_t1.mnc.gz")
data = img.get_fdata()

#slice_idx = data.shape[0] // 2
#slice_img = data[slice_idx, :, :]

slice_idx = data.shape[2] // 2
slice_img = data[:, :, slice_idx]

LL, (LH, HL, HH) = pywt.dwt2(slice_img, 'haar')

fig, ax = plt.subplots(1,4, figsize=(12,4))

ax[0].imshow(LL, cmap='gray')
ax[0].set_title("LL")

ax[1].imshow(LH, cmap='gray')
ax[1].set_title("LH")

ax[2].imshow(HL, cmap='gray')
ax[2].set_title("HL")

ax[3].imshow(HH, cmap='gray')
ax[3].set_title("HH")

plt.show()