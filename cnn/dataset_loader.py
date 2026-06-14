import nibabel as nib
import numpy as np
import pywt
import torch
from torch.utils.data import Dataset
import cv2


class BrainMRIDataset(Dataset):

    def __init__(self, mri_path):
        self.mri = nib.load(mri_path).get_fdata()

        self.slices = []

        # Extract axial slices
        for i in range(self.mri.shape[2]):

            slice_img = self.mri[:, :, i]

            # Skip almost empty slices
            if np.mean(slice_img) < 10:
                continue

            # Resize
            slice_img = cv2.resize(slice_img, (128, 128))

            # Wavelet decomposition
            LL, (LH, HL, HH) = pywt.dwt2(slice_img, 'haar')

            # Resize LL to CNN input size
            LL = cv2.resize(LL, (128, 128))

            self.slices.append(LL)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        image = self.slices[idx]

        image = image.astype(np.float32)

        # Normalize
        image = image / np.max(image)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)

        return torch.tensor(image)


if __name__ == "__main__":

    dataset = BrainMRIDataset(
        "dataset/subject04_t1.mnc.gz"
    )

    print("Total slices:", len(dataset))

    sample = dataset[0]

    print("Sample shape:", sample.shape)