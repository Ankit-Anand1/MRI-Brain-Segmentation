import nibabel as nib
import numpy as np
import pywt
import cv2
import torch
from torch.utils.data import Dataset


class BrainTrainingDataset(Dataset):

    def __init__(self, mri_path, label_path):

        self.mri = nib.load(mri_path).get_fdata()
        self.labels = nib.load(label_path).get_fdata()

        self.images = []
        self.masks = []

        for i in range(self.mri.shape[2]):

            img_slice = self.mri[:, :, i]
            label_slice = self.labels[:, :, i]

            # Skip empty slices
            if np.mean(img_slice) < 10:
                continue

            # Keep only WM/GM/CSF
            mask = np.zeros_like(label_slice)

            mask[label_slice == 1] = 1
            mask[label_slice == 2] = 2
            mask[label_slice == 3] = 3

            # Resize
            img_slice = cv2.resize(img_slice, (128, 128))
            mask = cv2.resize(
                mask,
                (128, 128),
                interpolation=cv2.INTER_NEAREST
            )

            # Wavelet
            LL, (LH, HL, HH) = pywt.dwt2(img_slice, "haar")
            LL = cv2.resize(LL, (128, 128))

            # Normalize
            LL = LL.astype(np.float32)

            if LL.max() > 0:
                LL = LL / LL.max()

            self.images.append(LL)
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = np.expand_dims(self.images[idx], axis=0)

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            self.masks[idx],
            dtype=torch.long
        )

        return image, mask


if __name__ == "__main__":

    dataset = BrainTrainingDataset(
        "dataset/subject04_t1.mnc.gz",
        "dataset/subject04_crisp.mnc.gz"
    )

    #image, mask = dataset[0]

    #print("Image Shape:", image.shape)
    #print("Mask Shape:", mask.shape)

    #print("Mask Labels:", torch.unique(mask))


    for i in [20, 40, 60, 80, 100]:

        image, mask = dataset[i]

    print(f"\nSlice {i}")

    print("Image Shape:", image.shape)
    print("Mask Labels:", torch.unique(mask))