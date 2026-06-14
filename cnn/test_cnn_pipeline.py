import torch

from cnn_model import BrainCNN
from dataset_loader import BrainMRIDataset

# Load dataset
dataset = BrainMRIDataset(
    "dataset/subject04_t1.mnc.gz"
)

# Load CNN
model = BrainCNN()

# Take one MRI slice
sample = dataset[0]

# Add batch dimension
sample = sample.unsqueeze(0)

print("Input Shape:", sample.shape)

# Forward pass
output = model(sample)

print("Output Shape:", output.shape)

print("Output:", output)