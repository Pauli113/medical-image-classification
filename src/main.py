import os

import torch, torchvision
from src.models.knn.knn import KNN
from src.models.unet.unet import UNet
from src.utils.timer import Timer
import matplotlib
import pandas
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
def main():
    #define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Directory paths for datasets
data_dirs = {
    'mri': os.path.join(os.path.dirname(__file__), '..', 'data', 'mri'),
    'xray': os.path.join(os.path.dirname(__file__), '..', 'data', 'xray')
}

# Correct the paths to reference the parent directory
for key in data_dirs.keys():
    data_dirs[key] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', key))

# Store results in a DataFrame
results = []
train_acc_history = {}
val_acc_history = {}
knn_acc_history = {}


print("Hallo")

if __name__ == "__main__":
    main()