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

    for dataset_name, data_dir in data_dirs.items():
        print(f'Running models on {dataset_name} dataset...')

        #Load dataset
        dataset = datasets.ImageFolder(data_dir, transform=transform)

        # Split dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        #Run U-Net model
        unet_model = UNet(num_classes=len(dataset.classes))
        timer = Timer()
        timer.start()
        train_accuracies, val_accuracies = unet_model.run_unet(train_loader, val_loader, epochs=5)
        unet_time = timer.stop()
        train_acc_history[dataset_name] = train_accuracies
        val_acc_history[dataset_name] = val_accuracies

        #Run KNN model
        knn_model = KNN()
        timer = Timer()
        timer.start()
        knn_accuracy = knn_model.run_knn(train_loader, val_loader)
        knn_time = timer.stop()
        knn_acc_history[dataset_name] = knn_accuracy

if __name__ == "__main__":
    main()