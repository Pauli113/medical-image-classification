import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader

from src.models.knn.knn import KNN
from src.models.unet.unet import UNet


def test_e2e():
    # Fake Bilder und Labels für DataLoader
    X = torch.randn(10, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    # KNN mit DataLoader
    knn = KNN()
    knn_accuracy = knn.run_knn(train_loader, val_loader)
    assert knn_accuracy >= 0

    # UNet
    model = UNet(num_classes=3)
    x_img = torch.rand(1, 3, 224, 224)
    output = model(x_img)
    prediction = torch.argmax(output, dim=1)
    assert prediction.item() in [0, 1, 2]

    train_acc, val_acc = model.run_unet(train_loader, val_loader, epochs=2)
    assert len(train_acc) == 2
    assert len(val_acc) == 2

    # Plot Creation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # UNet oben
    ax1.plot(train_acc, label="U-Net Train")
    ax1.plot(val_acc, label="U-Net Val")
    ax1.set_title('U-Net Training and Validation Accuracies')
    ax1.legend()
    # KNN unten
    ax2.bar(['KNN'], [knn_accuracy])
    ax2.set_title('KNN Validation Accuracy')
    plt.tight_layout()
    plt.savefig('test_plot.png')
    assert os.path.exists('test_plot.png')
    os.remove('test_plot.png')