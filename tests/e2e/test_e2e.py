import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader

from src.models.knn.knn import KNN
from src.models.unet.unet import UNet
from src.utils.plot import plot_results
from src.utils.runner import run_models


def test_e2e():
    # Fake Bilder und Labels für DataLoader
    X = torch.randn(10, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    # KNN mit DataLoader
    knn = KNN()
    knn_accuracy, knn_time = knn.run_knn(train_loader, val_loader)
    assert knn_accuracy >= 0

    # UNet
    model = UNet(num_classes=3)
    x_img = torch.rand(1, 3, 224, 224)
    output = model(x_img)
    prediction = torch.argmax(output, dim=1)
    assert prediction.item() in [0, 1, 2]

    train_acc, val_acc, unet_time = model.run_unet(train_loader, val_loader, epochs=2)
    assert len(train_acc) == 2
    assert len(val_acc) == 2

    # run_models testen
    train_accuracies, val_accuracies, knn_accuracy, result = run_models(
        'fake', train_loader, val_loader, num_classes=3, epochs=2
    )
    assert 'Dataset' in result
    assert 'U-Net Accuracy' in result
    assert 'KNN Accuracy' in result

    # Plot Creation
    train_acc_history = {'fake': train_acc}
    val_acc_history = {'fake': val_acc}
    knn_acc_history = {'fake': knn_accuracy}

    plot_results(train_acc_history, val_acc_history, knn_acc_history)
    assert os.path.exists('results.png')
    os.remove('results.png')