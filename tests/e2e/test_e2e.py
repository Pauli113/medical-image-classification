import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader

from src.models.knn.knn import KNN
from src.models.unet.unet import UNet


def test_e2e():
    model = UNet(num_classes=3)
    #Fake Data
    X_train = np.random.rand(30, 150528)
    y_train = np.array([0,1,2] * 10)
    x_img = torch.rand(1, 3, 224, 224)

    #Training
    #KNN
    knn = KNN()
    knn.fit(X_train, y_train)

    output = model(x_img)
    prediction = torch.argmax(output, dim=1)

    knn_pred = knn.predict(X_train[:1])
    assert knn_pred[0] in [0, 1, 2]
    assert prediction.item() in [0, 1, 2]

    #UNET
    model = UNet(num_classes=3)
    # Fake Bilder und Labels
    X = torch.randn(10, 3, 224, 224)  # 10 Bilder
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # Labels

    #Dataset und TrainLoader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)
    [train_acc, val_acc] = model.train_model(train_loader, val_loader, epochs=2)
    assert len(train_acc) == 2
    assert len(val_acc) == 2

    #Plot Creation
    knn_accuracy = knn.score(X_train, y_train)  # Accuracy on training data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
    #unet on top
    ax1.plot(train_acc, label="U-Net Train")
    ax1.plot(val_acc, label="U-Net Val")
    ax1.set_title('U-Net Training and Validation Accuracies')
    plt.bar(['KNN'], [knn_accuracy])
    plt.savefig('test_plot.png')
    assert os.path.exists('test_plot.png')
    os.remove('test_plot.png')


