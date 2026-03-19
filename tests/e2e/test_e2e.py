import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from src.models.knn.knn import KNN
from src.models.unet.unet import UNet


def test_e2e():
    model = UNet(num_classes=3)
    #Fake Data
    X_train = np.random.rand(30, 150528)
    y_train = np.array([0,1,2] * 10)
    x_img = torch.rand(1, 3, 224, 224)

    #Training
    knn = KNN()
    knn.fit(X_train, y_train)

    output = model(x_img)
    prediction = torch.argmax(output, dim=1)

    knn_pred = knn.predict(X_train[:1])
    assert knn_pred[0] in [0, 1, 2]
    assert prediction.item() in [0, 1, 2]

    #Plot Creation
    knn_accuracy = knn.score(X_train, y_train)  # Accuracy on training data
    plt.bar(['KNN'], [knn_accuracy])
    plt.savefig('test_plot.png')
   # assert os.path.exists('test_plot.png')
   # os.remove('test_plot.png')


