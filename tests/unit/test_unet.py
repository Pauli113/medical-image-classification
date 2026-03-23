import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.unet.unet import UNet, encoder

def test_unet_forward():
    model = UNet(num_classes=3)
    x = torch.randn(1, 3, 224, 224)  # 1 Bild, 3 Farbkanäle, 224x224
    output = model(x)
    assert output.shape == (1, 3)


def test_unet_encoder():
    block = encoder(3,64)
    x = torch.rand(1,3,224,224)
    output = block(x)
    assert output.shape == (1,64,224,224)

def test_unet_predict():
    model = UNet(num_classes=3)
    x = torch.randn(1, 3, 224, 224)  # 1 Bild, 3 Farbkanäle, 224x224
    output = model(x)
    prediction = torch.argmax(output, dim=1)
    assert prediction.item() in [0,1,2]

def test_unet_train():
    model = UNet(num_classes=3)
    # Fake Bilder und Labels
    X = torch.randn(10, 3, 224, 224)  # 10 Bilder
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # Labels

    #Dataset und TrainLoader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)
    [train_acc, val_acc] = model.run_unet(train_loader, val_loader, epochs=2)
    assert len(train_acc) == 2
    assert len(val_acc) == 2