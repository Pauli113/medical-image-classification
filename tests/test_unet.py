import torch
from src.models.unet.unet import UNet

def test_unet_forward():
    model = UNet(num_classes=3)
    x = torch.randn(1, 3, 224, 224)  # 1 Bild, 3 Farbkanäle, 224x224
    output = model(x)
    assert output.shape == (1, 3)

def test_unet_encoder():
    pass

def test_unet_predict():
    pass

