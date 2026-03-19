import torch
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
    pass

