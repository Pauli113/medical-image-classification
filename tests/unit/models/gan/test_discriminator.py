from src.models.gan.domain.discriminator import Discriminator
import torch

def test_discriminator():
    discriminator = Discriminator()
    x = torch.randn(1, 3, 224, 224) #fake image
    output = discriminator(x)
    assert output.shape == (1, 1)
    assert 0 <= output.item() <= 1
