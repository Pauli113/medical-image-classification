from src.models.gan.domain.generator import Generator
import torch

def test_generate_image():
    generator = Generator()
    x = torch.randn(1, 3, 224, 224) #fake image
    output = generator(x)
    assert output.shape == (1, 3, 224, 224)