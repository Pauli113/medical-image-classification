import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.gan.domain.generator import Generator
from src.utils.runner import augment_dataset


def test_augment_dataset():
    # Fake Bilder und Labels für DataLoader
    X = torch.randn(10, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)

    #generare images
    generator = Generator()
    augment_loader = augment_dataset(generator, train_loader)

    #double the amount of images
    assert len(augment_loader.dataset) == 2*len(dataset)
