import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.pix2pix.application.pix2pix_trainer import Pix2PixTrainer


def test_pix2pix_trainer():
    # Fake Bilder und Labels für DataLoader
    X = torch.randn(10, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)

    #training
    trainer = Pix2PixTrainer()
    generator, time = trainer.train(train_loader, epochs=1)
    assert isinstance(time, str)
    assert len(time) > 0
