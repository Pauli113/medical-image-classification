import torch
from torch.utils.data import TensorDataset, DataLoader
from src.models.gan.application.gan_trainer import GANTrainer



def test_gan_trainer():
    # Fake Bilder und Labels für DataLoader
    X = torch.randn(10, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)

    #training
    trainer = GANTrainer()
    gan_time = trainer.train(train_loader, epochs=1)
    assert isinstance(gan_time, str)
    assert len(gan_time) > 0