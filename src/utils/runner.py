import torch
from torchvision import datasets

from src.models.gan.application.gan_trainer import GANTrainer
from src.models.knn.application.knn_trainer import KNNTrainer
from src.models.unet.application.unet_trainer import UNetTrainer


def load_dataset(data_dir, transform, batch_size=32):
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(dataset.classes)


def run_models(dataset_name, train_loader, val_loader, num_classes, epochs=5):
    # Run U-Net
    unet_trainer = UNetTrainer(num_classes=num_classes)
    train_accuracies, val_accuracies, unet_time = unet_trainer.run_unet(train_loader, val_loader, epochs=epochs)

    # Run KNN
    knn_trainer = KNNTrainer()
    knn_accuracy, knn_time = knn_trainer.run_knn(train_loader, val_loader)

    result = {
        'Dataset': dataset_name,
        'U-Net Accuracy': val_accuracies[-1] * 100,
        'U-Net Time': unet_time,
        'KNN Accuracy': knn_accuracy * 100,
        'KNN Time': knn_time
    }

    return train_accuracies, val_accuracies, knn_accuracy, result

def run_gan(train_loader, epochs=5):
    trainer = GANTrainer()
    gan_time = trainer.train(train_loader, epochs)
    return trainer.generator, gan_time