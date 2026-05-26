import os

import torch
from torchvision import datasets

from src.models.gan.application.gan_trainer import GANTrainer
from src.models.knn.application.knn_trainer import KNNTrainer
from src.models.pix2pix.application.pix2pix_trainer import Pix2PixTrainer
from src.models.unet.application.unet_trainer import UNetTrainer


def load_dataset(data_dir, transform, batch_size=32):
    training_dir = os.path.join(data_dir, 'Training')
    if os.path.exists(training_dir):
        data_dir = training_dir

    dataset = datasets.ImageFolder(
        data_dir,
        transform=transform,
        is_valid_file=lambda x: 'gan' not in x.split(os.sep)
    )

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

def run_pix2pix(train_loader, epochs=5):
    trainer = Pix2PixTrainer()
    generator, pix2pix_time = trainer.train(train_loader, epochs)
    return generator, pix2pix_time
def augment_dataset(generator, train_loader):
    all_images = []
    all_labels = []

    with torch.no_grad():
        for images, labels in train_loader:

            #real images
            all_images.append(images)
            all_labels.append(labels)
            #fake images
            fake_images = generator(images)
            all_images.append(fake_images)
            all_labels.append(labels)

    #combine images and labels
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    #create dataloader
    augmented_dataset = torch.utils.data.TensorDataset(all_images, all_labels)
    return torch.utils.data.DataLoader(augmented_dataset, batch_size=32, shuffle=True)



