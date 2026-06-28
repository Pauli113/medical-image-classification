import os
import torch
from torchvision import datasets
from torch.utils.data import ConcatDataset

from src.models.knn.application.knn_trainer import KNNTrainer
from src.models.pix2pix.application.pix2pix_trainer import Pix2PixTrainer
from src.models.unet.application.unet_trainer import UNetTrainer


def load_dataset(data_dir, transform, batch_size=32):
    training_dir = os.path.join(data_dir, 'Training')
    testing_dir = os.path.join(data_dir, 'Testing')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # ignore generated image folders
    is_valid = lambda x: all(
        folder not in ['pix2pix']
        for folder in x.split(os.sep)
    )

    if os.path.exists(training_dir) and os.path.exists(testing_dir):
        # MRI - combine Training and Testing
        train_data = datasets.ImageFolder(training_dir, transform=transform, is_valid_file=is_valid)
        test_data = datasets.ImageFolder(testing_dir, transform=transform, is_valid_file=is_valid)
        dataset = ConcatDataset([train_data, test_data])
        num_classes = len(train_data.classes)

    elif os.path.exists(train_dir) and os.path.exists(test_dir):
        # xray_covid_19 - combine train and test
        train_data = datasets.ImageFolder(train_dir, transform=transform, is_valid_file=is_valid)
        test_data = datasets.ImageFolder(test_dir, transform=transform, is_valid_file=is_valid)
        dataset = ConcatDataset([train_data, test_data])
        num_classes = len(train_data.classes)

    else:
        # xray - filter out generated image folders
        dataset = datasets.ImageFolder(
        data_dir,
        transform=transform,
        is_valid_file=is_valid,
        allow_empty=True  # ← erlaubt leere Klassen
        )
        valid_classes = sorted([
             c for c in dataset.classes
            if c not in ['pix2pix']
        ])
        valid_idx = {dataset.class_to_idx[c] for c in valid_classes}
        dataset.samples = [(p, l) for p, l in dataset.samples if l in valid_idx]
        dataset.targets = [l for l in dataset.targets if l in valid_idx]
        dataset.classes = valid_classes
        dataset.class_to_idx = {c: i for i, c in enumerate(valid_classes)}
        num_classes = len(valid_classes)

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes


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


def run_pix2pix(train_loader, epochs=5):
    trainer = Pix2PixTrainer()
    generator, pix2pix_time = trainer.train(train_loader, epochs)
    return generator, pix2pix_time