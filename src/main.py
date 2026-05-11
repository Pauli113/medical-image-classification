import os

from torch.utils.data import DataLoader, ConcatDataset

from src.utils.plot_comparisons import plot_all_comparisons
from src.utils.runner import run_models, load_dataset, run_gan, augment_dataset

from torchvision import transforms, datasets
import pandas as pd


def main():
    # define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Directory paths for datasets
    data_dirs = {
        'mri': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mri')),
        'xray': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'xray')),
        'xray_covid_19': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'xray_dataset_covid19'))
    }

    #training without generated images
    results = []
    train_acc_history = {}
    val_acc_history = {}
    knn_acc_history = {}

    print("Datasets without augmentation")

    for dataset_name, data_dir in data_dirs.items():
        print(f'Running models on {dataset_name} dataset...')
        train_loader, val_loader, num_classes = load_dataset(data_dir, transform)
        train_accuracies, val_accuracies, knn_accuracy, result = run_models(
            dataset_name, train_loader, val_loader, num_classes
        )
        train_acc_history[dataset_name] = train_accuracies
        val_acc_history[dataset_name] = val_accuracies
        knn_acc_history[dataset_name] = knn_accuracy
        results.append(result)

    # Results table
    results_df = pd.DataFrame(results)
    print(results_df)

    # Training mit GAN - gespeicherte Bilder laden
    print("Datasets with augmentation")
    for dataset_name, data_dir in data_dirs.items():
        gan_dir = os.path.join(data_dir, 'gan')
        if os.path.exists(gan_dir):
            print(f'Loading GAN images for {dataset_name}...')
            train_loader, val_loader, num_classes = load_dataset(data_dir, transform)
            real_dataset = datasets.ImageFolder(data_dir, transform=transform)
            gan_dataset = datasets.ImageFolder(gan_dir, transform=transform)
            augmented_loader = DataLoader(
                ConcatDataset([real_dataset, gan_dataset]),
                batch_size=32, shuffle=True
            )
            train_accuracies, val_accuracies, knn_accuracy, result = run_models(
                f'{dataset_name}_gan', augmented_loader, val_loader, num_classes
            )
            train_acc_history[f'{dataset_name}_gan'] = train_accuracies
            val_acc_history[f'{dataset_name}_gan'] = val_accuracies
            knn_acc_history[f'{dataset_name}_gan'] = knn_accuracy
            results.append(result)
        else:
            print(f'No gan folder for: {dataset_name} – run gan_save.py first!')


    #all plots
    plot_all_comparisons(train_acc_history, val_acc_history, knn_acc_history)

    # Results table
    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    main()