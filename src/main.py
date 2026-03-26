import os

from src.utils.runner import run_models, load_dataset
from src.utils.plot import plot_results

from torchvision import transforms
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
        'xray': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'xray'))
    }

    results = []
    train_acc_history = {}
    val_acc_history = {}
    knn_acc_history = {}

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

    # Plot
    plot_results(train_acc_history, val_acc_history, knn_acc_history)

    # Results table
    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    main()