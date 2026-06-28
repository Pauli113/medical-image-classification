import os
from torch.utils.data import DataLoader, ConcatDataset
from src.utils.plot_comparisons import plot_all_comparisons
from src.utils.runner import run_models, load_dataset
from torchvision import transforms, datasets
import pandas as pd


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dirs = {
        'mri': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mri')),
        'xray': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'xray')),
        'xray_covid_19': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'xray_dataset_covid19'))
    }

    results = []
    train_acc_history = {}
    val_acc_history = {}
    knn_acc_history = {}

    # MRI baseline already computed - hardcoded
    train_acc_history['mri'] = [0.3470, 0.3707, 0.3669, 0.3838, 0.3953]
    val_acc_history['mri'] = [0.3645, 0.3538, 0.3920, 0.3706, 0.3982]
    knn_acc_history['mri'] = 0.8315

    train_acc_history['mri_10'] = [0.3022, 0.3451, 0.3815, 0.3949, 0.4546, 0.4944, 0.5400, 0.5672, 0.5921, 0.6021]
    val_acc_history['mri_10'] = [0.4717, 0.3583, 0.4518, 0.4931, 0.5391, 0.5314, 0.5712, 0.6172, 0.5804, 0.5881]
    knn_acc_history['mri_10'] = 0.8315

    train_acc_history['mri_20'] = [0.3332, 0.3711, 0.3818, 0.3799, 0.3857, 0.3822, 0.4328, 0.4699, 0.5190, 0.5224, 0.5477, 0.5776, 0.6090, 0.6212, 0.6151, 0.6312, 0.6312, 0.6442, 0.6377, 0.6538]
    val_acc_history['mri_20'] = [0.4165, 0.3553, 0.4104, 0.4334, 0.4456, 0.3706, 0.4487, 0.5467, 0.5145, 0.5345, 0.5804, 0.5559, 0.6309, 0.6493, 0.6340, 0.6386, 0.6462, 0.6325, 0.6172, 0.6524]
    knn_acc_history['mri_20'] = 0.8315

    # xray baseline already computed - hardcoded
    train_acc_history['xray'] = [0.4747, 0.5643, 0.6989, 0.7551, 0.8295]
    val_acc_history['xray'] = [0.5134, 0.6128, 0.7103, 0.8489, 0.8929]
    knn_acc_history['xray'] = 0.9254

    train_acc_history['xray_10'] = [0.4237, 0.5469, 0.5930, 0.6887, 0.7245, 0.7733, 0.8532, 0.8678, 0.8737, 0.8912]
    val_acc_history['xray_10'] = [0.4885, 0.5975, 0.6071, 0.6367, 0.8088, 0.8662, 0.8623, 0.8824, 0.9111, 0.8891]
    knn_acc_history['xray_10'] = 0.9254

    train_acc_history['xray_20'] = [0.4558, 0.5607, 0.6162, 0.6485, 0.7068, 0.7575, 0.6418, 0.4084, 0.3185, 0.3336, 0.3391, 0.3336, 0.3369, 0.3424, 0.3448, 0.3403, 0.3465, 0.3369, 0.3405, 0.3439]
    val_acc_history['xray_20'] = [0.5163, 0.6912, 0.5927, 0.7371, 0.7648, 0.7457, 0.6109, 0.3145, 0.3451, 0.3451, 0.3451, 0.3451, 0.3403, 0.3403, 0.3403, 0.3403, 0.3451, 0.3451, 0.3451, 0.3451]
    knn_acc_history['xray_20'] = 0.9254

    # xray_covid_19 baseline already computed - hardcoded
    train_acc_history['xray_covid_19'] = [0.4733, 0.5467, 0.5733, 0.6267, 0.5600]
    val_acc_history['xray_covid_19'] = [0.6842, 0.5526, 0.6579, 0.5000, 0.5263]
    knn_acc_history['xray_covid_19'] = 0.9211

    train_acc_history['xray_covid_19_10'] = [0.5400, 0.5333, 0.5400, 0.5933, 0.5867, 0.6667, 0.6533, 0.6667, 0.6533, 0.7000]
    val_acc_history['xray_covid_19_10'] = [0.4474, 0.5263, 0.5000, 0.5526, 0.5263, 0.6579, 0.6579, 0.5526, 0.6579, 0.6579]
    knn_acc_history['xray_covid_19_10'] = 0.9211

    train_acc_history['xray_covid_19_20'] = [0.5067, 0.5133, 0.6667, 0.6333, 0.6467, 0.7067, 0.7000, 0.6800, 0.6933, 0.7133, 0.6867, 0.7067, 0.6933, 0.7467, 0.7667, 0.7933, 0.6667, 0.7533, 0.7600, 0.8200]
    val_acc_history['xray_covid_19_20'] = [0.4737, 0.5526, 0.5789, 0.5789, 0.6316, 0.6316, 0.6579, 0.6316, 0.6579, 0.5789, 0.6579, 0.6579, 0.6579, 0.6842, 0.6842, 0.5526, 0.7105, 0.7368, 0.6842, 0.7105]
    knn_acc_history['xray_covid_19_20'] = 0.8947

    # skip baseline for all - already computed
    skip_baseline = ['mri', 'xray', 'xray_covid_19']

    # without augmentation - 5, 10, 20 epochs
    print("Datasets without augmentation")
    for dataset_name, data_dir in data_dirs.items():
        if dataset_name in skip_baseline:
            print(f'Skipping baseline for {dataset_name} - already computed')
            continue

        train_loader, val_loader, num_classes = load_dataset(data_dir, transform)
        for epochs in [5, 10, 20]:
            print(f'Running models on {dataset_name} dataset - {epochs} epochs...')
            key = f'{dataset_name}_{epochs}' if epochs != 5 else dataset_name
            train_acc, val_acc, knn_acc, result = run_models(
                key, train_loader, val_loader, num_classes, epochs=epochs
            )
            train_acc_history[key] = train_acc
            val_acc_history[key] = val_acc
            knn_acc_history[key] = knn_acc
            results.append(result)

    results_df = pd.DataFrame(results)
    print(results_df)

    # with Pix2Pix - 5, 10, 20 epochs - all datasets including MRI
    print("Datasets with Pix2Pix augmentation")
    for dataset_name, data_dir in data_dirs.items():
        # MRI pix2pix is directly in mri folder
        if dataset_name == 'mri':
            pix2pix_dir = os.path.join(data_dir, 'pix2pix')
        elif os.path.exists(os.path.join(data_dir, 'Training')):
            pix2pix_dir = os.path.join(data_dir, 'Training', 'pix2pix')
        else:
            pix2pix_dir = os.path.join(data_dir, 'pix2pix')

        if os.path.exists(pix2pix_dir):
            print(f'Loading Pix2Pix images for {dataset_name}...')
            train_loader, val_loader, num_classes = load_dataset(data_dir, transform)
            pix2pix_dataset = datasets.ImageFolder(pix2pix_dir, transform=transform)
            augmented_loader = DataLoader(
                ConcatDataset([train_loader.dataset, pix2pix_dataset]),
                batch_size=32, shuffle=True
            )

            for epochs in [5, 10, 20]:
                print(f'Running models on {dataset_name} pix2pix - {epochs} epochs...')
                key = f'{dataset_name}_pix2pix_{epochs}'
                train_acc, val_acc, knn_acc, result = run_models(
                    key, augmented_loader, val_loader, num_classes, epochs=epochs
                )
                train_acc_history[key] = train_acc
                val_acc_history[key] = val_acc
                knn_acc_history[key] = knn_acc
                results.append(result)
        else:
            print(f'No pix2pix folder for: {dataset_name} – run pix2pix_saver.py first!')

    plot_all_comparisons(train_acc_history, val_acc_history, knn_acc_history)
    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    main()