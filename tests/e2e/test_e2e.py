import torch
import os
from torch.utils.data import TensorDataset, DataLoader

from src.models.knn.application.knn_trainer import KNNTrainer
from src.models.unet.application.unet_trainer import UNetTrainer
from src.utils.plot import plot_results
from src.utils.plot_comparisons import plot_all_comparisons
from src.utils.runner import run_models, run_gan, augment_dataset


def test_e2e():
    # Fake Bilder und Labels für DataLoader
    X = torch.randn(10, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    print("Train models")
    print('KNN')
    knn = KNNTrainer()
    knn_accuracy, knn_time = knn.run_knn(train_loader, val_loader)
    assert knn_accuracy >= 0

    print('U-Net')
    model = UNetTrainer(num_classes=3)
    x_img = torch.rand(1, 3, 224, 224)
    output = model.model(x_img)
    prediction = torch.argmax(output, dim=1)
    assert prediction.item() in [0, 1, 2]

    train_acc, val_acc, unet_time = model.run_unet(train_loader, val_loader, epochs=2)
    assert len(train_acc) == 2
    assert len(val_acc) == 2

    print("Run models")
    train_accuracies, val_accuracies, knn_accuracy, result = run_models(
        'fake', train_loader, val_loader, num_classes=3, epochs=2
    )
    assert 'Dataset' in result
    assert 'U-Net Accuracy' in result
    assert 'KNN Accuracy' in result

    print("Test with generated images")
    gan_trainer_result, gan_time = run_gan(train_loader, epochs=1)
    assert isinstance(gan_time, str)

    augment_loader = augment_dataset(gan_trainer_result, train_loader)
    assert len(augment_loader.dataset) == 2 * len(dataset)

    print("Run models with generated images")
    train_accuracies_gan, val_accuracies_gan, knn_accuracy_gan, result_gan = run_models(
        'fake_gan', augment_loader, val_loader, num_classes=3, epochs=2
    )
    assert 'Dataset' in result_gan

    # Plot Comparisons testen
    train_acc_history = {'mri': train_acc, 'xray': val_acc, 'xray_covid_19': train_acc,
                         'mri_gan': train_accuracies_gan, 'xray_gan': val_accuracies_gan,
                         'xray_covid_19_gan': train_accuracies_gan}
    val_acc_history = {'mri': val_acc, 'xray': val_acc, 'xray_covid_19': val_acc,
                       'mri_gan': val_accuracies_gan, 'xray_gan': val_accuracies_gan,
                       'xray_covid_19_gan': val_accuracies_gan}
    knn_acc_history = {'mri': knn_accuracy, 'xray': knn_accuracy, 'xray_covid_19': knn_accuracy,
                       'mri_gan': knn_accuracy_gan, 'xray_gan': knn_accuracy_gan,
                       'xray_covid_19_gan': knn_accuracy_gan}

    plot_all_comparisons(train_acc_history, val_acc_history, knn_acc_history)
    assert os.path.exists('results_without_gan.png')
    assert os.path.exists('results_with_gan.png')
    assert os.path.exists('results_mri_comparison.png')
    assert os.path.exists('results_xray_comparison.png')
    assert os.path.exists('results_xray_covid_19_comparison.png')
    assert os.path.exists('results_xray_gan_comparison.png')
    os.remove('results_without_gan.png')
    os.remove('results_with_gan.png')
    os.remove('results_mri_comparison.png')
    os.remove('results_xray_comparison.png')
    os.remove('results_xray_covid_19_comparison.png')
    os.remove('results_xray_gan_comparison.png')