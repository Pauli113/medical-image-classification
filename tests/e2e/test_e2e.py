import torch
import os
from torch.utils.data import TensorDataset, DataLoader

from src.models.knn.application.knn_trainer import KNNTrainer
from src.models.unet.application.unet_trainer import UNetTrainer
from src.utils.plot import plot_results
from src.utils.runner import run_models, run_gan, augment_dataset


def test_e2e():
    #ohne gan
    # Fake Bilder und Labels für DataLoader
    X = torch.randn(10, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)
    print("Train models")
    print('KNN')
    # KNN mit DataLoader
    knn = KNNTrainer()
    knn_accuracy, knn_time = knn.run_knn(train_loader, val_loader)
    assert knn_accuracy >= 0

    print('U-Net')
    # UNet
    model = UNetTrainer(num_classes=3)
    x_img = torch.rand(1, 3, 224, 224)
    output = model.model(x_img)
    prediction = torch.argmax(output, dim=1)
    assert prediction.item() in [0, 1, 2]

    train_acc, val_acc, unet_time = model.run_unet(train_loader, val_loader, epochs=2)
    assert len(train_acc) == 2
    assert len(val_acc) == 2

    print("Run models")
    # run_models testen
    train_accuracies, val_accuracies, knn_accuracy, result = run_models(
        'fake', train_loader, val_loader, num_classes=3, epochs=2
    )
    assert 'Dataset' in result
    assert 'U-Net Accuracy' in result
    assert 'KNN Accuracy' in result

    # Plot Creation
    train_acc_history = {'fake': train_acc}
    val_acc_history = {'fake': val_acc}
    knn_acc_history = {'fake': knn_accuracy}

    plot_results(train_acc_history, val_acc_history, knn_acc_history)
    assert os.path.exists('results.png')
    os.remove('results.png')

    print("Test with generated images")
    #mit gan
    gan_trainer_result, gan_time = run_gan(train_loader, epochs=1)
    assert isinstance(gan_time, str)

    #augment dataset
    augment_loader = augment_dataset(gan_trainer_result, train_loader)
    assert len(augment_loader.dataset) == 2 * len(dataset)

    print("Run models with generated images")
    #test mit generierten Bilder
    train_accuracies_gan, val_accuracies_gan, knn_accuracy_gan, result_gan = run_models(
        'fake_gan', augment_loader, val_loader, num_classes=3, epochs=2
    )

    assert 'Dataset' in result_gan