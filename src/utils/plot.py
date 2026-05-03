from matplotlib import pyplot as plt


def plot_results(train_acc_history, val_acc_history, knn_acc_history, filename='results.png'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # UNet on top
    for dataset_name in train_acc_history.keys():
        train_vals = [v * 100 for v in train_acc_history[dataset_name]]
        val_vals = [v * 100 for v in val_acc_history[dataset_name]]
        ax1.plot(train_vals, label=f'{dataset_name} Train')
        ax1.plot(val_vals, label=f'{dataset_name} Val')
    ax1.set_title('U-Net Training and Validation Accuracies')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()

    # KNN below
    knn_vals = {k: v * 100 for k, v in knn_acc_history.items()}
    for dataset_name, val in knn_vals.items():
        ax2.bar(dataset_name, val, label=f'{dataset_name} KNN Val')
    ax2.set_title('KNN Validation Accuracies')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()