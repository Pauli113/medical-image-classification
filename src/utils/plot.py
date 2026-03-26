from matplotlib import pyplot as plt


def plot_results(train_acc_history, val_acc_history, knn_acc_history):
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
    ax2.bar(knn_vals.keys(), knn_vals.values())
    ax2.set_title('KNN Validation Accuracy')
    ax2.set_ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('results.png')