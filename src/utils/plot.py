import numpy as np
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


def plot_overall(knn_acc_history, val_acc_history, filename='results_overall.png'):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    datasets = list(knn_acc_history.keys())
    x = np.arange(4)  # KNN, 5, 10, 20 Epochen
    width = 0.25
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    for i, dataset in enumerate(datasets):
        # KNN bar
        knn_val = knn_acc_history[dataset] * 100
        ax1.bar(x[0] + (i - 1) * width, knn_val, width, color=colors[i], label=dataset)

        # U-Net bars per epoch
        for epoch_idx, epochs in enumerate([5, 10, 20]):
            key = dataset if epochs == 5 else f'{dataset}_{epochs}'
            if key in val_acc_history:
                val = val_acc_history[key][-1] * 100
                ax1.bar(x[epoch_idx + 1] + (i - 1) * width, val, width, color=colors[i])

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['KNN', 'U-Net (5 Ep.)', 'U-Net (10 Ep.)', 'U-Net (20 Ep.)'])
    ax1.set_title('Overall Accuracy Comparison')
    ax1.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()