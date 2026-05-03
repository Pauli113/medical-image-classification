from src.utils.plot import plot_results


def plot_all_comparisons(train_acc_history, val_acc_history, knn_acc_history):
    #no gan
    plot_results(
        {k: train_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        {k: val_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        {k: knn_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        'results_without_gan.png'
    )

    #with gan
    plot_results(
        {k: train_acc_history[k] for k in ['mri_gan', 'xray_gan', 'xray_covid_19_gan']},
        {k: val_acc_history[k] for k in ['mri_gan', 'xray_gan', 'xray_covid_19_gan']},
        {k: knn_acc_history[k] for k in ['mri_gan', 'xray_gan', 'xray_covid_19_gan']},
        'results_with_gan.png'
    )