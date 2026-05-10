from plot import plot_results


def plot_all_comparisons(train_acc_history, val_acc_history, knn_acc_history):
    # ohne GAN
    plot_results(
        {k: train_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        {k: val_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        {k: knn_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        'results_without_gan.png'
    )

    # mit GAN
    plot_results(
        {k: train_acc_history[k] for k in ['mri_gan', 'xray_gan', 'xray_covid_19_gan']},
        {k: val_acc_history[k] for k in ['mri_gan', 'xray_gan', 'xray_covid_19_gan']},
        {k: knn_acc_history[k] for k in ['mri_gan', 'xray_gan', 'xray_covid_19_gan']},
        'results_with_gan.png'
    )

    # MRI Vergleich
    plot_results(
        {k: train_acc_history[k] for k in ['mri', 'mri_gan']},
        {k: val_acc_history[k] for k in ['mri', 'mri_gan']},
        {k: knn_acc_history[k] for k in ['mri', 'mri_gan']},
        'results_mri_comparison.png'
    )

    # Xray Vergleich
    plot_results(
        {k: train_acc_history[k] for k in ['xray', 'xray_gan']},
        {k: val_acc_history[k] for k in ['xray', 'xray_gan']},
        {k: knn_acc_history[k] for k in ['xray', 'xray_gan']},
        'results_xray_comparison.png'
    )

    # Xray Covid Vergleich
    plot_results(
        {k: train_acc_history[k] for k in ['xray_covid_19', 'xray_covid_19_gan']},
        {k: val_acc_history[k] for k in ['xray_covid_19', 'xray_covid_19_gan']},
        {k: knn_acc_history[k] for k in ['xray_covid_19', 'xray_covid_19_gan']},
        'results_xray_covid_19_comparison.png'
    )

    # Xray_gan vs Xray_covid_19_gan
    plot_results(
        {k: train_acc_history[k] for k in ['xray_gan', 'xray_covid_19_gan']},
        {k: val_acc_history[k] for k in ['xray_gan', 'xray_covid_19_gan']},
        {k: knn_acc_history[k] for k in ['xray_gan', 'xray_covid_19_gan']},
        'results_xray_gan_comparison.png'
    )