from plot import plot_results, plot_overall

def plot_all_comparisons(train_acc_history, val_acc_history, knn_acc_history):
    # without augmentation 5 epochs
    plot_results(
        {k: train_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        {k: val_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        {k: knn_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        'results_without_pix2pix_5.png'
    )

    # without augmentation 10 epochs
    plot_results(
        {k: train_acc_history[k] for k in ['mri_10', 'xray_10', 'xray_covid_19_10']},
        {k: val_acc_history[k] for k in ['mri_10', 'xray_10', 'xray_covid_19_10']},
        {k: knn_acc_history[k] for k in ['mri_10', 'xray_10', 'xray_covid_19_10']},
        'results_without_pix2pix_10.png'
    )

    # without augmentation 20 epochs
    plot_results(
        {k: train_acc_history[k] for k in ['mri_20', 'xray_20', 'xray_covid_19_20']},
        {k: val_acc_history[k] for k in ['mri_20', 'xray_20', 'xray_covid_19_20']},
        {k: knn_acc_history[k] for k in ['mri_20', 'xray_20', 'xray_covid_19_20']},
        'results_without_pix2pix_20.png'
    )

    # with Pix2Pix 5 epochs
    plot_results(
        {k: train_acc_history[k] for k in ['mri_pix2pix_5', 'xray_pix2pix_5', 'xray_covid_19_pix2pix_5']},
        {k: val_acc_history[k] for k in ['mri_pix2pix_5', 'xray_pix2pix_5', 'xray_covid_19_pix2pix_5']},
        {k: knn_acc_history[k] for k in ['mri_pix2pix_5', 'xray_pix2pix_5', 'xray_covid_19_pix2pix_5']},
        'results_with_pix2pix_5.png'
    )

    # with Pix2Pix 10 epochs
    plot_results(
        {k: train_acc_history[k] for k in ['mri_pix2pix_10', 'xray_pix2pix_10', 'xray_covid_19_pix2pix_10']},
        {k: val_acc_history[k] for k in ['mri_pix2pix_10', 'xray_pix2pix_10', 'xray_covid_19_pix2pix_10']},
        {k: knn_acc_history[k] for k in ['mri_pix2pix_10', 'xray_pix2pix_10', 'xray_covid_19_pix2pix_10']},
        'results_with_pix2pix_10.png'
    )

    # with Pix2Pix 20 epochs
    plot_results(
        {k: train_acc_history[k] for k in ['mri_pix2pix_20', 'xray_pix2pix_20', 'xray_covid_19_pix2pix_20']},
        {k: val_acc_history[k] for k in ['mri_pix2pix_20', 'xray_pix2pix_20', 'xray_covid_19_pix2pix_20']},
        {k: knn_acc_history[k] for k in ['mri_pix2pix_20', 'xray_pix2pix_20', 'xray_covid_19_pix2pix_20']},
        'results_with_pix2pix_20.png'
    )

    # MRI comparison
    plot_results(
        {k: train_acc_history[k] for k in ['mri', 'mri_10', 'mri_20', 'mri_pix2pix_5', 'mri_pix2pix_10', 'mri_pix2pix_20']},
        {k: val_acc_history[k] for k in ['mri', 'mri_10', 'mri_20', 'mri_pix2pix_5', 'mri_pix2pix_10', 'mri_pix2pix_20']},
        {k: knn_acc_history[k] for k in ['mri', 'mri_pix2pix_5']},
        'results_mri_comparison.png'
    )

    # Xray comparison
    plot_results(
        {k: train_acc_history[k] for k in ['xray', 'xray_10', 'xray_20', 'xray_pix2pix_5', 'xray_pix2pix_10', 'xray_pix2pix_20']},
        {k: val_acc_history[k] for k in ['xray', 'xray_10', 'xray_20', 'xray_pix2pix_5', 'xray_pix2pix_10', 'xray_pix2pix_20']},
        {k: knn_acc_history[k] for k in ['xray', 'xray_pix2pix_5']},
        'results_xray_comparison.png'
    )

    # Xray Covid comparison
    plot_results(
        {k: train_acc_history[k] for k in ['xray_covid_19', 'xray_covid_19_10', 'xray_covid_19_20', 'xray_covid_19_pix2pix_5', 'xray_covid_19_pix2pix_10', 'xray_covid_19_pix2pix_20']},
        {k: val_acc_history[k] for k in ['xray_covid_19', 'xray_covid_19_10', 'xray_covid_19_20', 'xray_covid_19_pix2pix_5', 'xray_covid_19_pix2pix_10', 'xray_covid_19_pix2pix_20']},
        {k: knn_acc_history[k] for k in ['xray_covid_19', 'xray_covid_19_pix2pix_5']},
        'results_xray_covid_19_comparison.png'
    )

    # overall without pix2pix
    plot_overall(
        {k: knn_acc_history[k] for k in ['mri', 'xray', 'xray_covid_19']},
        val_acc_history,
        'results_overall_without_pix2pix.png'
    )

    # overall with pix2pix
    plot_overall(
        {k: knn_acc_history[f'{k}_pix2pix_5'] for k in ['mri', 'xray', 'xray_covid_19']},
        {f'{k}_pix2pix_{e}': val_acc_history[f'{k}_pix2pix_{e}'] for k in ['mri', 'xray', 'xray_covid_19'] for e in [5, 10, 20]},
        'results_overall_with_pix2pix.png'
    )