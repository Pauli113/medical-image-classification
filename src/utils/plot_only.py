from plot_comparisons import plot_all_comparisons

train_acc_history = {
    'mri': [0.8836, 0.8836, 0.8832, 0.8813, 0.8790],
    'xray': [0.4374, 0.5854, 0.7351, 0.8362, 0.8716],
    'xray_covid_19': [0.6467, 0.8067, 0.8067, 0.8067, 0.8067],
    'mri_gan': [0.8817, 0.8807, 0.8815, 0.8790, 0.8792],
    'xray_gan': [0.3825, 0.4406, 0.4951, 0.5526, 0.5807],
    'xray_covid_19_gan': [0.7567, 0.7933, 0.7933, 0.7933, 0.7933]
}

val_acc_history = {
    'mri': [0.8622, 0.8622, 0.8576, 0.8560, 0.8515],
    'xray': [0.4627, 0.6329, 0.8193, 0.8757, 0.8977],
    'xray_covid_19': [0.7105, 0.7105, 0.7105, 0.7105, 0.7105],
    'mri_gan': [0.8698, 0.8698, 0.8606, 0.8652, 0.8652],
    'xray_gan': [0.5153, 0.6463, 0.6788, 0.7830, 0.8518],
    'xray_covid_19_gan': [0.7632, 0.7632, 0.7632, 0.7632, 0.7632]
}

knn_acc_history = {
    'mri': 0.8576,
    'xray': 0.9293,
    'xray_covid_19': 0.6842,
    'mri_gan': 0.8714,
    'xray_gan': 0.9226,
    'xray_covid_19_gan': 0.7895
}

plot_all_comparisons(train_acc_history, val_acc_history, knn_acc_history)
print("Plots erstellt!")