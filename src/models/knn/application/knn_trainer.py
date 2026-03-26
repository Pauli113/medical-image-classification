import numpy as np
from src.models.knn.domain.knn import KNN
from src.utils.timer import Timer


class KNNTrainer:
    def __init__(self):
        self.model = KNN()

    def run_knn(self, train_loader, val_loader):
        timer = Timer()
        timer.start()

        x_train, y_train = [], []
        for x, y in train_loader:
            x_train.append(x.view(x.size(0), -1).numpy())
            y_train.append(y.numpy())

        x_val, y_val = [], []
        for x, y in val_loader:
            x_val.append(x.view(x.size(0), -1).numpy())
            y_val.append(y.numpy())

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        self.model.fit(x_train, y_train)
        knn_accuracy = self.model.score(x_val, y_val)
        knn_time = timer.stop()

        print(f'Best K: {self.model.best_k()}')
        print(f'KNN Validation Accuracy: {knn_accuracy*100:.2f}%')
        print(f'KNN Training Time: {knn_time}')

        return knn_accuracy, knn_time