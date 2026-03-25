import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNN:
    # init KNN with 3,5,7,9 as n
    def __init__(self, k_list=None):
        if k_list is None:
            k_list = [3, 5, 7, 9]
        param_grid = {"n_neighbors": list(k_list)}
        knn = KNeighborsClassifier()
        self.model = GridSearchCV(knn, param_grid, cv=3, n_jobs=1)
        self.scaler = StandardScaler()

    def best_k(self):
        return self.model.best_params_["n_neighbors"]
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def score(self, X, y):
        X = self.scaler.transform(X)
        return self.model.score(X, y)

    def run_knn(self, train_loader, val_loader):
        #extract images from dataloader and flatten
        #flatten to convert 2D images to 1D vectors
        x_train, y_train = [], []
        for x, y in train_loader:
            x_train.append(x.view(x.size(0), -1).numpy())
            y_train.append(y.numpy())
        x_val, y_val = [], []
        for x, y in val_loader:
            x_val.append(x.view(x.size(0), -1).numpy())
            y_val.append(y.numpy())

        #concat all batches from train arrays
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        #training
        self.fit(x_train, y_train)

        #accuracy
        return self.score(x_val, y_val)
