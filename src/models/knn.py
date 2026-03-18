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
