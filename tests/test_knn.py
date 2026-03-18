import numpy as np

from src.models.knn import KNN


#checks image to vector code
def test_knn_flatten():
    picture = np.random.rand(3,224,224)
    target = 3*224*224
    assert(target==picture.flatten().shape[0])

def test_knn_training():
    #fake picture generation using vectors
    X_train = np.random.rand(30, 150528)
    y_train = np.array([0, 1, 2] * 10)  # 10x wiederholt
    knn = KNN()
    knn.fit(X_train, y_train)
    assert(knn.best_k() in [3, 5, 7, 9])

def test_knn_prediction():
    ...