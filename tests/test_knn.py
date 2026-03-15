import numpy as np
#checks image to vector code
def test_knn_flatten():
    picture = np.random.rand(3,224,224)
    target = 3*224*224
    assert(target==picture.flatten().shape[0])

def test_knn_training():
    ...

def test_knn_prediction():
    ...