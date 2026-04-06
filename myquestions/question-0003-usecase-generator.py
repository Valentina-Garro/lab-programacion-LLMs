import numpy as np
from sklearn.datasets import make_classification  # type: ignore

def generar_caso_de_uso_clasificar_knn():
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2
    )

    nuevo_punto = np.random.rand(1, 2)

    return (X, y), nuevo_punto