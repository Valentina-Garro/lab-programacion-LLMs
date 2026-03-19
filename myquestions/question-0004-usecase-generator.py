import numpy as np

def generar_caso_de_uso_reducir_dimensionalidad_pca():
    n_muestras = np.random.randint(50, 150)
    n_features = np.random.randint(5, 10)

    X = np.random.rand(n_muestras, n_features)

    return X