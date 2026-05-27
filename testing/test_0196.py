import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from myanswers.answer_0196 import diagnosticar_sesgo_varianza


def generar_caso():

    np.random.seed(np.random.randint(0, 9999))

    modo = np.random.choice(["regresion", "clasificacion"])

    n = np.random.randint(400, 900)

    nf = np.random.randint(5, 20)

    X = np.random.randn(n, nf)

    if modo == "regresion":

        coef = np.random.randn(nf)

        y = X @ coef + np.random.randn(n) * np.random.uniform(1, 5)

        d = np.random.choice([2, None])

        modelo = DecisionTreeRegressor(
            max_depth=d,
            random_state=42
        )

    else:

        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        ne = np.random.choice([5, 200])

        modelo = RandomForestClassifier(
            n_estimators=ne,
            random_state=42
        )

    cv = np.random.choice([3, 5])

    n_puntos = np.random.randint(5, 10)

    return X, y, modelo, cv, n_puntos


for i in range(20):

    X, y, modelo, cv, n_puntos = generar_caso()

    resultado = diagnosticar_sesgo_varianza(
        X,
        y,
        modelo,
        cv,
        n_puntos
    )

    print(resultado)