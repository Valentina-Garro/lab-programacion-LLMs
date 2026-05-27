import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from myanswers.answer_0623 import agrupar_jugadores


def generar_caso_de_uso_agrupar_jugadores():

    n_rows = random.randint(6, 10)

    n_clusters = random.randint(2, 4)

    df = pd.DataFrame({
        "goles": np.random.randint(0, 20, size=n_rows),
        "asistencias": np.random.randint(0, 15, size=n_rows),
        "pases_clave": np.random.randint(5, 50, size=n_rows),
        "recuperaciones": np.random.randint(1, 40, size=n_rows)
    })

    entrada = {
        "df": df.copy(),
        "n_clusters": n_clusters
    }

    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    esperado = modelo.fit_predict(X_scaled)

    return entrada, esperado


correctos = 0

for i in range(100):

    entrada, esperado = generar_caso_de_uso_agrupar_jugadores()

    resultado = agrupar_jugadores(**entrada)

    if np.array_equal(resultado, esperado):
        correctos += 1

print(correctos)