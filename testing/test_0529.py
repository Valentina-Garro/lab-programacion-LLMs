import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import numpy as np

from myanswers.answer_0529 import filtrar_genes_criticos


def generar_caso_de_uso_filtrar_genes_criticos():

    n_genes, n_muestras = np.random.randint(10, 20), np.random.randint(50, 100)

    data = np.random.rand(n_muestras, n_genes)

    mask = np.random.choice(
        [True, False],
        size=data.shape,
        p=[0.1, 0.9]
    )

    data[mask] = np.nan

    df = pd.DataFrame(
        data,
        columns=[f"Gene_{i}" for i in range(n_genes)]
    )

    u_var, m_nans = 0.05, 0.15

    entrada = {
        "df": df,
        "umbral_varianza": u_var,
        "max_nans": m_nans
    }

    limite_nulos = len(df) * m_nans

    df_filtrado = df.loc[:, df.isnull().sum() <= limite_nulos]

    df_imputado = df_filtrado.fillna(df_filtrado.mean())

    esperado = df_imputado.loc[
        :,
        df_imputado.var() > u_var
    ]

    return entrada, esperado


correctos = 0

for i in range(100):

    entrada, esperado = generar_caso_de_uso_filtrar_genes_criticos()

    resultado = filtrar_genes_criticos(**entrada)

    if resultado.equals(esperado):
        correctos += 1

print(correctos)