import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import numpy as np

from myanswers.answer_0301 import detectar_rafagas_sensores


def generar_caso_de_uso_detectar_rafagas_sensores():

    n_sensores = np.random.randint(2, 5)

    sensores = [f"Sensor_{i}" for i in range(n_sensores)]

    data = []

    for s in sensores:

        tiempo_base = pd.Timestamp("2026-03-30 08:00:00")

        n_registros = np.random.randint(4, 8)

        for _ in range(n_registros):

            segundos = int(
                np.random.choice([1, 2, 3, 6, 10, 15])
            )

            tiempo_base += pd.Timedelta(seconds=segundos)

            data.append({
                "sensor_id": s,
                "ts": tiempo_base
            })

    df = (
        pd.DataFrame(data)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    margen = 5

    df_res = df.copy()

    df_res["_orden_original"] = np.arange(len(df_res))

    df_res["ts"] = pd.to_datetime(df_res["ts"])

    df_res = df_res.sort_values(["sensor_id", "ts"])

    diff = (
        df_res
        .groupby("sensor_id")["ts"]
        .diff()
        .dt.total_seconds()
    )

    df_res["es_rafaga"] = (diff < margen).fillna(False)

    esperado = (
        df_res
        .sort_values("_orden_original")
        .drop(columns="_orden_original")
    )

    entrada = {
        "df": df,
        "sensor_col": "sensor_id",
        "tiempo_col": "ts",
        "margen_segundos": margen
    }

    return entrada, esperado


correctos = 0

for i in range(100):

    entrada, esperado = generar_caso_de_uso_detectar_rafagas_sensores()

    resultado = detectar_rafagas_sensores(**entrada)

    if resultado.equals(esperado):
        correctos += 1

print(correctos)