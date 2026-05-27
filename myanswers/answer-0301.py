import pandas as pd
import numpy as np

def detectar_rafagas_sensores(df, sensor_col, tiempo_col, margen_segundos):

    df_res = df.copy()

    df_res["_orden_original"] = np.arange(len(df_res))

    df_res[tiempo_col] = pd.to_datetime(df_res[tiempo_col])

    df_res = df_res.sort_values([sensor_col, tiempo_col])

    diff = (
        df_res
        .groupby(sensor_col)[tiempo_col]
        .diff()
        .dt.total_seconds()
    )

    df_res["es_rafaga"] = (diff < margen_segundos).fillna(False)

    df_res = (
        df_res
        .sort_values("_orden_original")
        .drop(columns="_orden_original")
    )

    return df_res