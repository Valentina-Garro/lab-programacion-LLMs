import pandas as pd
import numpy as np

def filtrar_genes_criticos(df, umbral_varianza, max_nans):

    limite_nulos = len(df) * max_nans

    df_filtrado = df.loc[:, df.isnull().sum() <= limite_nulos]

    df_imputado = df_filtrado.fillna(df_filtrado.mean())

    return df_imputado.loc[:, df_imputado.var() > umbral_varianza]