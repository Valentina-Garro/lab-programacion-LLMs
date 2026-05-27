import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def agrupar_jugadores(df, n_clusters):

    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    return modelo.fit_predict(X_scaled)