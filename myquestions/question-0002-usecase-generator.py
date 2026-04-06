import pandas as pd
import numpy as np

def generar_caso_de_uso_detectar_outliers():
    datos = np.random.normal(loc=100, scale=20, size=50)

    outliers = np.random.randint(200, 300, size=5)
    datos = np.concatenate([datos, outliers])

    df = pd.DataFrame({
        "valores": datos
    })

    return df, None