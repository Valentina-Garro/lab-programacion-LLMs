import pandas as pd
import numpy as np

def generar_caso_de_uso_calcular_crecimiento_porcentual():
    n = np.random.randint(5, 15)
    fechas = pd.date_range(start="2023-01-01", periods=n)
    ventas = np.random.randint(50, 500, size=n)

    df = pd.DataFrame({
        "fecha": fechas,
        "ventas": ventas
    })

    return {"df": df}, None