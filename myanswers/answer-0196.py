import numpy as np
from sklearn.model_selection import learning_curve

def diagnosticar_sesgo_varianza(
    X,
    y,
    modelo,
    cv,
    n_puntos
):

    train_sizes = np.linspace(0.1, 1.0, n_puntos)

    sizes, train_scores, val_scores = learning_curve(
        modelo,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes
    )

    train_mean = train_scores.mean(axis=1)

    val_mean = val_scores.mean(axis=1)

    train_final = train_mean[-1]

    val_final = val_mean[-1]

    if train_final < 0.75:
        diagnostico = "alto sesgo"

    elif train_final >= 0.75 and (train_final - val_final) > 0.15:
        diagnostico = "alta varianza"

    else:
        diagnostico = "buen ajuste"

    return {
        "train_scores": train_mean.tolist(),
        "val_scores": val_mean.tolist(),
        "train_sizes": sizes.tolist(),
        "diagnostico": diagnostico
    }