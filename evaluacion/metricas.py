"""
Métricas de error entre temperatura medida y simulada.

Origen: calcular_metricas() del profesor (compartida: invernadero_20dias.py
la importaba), mae_por_dia() y metricas_archivo() de invernadero_20dias.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calcular_metricas(real, pred) -> dict:
    real = np.asarray(real)
    pred = np.asarray(pred)
    error = real - pred
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def mae_por_dia(real, pred, muestras_por_dia: int) -> list:
    """MAE de cada día de predicción (malla uniforme)."""
    real, pred = np.asarray(real), np.asarray(pred)
    dia = np.arange(len(real)) // muestras_por_dia
    return [float(np.mean(np.abs(real - pred)[dia == d])) for d in np.unique(dia)]


def metricas_archivo(res: pd.DataFrame):
    """Métricas sin contar el tramo de entrenamiento: (total, {mes: métricas})."""
    fuera = res[~res["entrenamiento"]]
    m = lambda d: calcular_metricas(d["real"].values, d["simulada"].values)
    return m(fuera), {mes: m(d) for mes, d in fuera.groupby("mes")}
