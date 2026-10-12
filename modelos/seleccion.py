"""
Criterios para comparar y elegir modelos candidatos.

Origen: la condición de buscar_mejor_modelo() del profesor
("preferimos menor RMSE"; en empate gana el primero) y la condición de
estabilidad aT < 0 que usa invernadero_20dias.py.
"""

from __future__ import annotations

import numpy as np


def es_estable(coef: dict) -> bool:
    """Un modelo lineal de primer orden es estable si aT < 0."""
    return bool(coef["aT"] < 0)


def es_mejor(metricas: dict, actual: dict | None) -> bool:
    """True si `metricas` mejora (RMSE estrictamente menor) a `actual`."""
    return actual is None or metricas["RMSE"] < actual["RMSE"]


def indice_mejor(lista_metricas: list) -> int:
    """Índice del menor RMSE; ante empate, el primero (como el original)."""
    mejor, idx = None, None
    for i, m in enumerate(lista_metricas):
        if m is not None and np.isfinite(m["RMSE"]) and es_mejor(m, mejor):
            mejor, idx = m, i
    if idx is None:
        raise RuntimeError("Ninguno de los modelos pudo ser identificado.")
    return idx
