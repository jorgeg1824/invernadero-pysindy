"""
Trayectorias simuladas sobre varios tramos y predictor de referencia.

Origen: invernadero_20dias.py -> simular_archivo() y el "perfil diario
promedio" que evaluar() calculaba como referencia ingenua.
"""

from __future__ import annotations

import pandas as pd

from simulacion.integrador import simular_lsim


def simular_archivo(bloques: list, coef: dict, train: pd.DataFrame, dt: float,
                    zona_horaria: str = "America/Bogota") -> pd.DataFrame:
    """
    Simula la ecuación en cada tramo continuo. La temperatura real solo se
    usa como valor inicial de cada tramo. Marca qué filas pertenecen al
    entrenamiento y el mes (en hora local) de cada fila.
    """
    partes = []
    for b in bloques:
        p = b[["Temperatura"]].rename(columns={"Temperatura": "real"})
        p["simulada"] = simular_lsim(coef, b, dt)
        partes.append(p)
    res = pd.concat(partes)
    res["entrenamiento"] = (res.index >= train.index[0]) & (res.index <= train.index[-1])
    res["mes"] = res.index.tz_convert(zona_horaria).strftime("%Y-%m")
    return res


def perfil_diario(train: pd.DataFrame, val: pd.DataFrame):
    """
    Referencia ingenua: para cada hora-minuto del día, la temperatura media
    de ese instante en el entrenamiento. Usa la hora UTC, como el original.
    """
    paso_del_dia = lambda idx: idx.hour * 60 + idx.minute
    perfil = train["Temperatura"].groupby(paso_del_dia(train.index)).mean()
    return perfil.reindex(paso_del_dia(val.index)).values
