"""
Limpieza de un tramo de datos.

Hay DOS procedimientos distintos y se conservan separados:

  limpiar_tramo()      invernadero_profesor.py
      Quita filas con NaN, ordena, elimina marcas duplicadas y descarta las
      muestras que llegan tras un salto dt > factor x dt mediano.
      NO crea una malla nueva ni interpola: las muestras quedan con su
      marca de tiempo irregular original.

  rellenar_en_malla()  invernadero_20dias.py (parte interna de bloques_rellenados)
      Lleva un tramo a una malla UNIFORME (p. ej. cada 5 min) e interpola
      linealmente en el tiempo los valores de la malla.

Ambas "limpian", pero producen datos de naturaleza diferente (muestreo
irregular vs uniforme), y el resto de cada flujo depende de eso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import VARIABLES


def limpiar_tramo(datos: pd.DataFrame, factor: float = 5.0) -> pd.DataFrame:
    """Copia fiel del original (factor fijo 5 en el original)."""
    datos = datos.copy()
    datos = datos.dropna(subset=VARIABLES)
    datos = datos.sort_values("datetime_utc")
    datos = datos.drop_duplicates(subset=["datetime_utc"], keep="first").reset_index(drop=True)

    t = datos["datetime_utc"].astype("int64") / 1e9
    dt = np.diff(t)
    if len(dt) == 0:
        return datos

    dt_mediana = np.median(dt)
    mascara = np.ones(len(datos), dtype=bool)
    mascara[1:] = dt <= factor * dt_mediana
    return datos.loc[mascara].copy()


def resumen_limpieza_tramo(antes: pd.DataFrame, despues: pd.DataFrame) -> dict:
    con_nan = int(antes[VARIABLES].isna().any(axis=1).sum())
    return {
        "filas_antes": int(len(antes)),
        "filas_con_nan": con_nan,
        "filas_despues": int(len(despues)),
        "filas_descartadas": int(len(antes) - len(despues)),
        "descartadas_por_salto_temporal": int(len(antes) - con_nan - len(despues)),
        "interpoladas": 0,
    }


def rellenar_en_malla(g: pd.DataFrame, paso: pd.Timedelta):
    """
    g: DataFrame indexado por fecha con las columnas VARIABLES.
    Devuelve el tramo en una malla uniforme [ceil(inicio), floor(fin)] con
    paso `paso`, interpolando en el tiempo, o None si la malla tiene < 2 puntos.
    (Mismo cálculo que el original.)
    """
    malla = pd.date_range(g.index[0].ceil(paso), g.index[-1].floor(paso), freq=paso)
    if len(malla) < 2:
        return None
    return g.reindex(g.index.union(malla)).interpolate(method="time").reindex(malla)


def puntos_en_huecos(datos_originales: pd.DataFrame, bloque: pd.DataFrame, paso: pd.Timedelta) -> np.ndarray:
    """
    Máscara de los puntos de la malla que NO tienen ninguna medición real a
    menos de `paso`: son los que se "inventan" al rellenar un hueco.
    (Los demás también salen de interpolar, pero entre mediciones vecinas.)
    """
    orig = pd.DatetimeIndex(datos_originales["datetime_utc"]).sort_values()
    pos = orig.searchsorted(bloque.index)
    antes = orig[np.clip(pos - 1, 0, len(orig) - 1)]
    despues = orig[np.clip(pos, 0, len(orig) - 1)]
    distancia = np.minimum(np.abs((bloque.index - antes).total_seconds()),
                           np.abs((despues - bloque.index).total_seconds()))
    return distancia > paso.total_seconds()


def resumen_relleno(datos_originales: pd.DataFrame, bloques: list, paso: pd.Timedelta) -> dict:
    """Mediciones usadas/descartadas y puntos de malla interpolados o rellenados en huecos."""
    originales = pd.DatetimeIndex(datos_originales["datetime_utc"])
    puntos = sum(len(b) for b in bloques)
    en_huecos = sum(int(puntos_en_huecos(datos_originales, b, paso).sum()) for b in bloques)
    usadas = sum(int(((originales >= b.index[0] - paso) & (originales <= b.index[-1] + paso)).sum())
                 for b in bloques)
    return {
        "mediciones_originales": int(len(originales)),
        "mediciones_dentro_de_tramos": int(usadas),
        "mediciones_fuera_de_tramos (descartadas)": int(len(originales) - usadas),
        "puntos_malla": int(puntos),
        "puntos_interpolados_entre_mediciones_vecinas": int(puntos - en_huecos),
        "puntos_rellenados_en_huecos": int(en_huecos),
        "paso_min": paso.total_seconds() / 60,
    }
