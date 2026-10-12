"""
Eje de tiempo y derivada de la temperatura (dT/dt).

Dos procedimientos DIFERENTES para la derivada:

  calcular_derivada_temperatura()   invernadero_profesor.py
      1) suaviza T con Savitzky-Golay (ventana 21, orden 2) tratando las
         muestras como si fueran equiespaciadas, y 2) deriva con
         np.gradient usando las marcas de tiempo REALES (irregulares).

  derivada_savgol_directa()         invernadero_20dias.py
      Savitzky-Golay con deriv=1 sobre una malla UNIFORME, con
      delta = paso en horas. Suaviza y deriva en un solo paso.

Corrección C1 (tiempo en segundos)
----------------------------------
El original calculaba el tiempo con  fechas.astype("int64") / 1e9,
suponiendo que pandas guarda las fechas en nanosegundos. Desde pandas 3.0
las fechas leídas de texto se guardan en MICROsegundos, y esa expresión da
kilosegundos: la ecuación sale en °C/ks y el barrido de threshold elige
otro modelo (ver docs/informe_final.md). tiempo_absoluto(unidad="s") usa
total_seconds(), que es correcto en cualquier versión. unidad="original"
ejecuta la expresión original tal cual, para poder reproducir resultados
antiguos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_EPOCH = pd.Timestamp(0, tz="UTC")
UNIDADES_TIEMPO = {"s": "s", "original": "astype('int64')/1e9"}


def tiempo_absoluto(fechas, unidad: str = "s") -> np.ndarray:
    """Segundos desde 1970 (o la expresión original si unidad='original')."""
    fechas = pd.Series(fechas)
    if unidad == "original":
        return fechas.astype("int64").values / 1e9
    if unidad == "s":
        return (fechas - _EPOCH).dt.total_seconds().values
    raise ValueError(f"unidad_tiempo desconocida: {unidad!r} (use 's' u 'original')")


def tiempo_relativo(fechas, unidad: str = "s") -> np.ndarray:
    t = tiempo_absoluto(fechas, unidad)
    return t - t[0]


def unidad_efectiva(unidad: str) -> str:
    """Qué unidad produce realmente `unidad` con la versión de pandas instalada."""
    if unidad == "s":
        return "s"
    prueba = pd.to_datetime(pd.Series(["1970-01-01T00:00:01Z"]), utc=True)
    valor = tiempo_absoluto(prueba, "original")[0]
    return {1.0: "s", 1e-3: "ks", 1e-6: "Ms"}.get(round(valor, 9), f"x{valor:g} s")


# ------------------------------------------------------------
# Derivada del programa del profesor
# ------------------------------------------------------------

def calcular_derivada_temperatura(datos: pd.DataFrame, ventana_max: int = 21,
                                  polyorder: int = 2, unidad: str = "s"):
    """Devuelve (T_suavizada, dT/dt). dT/dt en °C por unidad de tiempo."""
    T = datos["Temperatura"].values.astype(float)
    t = tiempo_absoluto(datos["datetime_utc"], unidad)

    n = len(T)
    if n < 7:
        raise ValueError("Muy pocos datos para calcular la derivada.")

    ventana = min(ventana_max, n)
    if ventana % 2 == 0:
        ventana -= 1
    if ventana < 5:
        ventana = 5

    T_suave = savgol_filter(T, window_length=ventana, polyorder=polyorder)

    orden = np.argsort(t, kind="stable")
    t = t[orden]
    T_suave = T_suave[orden]
    if np.any(np.diff(t) <= 0):
        raise ValueError("El tiempo no es estrictamente creciente después de limpiar duplicados.")

    return T_suave, np.gradient(T_suave, t)


# ------------------------------------------------------------
# Derivada de invernadero_20dias.py
# ------------------------------------------------------------

def derivada_savgol_directa(T, ventana: int = 21, polyorder: int = 2, delta: float = 5 / 60):
    """dT/dt sobre malla uniforme; delta = paso de la malla (en horas en el original)."""
    return savgol_filter(np.asarray(T, dtype=float), ventana, polyorder, deriv=1, delta=delta)
