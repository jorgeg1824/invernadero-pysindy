"""
División de los datos en tramos continuos (bloques) separados por huecos.

Dos criterios DIFERENTES, conservados por separado:

  obtener_bloques_continuos()   invernadero_profesor.py
      Hueco = dt > factor x dt mediano (con factor 4 y dt ~300 s: > 20 min).
      Los bloques conservan las muestras originales; descarta bloques con
      menos de 20 muestras.

  bloques_rellenados()          invernadero_20dias.py
      Hueco = dt > max_hueco_horas (2 h). Los huecos más cortos se rellenan
      con rellenar_en_malla() (malla uniforme de 5 min).

Con el CSV de ejemplo el primero produce bloques de como mucho ~24 días y
el segundo de hasta ~41 días; por eso el flujo de 20 días usa el segundo.
"""

from __future__ import annotations

import pandas as pd

from config import VARIABLES
from datos.limpieza import rellenar_en_malla


def obtener_bloques_continuos(datos: pd.DataFrame, factor: float = 4.0) -> list:
    """Copia fiel del original. Devuelve una lista de diccionarios."""
    datos = datos.copy()
    datos["dt"] = datos["datetime_utc"].diff().dt.total_seconds()

    dt_validos = datos["dt"].dropna()
    if len(dt_validos) == 0:
        return []

    dt_mediana = dt_validos.median()
    datos["corte"] = datos["dt"] > factor * dt_mediana
    datos["bloque"] = datos["corte"].cumsum()

    bloques = []
    for _, grupo in datos.groupby("bloque"):
        grupo = grupo.drop(columns=["dt", "corte", "bloque"])
        if len(grupo) < 20:
            continue
        inicio = grupo["datetime_utc"].iloc[0]
        fin = grupo["datetime_utc"].iloc[-1]
        bloques.append({
            "datos": grupo,
            "inicio": inicio,
            "fin": fin,
            "duracion": (fin - inicio).total_seconds(),
            "n": len(grupo),
        })
    return bloques


def bloques_rellenados(datos: pd.DataFrame, max_hueco_horas: float = 2.0,
                       paso: pd.Timedelta = pd.Timedelta(minutes=5)) -> list:
    """Copia fiel del original. Devuelve una lista de DataFrames en malla uniforme."""
    d = datos.dropna(subset=VARIABLES).set_index("datetime_utc")[VARIABLES]
    corte = d.index.to_series().diff() > pd.Timedelta(hours=max_hueco_horas)

    bloques = []
    for _, g in d.groupby(corte.cumsum().values):
        bloque = rellenar_en_malla(g, paso)
        if bloque is not None:
            bloques.append(bloque)
    return bloques


def listar_huecos(datos: pd.DataFrame, umbral: pd.Timedelta) -> pd.DataFrame:
    """Tabla de huecos mayores que `umbral` (inicio, fin, duración en horas)."""
    t = pd.Series(pd.DatetimeIndex(datos["datetime_utc"]).sort_values())
    dt = t.diff()
    idx = dt[dt > umbral].index
    return pd.DataFrame({
        "inicio": t[idx - 1].values,
        "fin": t[idx].values,
        "duracion_h": (dt[idx].dt.total_seconds() / 3600).values,
    })


def tabla_bloques(bloques: list) -> pd.DataFrame:
    """Resumen de bloques de cualquiera de los dos tipos."""
    filas = []
    for i, b in enumerate(bloques):
        if isinstance(b, dict):
            ini, fin, n = b["inicio"], b["fin"], b["n"]
        else:
            ini, fin, n = b.index[0], b.index[-1], len(b)
        filas.append({"bloque": i, "inicio": ini, "fin": fin,
                      "dias": (fin - ini).total_seconds() / 86400, "muestras": n})
    return pd.DataFrame(filas)
