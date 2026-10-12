"""
Alineación temporal de Temperatura, Luminosidad y Humedad.

Origen: invernadero_profesor.py -> estimar_dt_median(), alinear_variables().
invernadero_20dias.py y validar_modelo.py importaban estas mismas
funciones: el criterio temporal es el mismo en los tres programas, por eso
se comparte una única versión.

Criterio: la Temperatura es la referencia. A cada instante de T se le
asigna la L y la H con la marca de tiempo MÁS CERCANA (merge_asof
'nearest') siempre que esté a menos de `factor_tolerancia` x dt típico.
No se crea una malla nueva.

diferencias_temporales() es nueva: solo diagnostica (no cambia los datos)
para poder ver en la etapa 3 qué tan separadas estaban las mediciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimar_dt_median(serie: pd.DataFrame) -> float:
    """Intervalo típico (mediana de los dt positivos), en segundos."""
    t = serie["datetime_utc"].sort_values()
    dt = t.diff().dt.total_seconds()
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return np.nan
    return float(dt.median())


def alinear_variables(series: dict, factor_tolerancia: float = 2.0):
    """
    Devuelve (datos_alineados, dt_referencia_segundos).
    datos_alineados.attrs['timestamps_duplicados_eliminados'] registra
    cuántas marcas repetidas se quitaron.
    """
    T = series["Temperatura"].copy().sort_values("datetime_utc")
    L = series["Luminosidad"].copy().sort_values("datetime_utc")
    H = series["Humedad"].copy().sort_values("datetime_utc")

    dt_referencia = np.nanmedian([estimar_dt_median(T), estimar_dt_median(L), estimar_dt_median(H)])
    if not np.isfinite(dt_referencia):
        raise ValueError("No fue posible determinar el intervalo temporal.")

    tolerancia = pd.Timedelta(seconds=factor_tolerancia * dt_referencia)

    datos = pd.merge_asof(T, L, on="datetime_utc", direction="nearest", tolerance=tolerancia)
    datos = pd.merge_asof(datos.sort_values("datetime_utc"), H, on="datetime_utc",
                          direction="nearest", tolerance=tolerancia)
    datos = datos.sort_values("datetime_utc")

    # PySINDy y np.gradient requieren tiempo estrictamente creciente.
    duplicados = datos["datetime_utc"].duplicated().sum()
    datos = datos.drop_duplicates(subset=["datetime_utc"], keep="first").reset_index(drop=True)
    datos.attrs["timestamps_duplicados_eliminados"] = int(duplicados)
    datos.attrs["tolerancia_s"] = float(tolerancia.total_seconds())

    return datos, dt_referencia


def diferencias_temporales(series: dict, factor_tolerancia: float = 2.0) -> pd.DataFrame:
    """
    Para cada marca de Temperatura, a cuántos segundos está la medición de
    L y de H más cercana. Solo diagnóstico.
    """
    T = series["Temperatura"][["datetime_utc"]].sort_values("datetime_utc").copy()
    resultado = T.copy()
    for var in ["Luminosidad", "Humedad"]:
        otra = series[var][["datetime_utc"]].sort_values("datetime_utc").copy()
        otra["t_" + var] = otra["datetime_utc"]
        m = pd.merge_asof(T, otra, on="datetime_utc", direction="nearest")
        resultado["desfase_" + var + "_s"] = (m["t_" + var] - m["datetime_utc"]).dt.total_seconds().values
    return resultado.reset_index(drop=True)


def resumen_alineacion(series: dict, datos: pd.DataFrame, dt_ref: float) -> dict:
    tol = datos.attrs.get("tolerancia_s", np.nan)
    sin_par = {v: int(datos[v].isna().sum()) for v in ["Luminosidad", "Humedad"]}
    return {
        "dt_por_variable_s": {v: estimar_dt_median(s) for v, s in series.items()},
        "dt_referencia_s": float(dt_ref),
        "tolerancia_s": tol,
        "filas_temperatura": int(len(series["Temperatura"])),
        "filas_alineadas": int(len(datos)),
        "duplicados_eliminados": int(datos.attrs.get("timestamps_duplicados_eliminados", 0)),
        "sin_pareja_dentro_de_tolerancia": sin_par,
    }
