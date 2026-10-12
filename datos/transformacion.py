"""
Transformaciones del CSV: conversión de fechas y paso del formato largo
(una fila por medición) a una serie por variable.

Origen: invernadero_profesor.py -> construir_series() y la parte de
cargar_csv() que convertía 'datetime_utc'.

Zona horaria: las fechas del CSV de ejemplo terminan en "Z"
(p. ej. 2026-06-15T15:36:17Z), es decir, vienen EXPLÍCITAMENTE en UTC.
analizar_zona_horaria() lo comprueba en cada ejecución y lo deja escrito en
el registro. Si un archivo no trae zona, pandas las interpreta como UTC
porque se usa utc=True; en ese caso el registro lo advierte en lugar de
asumirlo en silencio.
"""

from __future__ import annotations

import re

import pandas as pd

from config import VARIABLES, VARIABLES_ALIASES

_PATRON_ZONA = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def analizar_zona_horaria(texto: pd.Series) -> dict:
    """Revisa si las fechas en texto declaran su zona horaria."""
    muestras = texto.dropna().astype(str).str.strip()
    con_zona = muestras.str.contains(_PATRON_ZONA)
    sufijos = muestras[con_zona].str.extract(r"(Z|[+-]\d{2}:?\d{2})$")[0].value_counts().to_dict()
    total = int(len(muestras))
    n = int(con_zona.sum())
    if total and n == total:
        if set(sufijos) == {"Z"}:
            conclusion = "Todas las fechas terminan en 'Z': están explícitamente en UTC."
        else:
            conclusion = f"Todas las fechas declaran su zona: {sufijos}. Se convierten a UTC."
    elif n == 0:
        conclusion = ("ADVERTENCIA: las fechas NO declaran zona horaria. "
                      "Se interpretan como UTC (utc=True); verifique que sea correcto.")
    else:
        conclusion = (f"ADVERTENCIA: solo {n} de {total} fechas declaran zona. "
                      "Las que no la tienen se interpretan como UTC.")
    return {"total": total, "con_zona": n, "sufijos": sufijos, "conclusion": conclusion}


def convertir_fechas(df: pd.DataFrame, columna: str = "datetime_utc"):
    """
    Convierte la columna de fechas a datetime UTC (como el original:
    pd.to_datetime(utc=True, errors="coerce")).

    Devuelve (df_convertido, info) donde info registra lo ocurrido.
    """
    original = df[columna]
    zona = analizar_zona_horaria(original) if original.dtype == object or \
        pd.api.types.is_string_dtype(original) else {"conclusion": "La columna ya era de tipo fecha."}

    convertido = pd.to_datetime(original, utc=True, errors="coerce")

    df = df.copy()
    df[columna] = convertido

    no_convertidas = int(convertido.isna().sum() - original.isna().sum())
    info = {
        "tipo_antes": str(original.dtype),
        "tipo_despues": str(convertido.dtype),
        "fechas_no_convertibles": no_convertidas,
        "ordenado_antes": bool(original.is_monotonic_increasing),
        "ordenado_despues": bool(convertido.dropna().is_monotonic_increasing),
        "zona_horaria": zona,
        "primera": str(convertido.min()),
        "ultima": str(convertido.max()),
    }
    return df, info


def construir_series(df: pd.DataFrame) -> dict:
    """
    Convierte el formato  datetime_utc | metric_name | number_value
    en un DataFrame por variable:  datetime_utc | <Variable>.
    (Copia fiel del original.)
    """
    if "metric_name" not in df.columns:
        raise ValueError("Se esperaba una columna 'metric_name'.")
    if "number_value" not in df.columns:
        raise ValueError("Se esperaba una columna 'number_value'.")

    datos = df.copy()
    datos["datetime_utc"] = pd.to_datetime(datos["datetime_utc"], utc=True, errors="coerce")
    datos["number_value"] = pd.to_numeric(datos["number_value"], errors="coerce")
    datos = datos.dropna(subset=["datetime_utc", "number_value"])

    resultado = {}
    for variable in VARIABLES:
        aliases = VARIABLES_ALIASES[variable]
        mascara = datos["metric_name"].astype(str).str.lower().isin([a.lower() for a in aliases])
        temp = datos.loc[mascara, ["datetime_utc", "number_value"]].copy()
        if temp.empty:
            raise ValueError(f"No se encontraron datos para {variable}")
        temp = temp.sort_values("datetime_utc")
        temp = temp.rename(columns={"number_value": variable})
        resultado[variable] = temp

    return resultado


def resumen_construccion(df: pd.DataFrame, series: dict) -> dict:
    """Cuántos registros entran, cuántos se descartan y cuántos quedan por variable."""
    valores = pd.to_numeric(df["number_value"], errors="coerce")
    fechas = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    validos = valores.notna() & fechas.notna()
    usados = set()
    for v in VARIABLES:
        usados |= {a.lower() for a in VARIABLES_ALIASES[v]}
    metricas = df["metric_name"].astype(str)
    ignoradas = metricas[~metricas.str.lower().isin(usados)].value_counts().to_dict()
    return {
        "filas_entrada": int(len(df)),
        "filas_sin_fecha_o_valor": int((~validos).sum()),
        "metricas_ignoradas": {k: int(v) for k, v in ignoradas.items()},
        "muestras_por_variable": {v: int(len(s)) for v, s in series.items()},
    }
