"""
Lectura del CSV y validación básica de su estructura.

Origen: invernadero_profesor.py -> encontrar_columna(), detectar_columnas(),
cargar_csv().  invernadero_20dias.py y validar_modelo.py usaban estas mismas
funciones importándolas, así que hay una sola versión compartida.

Diferencias con el original (docs/informe_final.md):
  * C2: el original intentaba sep="\t" y solo si fallaba probaba ",".
        Un CSV separado por comas NO falla con sep="\t" (queda en una sola
        columna) y el programa terminaba diciendo que no existe
        'datetime_utc'. Ahora se reintenta con "," si la lectura con
        tabulador deja una sola columna. Con el CSV de ejemplo (tabulado)
        el resultado es idéntico.
  * Se eliminó un bloque sin efecto de cargar_csv() (un bucle cuyo cuerpo
    era `pass`).
  * La conversión de fechas pasó a datos/transformacion.py para poder
    inspeccionar el CSV crudo (etapa 1) separado de la conversión (etapa 2).
    cargar_csv() sigue haciendo ambas cosas, igual que el original.
"""

from __future__ import annotations

import pandas as pd

from config import VARIABLES, VARIABLES_ALIASES

COLUMNAS_FORMATO_LARGO = ["metric_name", "datetime_utc", "number_value"]


def encontrar_columna(df: pd.DataFrame, variable: str):
    """Busca la columna real correspondiente a una variable (formato ancho)."""
    if variable in df.columns:
        return variable
    for alias in VARIABLES_ALIASES[variable]:
        if alias in df.columns:
            return alias
    columnas_lower = {str(c).lower(): c for c in df.columns}
    for alias in VARIABLES_ALIASES[variable]:
        if alias.lower() in columnas_lower:
            return columnas_lower[alias.lower()]
    return None


def detectar_columnas(df: pd.DataFrame) -> dict:
    """Detecta las columnas necesarias en un CSV de formato ancho."""
    resultado = {}
    for variable in VARIABLES:
        columna = encontrar_columna(df, variable)
        if columna is None:
            raise ValueError(
                f"No se encontró la variable '{variable}'.\n\n"
                f"Columnas disponibles:\n{list(df.columns)}"
            )
        resultado[variable] = columna
    return resultado


def leer_csv_crudo(ruta) -> pd.DataFrame:
    """Lee el archivo sin transformar ninguna columna (todo queda como texto/número)."""
    df = pd.read_csv(ruta, sep="\t")
    if df.shape[1] == 1:          # C2: no era tabulado
        df = pd.read_csv(ruta)
    return df


def validar_estructura(df: pd.DataFrame) -> dict:
    """
    Validación básica. Devuelve un diccionario con lo encontrado y lanza
    ValueError con un mensaje claro si falta algo imprescindible.
    """
    if "datetime_utc" not in df.columns:
        raise ValueError("No se encontró la columna 'datetime_utc'.")

    formato_largo = "metric_name" in df.columns and "number_value" in df.columns
    info = {
        "filas": int(len(df)),
        "columnas": list(map(str, df.columns)),
        "formato": "largo (metric_name / number_value)" if formato_largo else "ancho",
    }

    if formato_largo:
        metricas = df["metric_name"].astype(str).str.lower()
        info["metricas"] = df["metric_name"].value_counts().to_dict()
        faltantes = [
            v for v in VARIABLES
            if not metricas.isin([a.lower() for a in VARIABLES_ALIASES[v]]).any()
        ]
        if faltantes:
            raise ValueError(f"El CSV no contiene datos para: {faltantes}")
    else:
        info["columnas_variables"] = detectar_columnas(df)

    return info


def cargar_csv(ruta):
    """
    Igual que el original: lee el CSV, convierte 'datetime_utc' a fecha UTC
    y devuelve (df, columnas).
    """
    from datos.transformacion import convertir_fechas

    df = leer_csv_crudo(ruta)
    if "datetime_utc" not in df.columns:
        raise ValueError("No se encontró la columna 'datetime_utc'.")

    df, _ = convertir_fechas(df)

    if "metric_name" in df.columns and "number_value" in df.columns:
        columnas = {variable: "number_value" for variable in VARIABLES}
    else:
        columnas = detectar_columnas(df)

    return df, columnas
