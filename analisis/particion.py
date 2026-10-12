"""
División de un tramo en entrenamiento y validación.

Tres procedimientos de los programas originales, más una mejora opcional:

  dividir_train_validacion()  invernadero_profesor.py
      Los PRIMEROS `dias_entrenamiento` días entrenan; todo lo que sigue valida.

  dividir_ultimos_dias()      invernadero_20dias.py (antes: dividir)
      Los ÚLTIMOS `dias_prediccion` días validan; todo lo anterior entrena
      (el entrenamiento puede durar más de dias_entrenamiento_min).

  cortes_ventanas()           validar_modelo.py
      Recorta el final de un tramo cada `paso` días para obtener varias
      ventanas de predicción (se solapan y comparten parte del entrenamiento).

  separar_seleccion()         MEJORA M2 (no existía)
      Aparta el final del entrenamiento como conjunto de SELECCIÓN del
      threshold, para que la validación sea una prueba independiente.
"""

from __future__ import annotations

import pandas as pd


def dividir_train_validacion(datos: pd.DataFrame, dias_entrenamiento: float = 5):
    """datos con columna 'datetime_utc'. Copia fiel del original."""
    datos = datos.sort_values("datetime_utc").copy()
    datos = datos.drop_duplicates(subset=["datetime_utc"], keep="first").reset_index(drop=True)

    inicio = datos["datetime_utc"].iloc[0]
    limite = inicio + pd.Timedelta(days=dias_entrenamiento)

    train = datos[datos["datetime_utc"] < limite].copy()
    validacion = datos[datos["datetime_utc"] >= limite].copy()

    if len(train) < 20:
        raise ValueError("El tramo de entrenamiento contiene muy pocos datos.")
    if len(validacion) < 20:
        raise ValueError(
            f"No quedan suficientes datos posteriores a los {dias_entrenamiento} "
            "días de entrenamiento para validar.")
    return train, validacion


def dividir_ultimos_dias(bloque: pd.DataFrame, dias_prediccion: float = 20):
    """bloque indexado por fecha. Copia fiel de dividir() de invernadero_20dias."""
    limite = bloque.index[-1] - pd.Timedelta(days=dias_prediccion)
    return bloque[bloque.index <= limite], bloque[bloque.index > limite]


def cortes_ventanas(bloque: pd.DataFrame, minimo_dias: float, paso_dias: int = 4) -> list:
    """
    Duraciones (en días desde el inicio del bloque) en las que se corta el
    bloque para formar cada ventana. Igual que validar_modelo.py:
        sorted({dias} | set(range(minimo, int(dias), paso)))
    """
    dias = (bloque.index[-1] - bloque.index[0]).total_seconds() / 86400
    return sorted({dias} | set(range(int(minimo_dias), int(dias), paso_dias)))


def recortar(bloque: pd.DataFrame, dias: float) -> pd.DataFrame:
    return bloque[bloque.index <= bloque.index[0] + pd.Timedelta(days=dias)]


def separar_seleccion(train: pd.DataFrame, fraccion: float = 0.2):
    """
    MEJORA M2. Divide el entrenamiento en (ajuste, seleccion) por tiempo:
    el último `fraccion` del entrenamiento se usa para elegir el threshold.
    """
    t = train["datetime_utc"]
    limite = t.iloc[0] + (t.iloc[-1] - t.iloc[0]) * (1 - fraccion)
    ajuste = train[t < limite].copy()
    seleccion = train[t >= limite].copy()
    if len(ajuste) < 20 or len(seleccion) < 20:
        raise ValueError("El entrenamiento es demasiado corto para separar un conjunto de selección.")
    return ajuste, seleccion


def describir(nombre: str, d: pd.DataFrame) -> dict:
    """Resumen de un conjunto (sirve para DataFrames con columna o índice de fechas)."""
    t = d["datetime_utc"] if "datetime_utc" in d.columns else pd.Series(d.index)
    return {
        "conjunto": nombre,
        "inicio": t.iloc[0],
        "fin": t.iloc[-1],
        "dias": (t.iloc[-1] - t.iloc[0]).total_seconds() / 86400,
        "muestras": int(len(d)),
    }
