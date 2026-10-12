"""
Elección del tramo de datos con el que se identifica el modelo.

Dos criterios DIFERENTES:

  seleccionar_mejor_bloque()   invernadero_profesor.py
      Entre los bloques de al menos `dias_minimos` días, el de mayor puntaje
      0.55·log(1+horas) + 0.45·excitación   (excitación = variación relativa
      de T, L y H).

  elegir_bloque()              invernadero_20dias.py
      Entre los bloques de al menos dias_prediccion + dias_entrenamiento_min
      días, el que tiene MÁS muestras.

Ambas funciones devuelven también un texto con el motivo de la elección
(etapa 5 de la guía).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import VARIABLES


# ------------------------------------------------------------
# Criterio del programa del profesor
# ------------------------------------------------------------

def calcular_excitacion(datos: pd.DataFrame) -> float:
    """Mide cuánto cambian las variables (peso doble para T)."""
    resultado = {}
    for variable in VARIABLES:
        x = pd.to_numeric(datos[variable], errors="coerce").dropna()
        if len(x) < 2:
            resultado[variable] = 0.0
            continue
        rango = x.max() - x.min()
        std = x.std()
        escala = max(abs(x.median()), 1e-9)
        resultado[variable] = 0.5 * std / escala + 0.5 * rango / escala
    return (0.5 * resultado["Temperatura"]
            + 0.25 * resultado["Luminosidad"]
            + 0.25 * resultado["Humedad"])


# Nombre original (con tilde) para quien venga del código anterior.
calcular_excitación = calcular_excitacion


def puntuar_bloque(bloque: dict) -> float:
    datos = bloque["datos"]
    if len(datos) < 20:
        return -np.inf
    excitacion = calcular_excitacion(datos)
    duracion_horas = bloque["duracion"] / 3600.0
    score_duracion = np.log1p(max(duracion_horas, 0))
    return 0.55 * score_duracion + 0.45 * excitacion


def seleccionar_mejor_bloque(bloques: list, dias_minimos: float = 6):
    """
    Recibe la lista de obtener_bloques_continuos() (en el original la
    función recibía los datos y segmentaba adentro; aquí la segmentación es
    un paso aparte para poder inspeccionarla).

    Devuelve (mejor, bloques_validos_ordenados, motivo).
    """
    if not bloques:
        raise ValueError("No se encontraron bloques continuos.")

    validos = [b for b in bloques if b["duracion"] >= dias_minimos * 86400]
    if not validos:
        max_dias = max(b["duracion"] for b in bloques) / 86400.0
        raise ValueError(
            f"No existe un bloque continuo de {dias_minimos} días.\n\n"
            f"El bloque continuo más largo tiene {max_dias:.2f} días."
        )

    for b in validos:
        b["excitación"] = calcular_excitacion(b["datos"])
        b["excitacion"] = b["excitación"]
        b["score"] = puntuar_bloque(b)

    validos = sorted(validos, key=lambda x: x["score"], reverse=True)
    mejor = validos[0]
    motivo = (
        f"{len(bloques)} bloques continuos; {len(validos)} duran al menos "
        f"{dias_minimos:g} días. Se eligió el de mayor puntaje "
        f"0.55·log(1+horas) + 0.45·excitación = {mejor['score']:.3f} "
        f"(duración {mejor['duracion'] / 86400:.2f} días, excitación {mejor['excitacion']:.3f})."
    )
    return mejor, validos, motivo


# ------------------------------------------------------------
# Criterio de invernadero_20dias.py
# ------------------------------------------------------------

def elegir_bloque(bloques: list, dias_prediccion: float = 20, dias_entrenamiento_min: float = 5):
    """Devuelve (bloque, motivo). Igual al original: el válido con más muestras."""
    necesarios = pd.Timedelta(days=dias_prediccion + dias_entrenamiento_min)
    validos = [b for b in bloques if b.index[-1] - b.index[0] >= necesarios]
    if not validos:
        mas_largo = max(b.index[-1] - b.index[0] for b in bloques)
        raise ValueError(
            f"Se necesita un tramo de {dias_prediccion + dias_entrenamiento_min:g} dias.\n"
            f"El mas largo tiene {mas_largo.total_seconds() / 86400:.1f} dias."
        )
    bloque = max(validos, key=len)
    motivo = (
        f"{len(bloques)} tramos rellenados; {len(validos)} duran al menos "
        f"{dias_prediccion:g} + {dias_entrenamiento_min:g} días. Se eligió el de más "
        f"muestras ({len(bloque)}, {(bloque.index[-1] - bloque.index[0]).total_seconds() / 86400:.2f} días)."
    )
    return bloque, motivo
