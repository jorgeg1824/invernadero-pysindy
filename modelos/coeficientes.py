"""
Conversión de coeficientes normalizados a unidades físicas y presentación
de la ecuación.

    dTn/dt = c0 + cT·Tn + cL·Ln + cH·Hn
      ->
    dT/dt  = a0 + aT·T  + aL·L  + aH·H

Origen: obtener_coeficientes_fisicos() del profesor y a_fisico() de
invernadero_20dias.py. Son la MISMA fórmula (una busca los coeficientes por
nombre, la otra por posición), así que obtener_coeficientes_fisicos() ahora
llama a a_fisico(). tests/test_modelos.py comprueba que coinciden.

La unidad de tiempo de los coeficientes es la de la derivada usada:
segundos en el flujo del profesor, horas en el de 20 días.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import VARIABLES

CLAVES = ["a0", "aT", "aL", "aH"]


def a_fisico(c, esc) -> dict:
    c0, cT, cL, cH = c
    sT, sL, sH = (esc.std[v] for v in VARIABLES)
    mT, mL, mH = (esc.media[v] for v in VARIABLES)
    return {
        "a0": sT * c0 - cT * mT - sT * cL * mL / sL - sT * cH * mH / sH,
        "aT": cT,
        "aL": sT * cL / sL,
        "aH": sT * cH / sH,
    }


def obtener_coeficientes_fisicos(modelo, escalador) -> dict:
    coef = np.asarray(modelo.coefficients()).flatten()
    d = dict(zip(modelo.get_feature_names(), coef))
    c = [d.get("1", 0.0), d.get("x0", 0.0), d.get("u0", 0.0), d.get("u1", 0.0)]
    return a_fisico(c, escalador)


def cambiar_unidad_tiempo(coef: dict, factor: float) -> dict:
    """Divide todos los coeficientes por `factor` (p. ej. 3600 para pasar de /h a /s)."""
    return {k: v / factor for k, v in coef.items()}


def por_segundo(coef: dict) -> dict:
    """invernadero_20dias.py: coeficientes por hora -> por segundo."""
    return cambiar_unidad_tiempo(coef, 3600)


def ecuacion_fisica(coef: dict, unidad: str | None = None) -> str:
    """Mismo formato que el original; opcionalmente indica la unidad de tiempo."""
    texto = (
        f"dT/dt = {coef['a0']:.8g} "
        f"+ ({coef['aT']:.8g})·T "
        f"+ ({coef['aL']:.8g})·L "
        f"+ ({coef['aH']:.8g})·H"
    )
    if unidad:
        texto += f"     [°C/{unidad}]"
    return texto


def constante_tiempo(coef: dict, unidad: str = "h") -> str:
    """τ = -1/aT, si el modelo es estable."""
    if coef["aT"] >= 0:
        return "inestable (aT ≥ 0): la simulación diverge"
    tau = -1.0 / coef["aT"]
    a_minutos = {"s": 1 / 60, "h": 60, "ks": 1000 / 60}.get(unidad)
    if a_minutos:
        return f"τ = {tau * a_minutos:.3g} min"
    return f"τ = {tau:.4g} {unidad}"


def tabla_coeficientes(coef: dict, coef_normalizados=None) -> pd.DataFrame:
    filas = {"termino": ["constante", "T", "L", "H"], "fisico": [coef[k] for k in CLAVES]}
    if coef_normalizados is not None:
        filas["normalizado"] = list(coef_normalizados)
        filas["activo"] = [abs(v) > 1e-10 for v in coef_normalizados]
    return pd.DataFrame(filas)
