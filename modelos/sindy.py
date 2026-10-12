"""
Construcción y ajuste del modelo PySINDy.

    dTn/dt = c0 + cT·Tn + cL·Ln + cH·Hn      (variables normalizadas)

Origen:
  * crear_modelo()  invernadero_profesor.py. invernadero_20dias.py creaba
    el MISMO modelo en línea (PolynomialLibrary(degree=1) con sesgo,
    STLSQ(threshold, alpha=1e-3, normalize_columns=True)), así que se
    comparte.
  * ajustar_sindy() reúne la parte común de identificar_modelo() (profesor)
    e identificar() (20dias): normalizar con el entrenamiento y ajustar
    PySINDy con una derivada ya calculada. Lo que difiere entre programas
    (cómo se calcula la derivada) se pasa como argumento; ver
    modelos/estrategias/identificacion_base.py.
"""

from __future__ import annotations

import numpy as np
import pysindy as ps

from analisis.normalizacion import Escalador
from config import ENTRADAS

NOMBRES_TERMINOS = {"1": "constante", "x0": "T", "u0": "L", "u1": "H"}


def crear_modelo(threshold: float, alpha: float = 0.001) -> ps.SINDy:
    library = ps.PolynomialLibrary(degree=1, include_bias=True)
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True)
    # No se usa la diferenciación interna: la derivada se entrega explícita.
    return ps.SINDy(optimizer=optimizer, feature_library=library)


def ajustar_sindy(train, dTdt, threshold: float, alpha: float = 0.001, t=None):
    """
    train: DataFrame con columnas Temperatura, Luminosidad, Humedad.
    dTdt:  derivada física de T (°C por unidad de tiempo).
    t:     se pasa a PySINDy como en el original (no afecta el ajuste porque
           la derivada ya viene calculada).

    Devuelve (modelo, escalador).
    """
    escalador = Escalador()
    escalador.fit(train)
    tn = escalador.transform(train)

    T = tn[["Temperatura"]].values
    U = tn[ENTRADAS].values
    dTdt_n = np.asarray(dTdt) / escalador.std["Temperatura"]

    modelo = crear_modelo(threshold, alpha)
    modelo.fit(T, u=U, t=t, x_dot=dTdt_n.reshape(-1, 1))
    return modelo, escalador


def coeficientes_normalizados(modelo) -> np.ndarray:
    """[c0, cT, cL, cH] en el orden de PySINDy (1, x0, u0, u1)."""
    nombres = list(modelo.get_feature_names())
    if nombres != ["1", "x0", "u0", "u1"]:
        raise ValueError(f"Orden de términos inesperado: {nombres}")
    return np.asarray(modelo.coefficients()).ravel().astype(float)


def terminos_activos(modelo, tol: float = 1e-10) -> list:
    c = coeficientes_normalizados(modelo)
    return [NOMBRES_TERMINOS[n] for n, v in zip(modelo.get_feature_names(), c) if abs(v) > tol]


def texto_modelo(modelo) -> str:
    """Ecuación en variables normalizadas (equivale a modelo.print())."""
    c = coeficientes_normalizados(modelo)
    terminos = [f"{v:+.4g}·{n}" for v, n in zip(c, modelo.get_feature_names()) if abs(v) > 1e-10]
    return "(x0)' = " + " ".join(terminos) + "     [x0 = T, u0 = L, u1 = H normalizadas]"
