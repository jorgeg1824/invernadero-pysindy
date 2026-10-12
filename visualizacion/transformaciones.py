"""
Gráficas de la preparación del modelo (etapa 6): partición de los datos,
derivada calculada y normalización.
"""

from __future__ import annotations

import pandas as pd

from config import VARIABLES
from visualizacion.datos import COLORES, _fechas


def _t(d) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(d["datetime_utc"] if "datetime_utc" in d.columns else d.index)


def grafica_particion(fig, conjuntos: dict):
    """conjuntos: nombre -> DataFrame (entrenamiento, seleccion, validacion...)."""
    ax = fig.add_subplot()
    colores = {"entrenamiento": "0.45", "ajuste": "0.45", "seleccion": "C4", "validacion": "C0", "prediccion": "C0"}
    for nombre, d in conjuntos.items():
        dias = _t(d)[-1] - _t(d)[0]
        ax.plot(_t(d), d["Temperatura"], lw=0.6, color=colores.get(nombre, None),
                label=f"{nombre}: {len(d)} muestras, {dias.total_seconds() / 86400:.1f} días")
    ax.set_ylabel("Temperatura [°C]")
    ax.set_title("Etapa 6 - División del tramo")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    _fechas(ax)


def grafica_derivada(fig, train: pd.DataFrame, dTdt, unidad: str, dias_zoom: float = 2):
    """T y dT/dt del entrenamiento (acercamiento a los primeros días)."""
    t = _t(train)
    m = t <= t[0] + pd.Timedelta(days=dias_zoom)
    a1, a2 = fig.subplots(2, 1, sharex=True)
    a1.plot(t[m], train["Temperatura"].values[m], lw=0.8, color="C3")
    a1.set_ylabel("Temperatura [°C]")
    a1.grid(alpha=0.3)
    a2.plot(t[m], dTdt[m], lw=0.8, color="C4")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel(f"dT/dt [°C/{unidad}]")
    a2.grid(alpha=0.3)
    _fechas(a2)
    fig.suptitle(f"Etapa 6 - Derivada numérica usada por PySINDy (primeros {dias_zoom:g} días del entrenamiento)")


def grafica_normalizacion(fig, train: pd.DataFrame, escalador):
    tn = escalador.transform(train)
    ax = fig.add_subplot()
    for v in VARIABLES:
        ax.plot(_t(train), tn[v], lw=0.5, color=COLORES[v],
                label=f"{v}: media {escalador.media[v]:.4g}, desv. {escalador.std[v]:.4g}")
    ax.set_ylabel("Valor normalizado (z-score) [adimensional]")
    ax.set_title("Etapa 6 - Variables normalizadas con media y desviación del entrenamiento")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    _fechas(ax)
