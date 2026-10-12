"""
Interfaz común de las estrategias de modelado y del resultado que producen.

Cada estrategia deja escrito en su resultado:
  * qué estrategia se ejecutó y con qué parámetros,
  * con qué datos identificó o ajustó (y, si eligió algo, con qué datos lo eligió),
de modo que nunca se mezclen resultados de procedimientos distintos
(etapa 6 de la guía).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from modelos.coeficientes import constante_tiempo, ecuacion_fisica
from modelos.seleccion import es_estable


@dataclass
class ResultadoModelo:
    estrategia: str
    parametros: dict
    coef: dict                      # a0, aT, aL, aH en unidades físicas
    unidad_tiempo: str              # unidad de tiempo de los coeficientes
    datos_identificacion: dict      # describir(train) de analisis/particion.py
    coef_normalizados: np.ndarray | None = None
    modelo: object = None           # objeto PySINDy (si aplica)
    escalador: object = None
    terminos_activos: list = field(default_factory=list)
    datos_seleccion: dict | None = None
    origen: "ResultadoModelo | None" = None   # resultado del que parte (ajuste)
    detalles: dict = field(default_factory=dict)

    @property
    def estable(self) -> bool:
        return es_estable(self.coef)

    def ecuacion(self) -> str:
        return ecuacion_fisica(self.coef, self.unidad_tiempo)

    def resumen(self) -> str:
        lineas = [
            f"Estrategia: {self.estrategia}",
            f"Parámetros: {self.parametros}",
            f"Identificado con: {self.datos_identificacion['conjunto']} "
            f"{self.datos_identificacion['inicio']:%Y-%m-%d %H:%M} -> "
            f"{self.datos_identificacion['fin']:%Y-%m-%d %H:%M} "
            f"({self.datos_identificacion['muestras']} muestras)",
        ]
        if self.datos_seleccion:
            s = self.datos_seleccion
            lineas.append(f"Elegido con: {s['conjunto']} {s['inicio']:%Y-%m-%d %H:%M} -> "
                          f"{s['fin']:%Y-%m-%d %H:%M} ({s['muestras']} muestras)")
        lineas += [
            f"Términos activos: {', '.join(self.terminos_activos) or '-'}",
            self.ecuacion(),
            constante_tiempo(self.coef, self.unidad_tiempo),
        ]
        return "\n".join(lineas)


class Estrategia(ABC):
    """Una forma de obtener los coeficientes de la ecuación."""

    nombre: str = "estrategia"

    @abstractmethod
    def parametros(self) -> dict:
        ...

    @abstractmethod
    def ejecutar(self, train, **kwargs) -> ResultadoModelo:
        ...
