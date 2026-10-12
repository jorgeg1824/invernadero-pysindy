"""
Estrategia A - Identificación base con PySINDy.

Obtiene la ecuación por regresión dispersa (STLSQ) sobre la derivada
numérica de T. No mira la simulación.

La única diferencia entre los dos programas originales es CÓMO se calcula
la derivada; por eso la derivada se elige con un objeto:

  DerivadaProfesor  -> invernadero_profesor.py  (datos irregulares, segundos)
  DerivadaMalla     -> invernadero_20dias.py    (malla uniforme, horas)
"""

from __future__ import annotations

from analisis.derivadas import (calcular_derivada_temperatura, derivada_savgol_directa,
                                tiempo_absoluto, unidad_efectiva)
from analisis.particion import describir
from modelos.coeficientes import a_fisico
from modelos.estrategias.base import Estrategia, ResultadoModelo
from modelos.sindy import ajustar_sindy, coeficientes_normalizados, terminos_activos


class DerivadaProfesor:
    """Suaviza con Savitzky-Golay y deriva con np.gradient sobre el tiempo real."""

    def __init__(self, ventana: int = 21, polyorder: int = 2, unidad: str = "s"):
        self.ventana, self.polyorder, self.unidad = ventana, polyorder, unidad
        self.unidad_tiempo = unidad_efectiva(unidad)

    def parametros(self):
        return {"derivada": "savgol+gradiente (tiempo real)", "ventana_savgol": self.ventana,
                "polyorder": self.polyorder, "unidad_tiempo": self.unidad,
                "unidad_efectiva": self.unidad_tiempo}

    def __call__(self, train):
        _, dTdt = calcular_derivada_temperatura(train, self.ventana, self.polyorder, self.unidad)
        return dTdt, tiempo_absoluto(train["datetime_utc"], self.unidad)


class DerivadaMalla:
    """Savitzky-Golay con deriv=1 sobre malla uniforme de paso dt (horas)."""

    unidad_tiempo = "h"

    def __init__(self, ventana: int = 21, polyorder: int = 2, dt: float = 5 / 60):
        self.ventana, self.polyorder, self.dt = ventana, polyorder, dt

    def parametros(self):
        return {"derivada": "savgol deriv=1 (malla uniforme)", "ventana_savgol": self.ventana,
                "polyorder": self.polyorder, "paso_h": self.dt}

    def __call__(self, train):
        return derivada_savgol_directa(train["Temperatura"].values, self.ventana,
                                       self.polyorder, self.dt), self.dt


class IdentificacionBase(Estrategia):

    nombre = "A_identificacion_base"

    def __init__(self, threshold: float, derivada, alpha: float = 0.001):
        self.threshold, self.derivada, self.alpha = threshold, derivada, alpha

    def parametros(self):
        return {"threshold": self.threshold, "alpha": self.alpha, **self.derivada.parametros()}

    def ejecutar(self, train, nombre_conjunto: str = "entrenamiento") -> ResultadoModelo:
        dTdt, t = self.derivada(train)
        modelo, escalador = ajustar_sindy(train, dTdt, self.threshold, self.alpha, t)
        c = coeficientes_normalizados(modelo)
        return ResultadoModelo(
            estrategia=self.nombre,
            parametros=self.parametros(),
            coef=a_fisico(c, escalador),
            unidad_tiempo=self.derivada.unidad_tiempo,
            datos_identificacion=describir(nombre_conjunto, train),
            coef_normalizados=c,
            modelo=modelo,
            escalador=escalador,
            terminos_activos=terminos_activos(modelo),
            detalles={"dTdt": dTdt},
        )
