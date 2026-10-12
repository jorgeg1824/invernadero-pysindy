"""
Estrategia B - Ajuste de coeficientes mediante simulación.

Parte del resultado de la estrategia A y cambia el VALOR de sus
coeficientes para minimizar el error de la SIMULACIÓN en el entrenamiento
(no el error de la derivada). Restricciones del original, conservadas:

  * solo se ajustan los términos que PySINDy dejó activos (más T siempre);
  * el ajuste se hace sobre los coeficientes NORMALIZADOS;
  * aT se escribe como -exp(q) para que el modelo sea siempre estable;
  * si PySINDy dio aT >= 0 (inestable), se arranca desde aT = -1.

Origen: segunda mitad de identificar() en invernadero_20dias.py.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from analisis.particion import describir
from modelos.coeficientes import a_fisico
from modelos.estrategias.base import Estrategia, ResultadoModelo
from modelos.sindy import NOMBRES_TERMINOS


class AjusteSimulacion(Estrategia):

    nombre = "B_ajuste_por_simulacion"

    def __init__(self, simular, descripcion_simulador: str = "lsim"):
        """simular(coef, datos) -> trayectoria de T (misma unidad de tiempo que coef)."""
        self.simular = simular
        self.descripcion_simulador = descripcion_simulador

    def parametros(self):
        return {"optimizador": "scipy.optimize.least_squares",
                "objetivo": "error de simulación en el entrenamiento",
                "restriccion": "aT = -exp(q) < 0",
                "simulador": self.descripcion_simulador}

    def ejecutar(self, train, base: ResultadoModelo = None,
                 nombre_conjunto: str = "entrenamiento") -> ResultadoModelo:
        if base is None or base.coef_normalizados is None:
            raise ValueError("El ajuste por simulación necesita el resultado de la identificación base.")

        c_sindy = base.coef_normalizados
        esc = base.escalador

        activos = np.flatnonzero(c_sindy != 0)
        activos = np.union1d(activos, [1])
        pos_T = np.searchsorted(activos, 1)

        def armar(q):
            c = np.zeros(4)
            c[activos] = q
            c[1] = -np.exp(q[pos_T])
            return c

        q0 = c_sindy[activos].copy()
        q0[pos_T] = np.log(-c_sindy[1]) if c_sindy[1] < 0 else 0.0

        real = train["Temperatura"].values
        ajuste = least_squares(lambda q: self.simular(a_fisico(armar(q), esc), train) - real, q0)
        c = armar(ajuste.x)

        nombres = ["1", "x0", "u0", "u1"]
        return ResultadoModelo(
            estrategia=self.nombre,
            parametros={**self.parametros(), "parte_de": base.estrategia,
                        "terminos_ajustados": [NOMBRES_TERMINOS[nombres[i]] for i in activos]},
            coef=a_fisico(c, esc),
            unidad_tiempo=base.unidad_tiempo,
            datos_identificacion=describir(nombre_conjunto, train),
            coef_normalizados=c,
            modelo=None,
            escalador=esc,
            terminos_activos=[NOMBRES_TERMINOS[nombres[i]] for i in range(4) if abs(c[i]) > 1e-10],
            origen=base,
            detalles={"q0": q0, "costo_final": float(ajuste.cost), "evaluaciones": int(ajuste.nfev),
                      "arranque_aT": "aT de PySINDy" if c_sindy[1] < 0 else "aT = -1 (PySINDy inestable)"},
        )
