"""
Estrategia C - Selección mediante barrido de threshold.

Repite la identificación base (estrategia A) con varios valores de
threshold, simula cada ecuación sobre un conjunto de SELECCIÓN y se queda
con la de menor RMSE.

Origen: buscar_mejor_modelo() de invernadero_profesor.py.

IMPORTANTE: el original elige el threshold con el MISMO conjunto de
validación con el que después reporta las métricas. Esas métricas no son
una evaluación independiente. El resultado guarda con qué datos se eligió
(`datos_seleccion`) y la bandera `seleccion_independiente`; la mejora M2
(config.ConfigBarrido.seleccion_threshold = "separado") permite elegir con
un conjunto aparte.
"""

from __future__ import annotations

import numpy as np

from analisis.particion import describir
from evaluacion.metricas import calcular_metricas
from modelos.estrategias.base import Estrategia, ResultadoModelo
from modelos.seleccion import es_mejor


class BarridoThreshold(Estrategia):

    nombre = "C_barrido_threshold"

    def __init__(self, thresholds, crear_base, simular, aviso=print):
        """
        crear_base(threshold) -> IdentificacionBase
        simular(resultado, datos) -> trayectoria de T sobre `datos`
        aviso(texto): dónde escribir el progreso (el original usaba print).
        """
        self.thresholds = list(thresholds)
        self.crear_base = crear_base
        self.simular = simular
        self.aviso = aviso

    def parametros(self):
        return {"thresholds": self.thresholds, "criterio": "menor RMSE de simulación"}

    def ejecutar(self, train, seleccion=None, nombre_seleccion: str = "validacion",
                 independiente: bool = False) -> ResultadoModelo:
        tabla, mejor = [], None
        real = seleccion["Temperatura"].values

        for threshold in self.thresholds:
            try:
                res = self.crear_base(threshold).ejecutar(train)
                pred = self.simular(res, seleccion)
                metricas = calcular_metricas(real, pred)
                n_activos = int(np.sum(np.abs(res.modelo.coefficients()) > 1e-10))
                tabla.append({"threshold": threshold, **metricas, "terminos": n_activos})
                self.aviso(f"threshold={threshold:.5g} RMSE={metricas['RMSE']:.5f} "
                           f"R2={metricas['R2']:.5f} terminos={n_activos}")
                if mejor is None or es_mejor(metricas, mejor["metricas"]):
                    mejor = {"res": res, "threshold": threshold, "pred": pred, "metricas": metricas}
            except Exception as e:               # igual que el original: se informa y se sigue
                tabla.append({"threshold": threshold, "error": str(e)})
                self.aviso(f"Threshold {threshold} falló: {e}")

        if mejor is None:
            raise RuntimeError("Ninguno de los modelos pudo ser identificado.")

        base = mejor["res"]
        return ResultadoModelo(
            estrategia=self.nombre,
            parametros={**self.parametros(), "threshold_elegido": mejor["threshold"],
                        "identificacion": base.parametros},
            coef=base.coef,
            unidad_tiempo=base.unidad_tiempo,
            datos_identificacion=base.datos_identificacion,
            coef_normalizados=base.coef_normalizados,
            modelo=base.modelo,
            escalador=base.escalador,
            terminos_activos=base.terminos_activos,
            datos_seleccion=describir(nombre_seleccion, seleccion),
            origen=base,
            detalles={"barrido": tabla, "pred_seleccion": mejor["pred"],
                      "metricas_seleccion": mejor["metricas"],
                      "seleccion_independiente": independiente},
        )
