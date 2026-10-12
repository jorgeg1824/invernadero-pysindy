"""
Funcionalidad 'estrategias' (nueva)

Aplica las estrategias A, B y C a los mismos datos (tramo y partición de
'prediccion') para ver qué aporta cada una. C elige el threshold con el
final del entrenamiento, así la validación es independiente para las tres.
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pandas as pd

import config as C
from analisis import particion as P
from analisis.seleccion_tramo import elegir_bloque
from evaluacion import reportes as R
from evaluacion.metricas import calcular_metricas
from modelos.estrategias.ajuste_simulacion import AjusteSimulacion
from modelos.estrategias.barrido_threshold import BarridoThreshold
from modelos.estrategias.identificacion_base import (DerivadaMalla, IdentificacionBase)
from simulacion.integrador import simular_lsim
from visualizacion import modelos as VM
from visualizacion import transformaciones as VT

from flujos.comun import Contexto, etapas_datos, paso_malla
from flujos.prediccion import etapa4_relleno, preparar_bloques


def comparar_estrategias(ruta_csv=None, cfg: C.ConfigPrediccion | None = None,
                         thresholds=None, fraccion_seleccion: float = 0.2, guardar=True,
                         carpeta=None, aviso=print, graficas=True) -> R.Ejecucion:
    """
    A, B y C con los mismos datos (tramo y partición del flujo 'prediccion').
    C elige el threshold con el final del entrenamiento, de modo que la
    validación es independiente para las tres.
    """
    cfg = cfg or C.ConfigPrediccion()
    thresholds = thresholds or C.ConfigBarrido().thresholds
    ctx = Contexto("estrategias", cfg, guardar, carpeta, aviso, graficas)
    ctx.ejec.config.update({"thresholds_C": thresholds, "fraccion_seleccion_C": fraccion_seleccion})
    datos, _ = etapas_datos(ctx, ruta_csv or C.CSV_EJEMPLO, cfg.tolerancia_alineacion)
    dt_h = paso_malla(cfg).total_seconds() / 3600

    bloques = preparar_bloques(datos, cfg)
    bloque, motivo = elegir_bloque(bloques, cfg.dias_prediccion, cfg.dias_entrenamiento_min)
    tabla = etapa4_relleno(ctx, datos, bloques, bloque, cfg)
    ctx.etapa(R.Etapa(5, "Selección del tramo", texto=motivo, tablas={"candidatos": tabla}))

    train, val = P.dividir_ultimos_dias(bloque, cfg.dias_prediccion)
    t_sel = train.index[0] + (train.index[-1] - train.index[0]) * (1 - fraccion_seleccion)
    ajuste_c, sel_c = train[train.index < t_sel], train[train.index >= t_sel]
    conjuntos = {"entrenamiento": train, "seleccion": sel_c, "validacion": val}
    tabla_part = pd.DataFrame([P.describir(k, v) for k, v in conjuntos.items()])
    ctx.etapa(R.Etapa(6, "Preparación del modelo",
                      texto="A y B se identifican con todo el entrenamiento. C ajusta con el entrenamiento "
                            "sin su parte final y elige el threshold con esa parte final (selección).\n\n"
                            + tabla_part.to_string(index=False),
                      tablas={"particion": tabla_part},
                      graficas={"particion": partial(VT.grafica_particion, conjuntos=conjuntos)}))

    derivada = DerivadaMalla(cfg.ventana_savgol, cfg.polyorder_savgol, dt_h)
    simular = lambda coef, d: simular_lsim(coef, d, dt_h)
    A = IdentificacionBase(cfg.threshold, derivada, cfg.alpha).ejecutar(train)
    B = AjusteSimulacion(simular).ejecutar(train, base=A)
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        Cr = BarridoThreshold(thresholds, lambda th: IdentificacionBase(th, derivada, cfg.alpha),
                              lambda res, d: simular(res.coef, d), aviso=lambda s: ctx.reg.info(7, s)
                              ).ejecutar(ajuste_c, sel_c, "seleccion (final del entrenamiento)", True)
    estr = {"A_identificacion_base": A, "B_ajuste_por_simulacion": B, "C_barrido_threshold": Cr}

    filas, preds, metr = [], {}, {}
    real = val["Temperatura"].values
    for nombre, res in estr.items():
        with np.errstate(over="ignore", invalid="ignore"):
            p = simular(res.coef, val)
        m = calcular_metricas(real, p)
        preds[nombre], metr[nombre] = p, m
        filas.append({"estrategia": nombre, "estable": res.estable, "terminos": ", ".join(res.terminos_activos),
                      **{k: v for k, v in res.coef.items()},
                      "identificado_con": f"{res.datos_identificacion['conjunto']} ({res.datos_identificacion['muestras']})",
                      "elegido_con": (res.datos_seleccion or {}).get("conjunto", "-"), **m})
    tabla_e = pd.DataFrame(filas)
    ctx.etapa(R.Etapa(7, "Identificación",
                      texto="\n\n".join(f"--- {k} ---\n{v.resumen()}" for k, v in estr.items())
                            + "\n\n" + R.texto_barrido(Cr.detalles["barrido"]),
                      tablas={"estrategias": tabla_e, "barrido_C": pd.DataFrame(Cr.detalles["barrido"])},
                      graficas={"coeficientes": partial(VM.grafica_coeficientes, resultados=estr),
                                "barrido_C": partial(VM.grafica_barrido, tabla=Cr.detalles["barrido"],
                                                     elegido=Cr.parametros["threshold_elegido"],
                                                     conjunto="seleccion")}))
    texto = ("COMPARACIÓN DE ESTRATEGIAS sobre la misma validación "
             f"({val.index[0]:%Y-%m-%d} -> {val.index[-1]:%Y-%m-%d})\n\n"
             + tabla_e[["estrategia", "estable", "terminos", "RMSE", "MAE", "R2", "identificado_con", "elegido_con"]]
             .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    ctx.etapa(R.Etapa(8, "Simulación y evaluación", texto=texto,
                      tablas={"trayectorias": pd.DataFrame({"T_real": real, **preds}, index=val.index)},
                      graficas={"estrategias": partial(VM.grafica_estrategias, val=val, preds=preds, metricas=metr)}))
    ctx.ejec.resultado.update({"tabla": tabla_e.to_dict("records")})
    return ctx.cerrar(texto)
