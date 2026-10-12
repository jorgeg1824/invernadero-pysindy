"""
Funcionalidad 'barrido' = invernadero_profesor.py

Tramo por puntaje, limpieza del tramo, primeros días para entrenar y
estrategia C (barrido de threshold sobre la identificación base A).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

import config as C
from analisis import particion as P
from analisis.normalizacion import Escalador
from analisis.seleccion_tramo import seleccionar_mejor_bloque
from datos import limpieza, segmentacion
from evaluacion import reportes as R
from evaluacion.metricas import calcular_metricas
from modelos.coeficientes import constante_tiempo, tabla_coeficientes
from modelos.estrategias.barrido_threshold import BarridoThreshold
from modelos.estrategias.identificacion_base import (DerivadaProfesor, IdentificacionBase)
from modelos.sindy import texto_modelo
from simulacion.integrador import simular_modelo
from visualizacion import datos as VD
from visualizacion import modelos as VM
from visualizacion import transformaciones as VT

from flujos.comun import Contexto, etapas_datos


def ejecutar_barrido(ruta_csv=None, cfg: C.ConfigBarrido | None = None, guardar=True,
                     carpeta=None, aviso=print, graficas=True) -> R.Ejecucion:
    cfg = cfg or C.ConfigBarrido()
    ctx = Contexto("barrido", cfg, guardar, carpeta, aviso, graficas)
    ruta_csv = ruta_csv or C.CSV_EJEMPLO
    datos, dt_ref = etapas_datos(ctx, ruta_csv, cfg.tolerancia_alineacion)

    # ---------- Etapa 4: segmentación en bloques continuos (sin interpolar)
    bloques = segmentacion.obtener_bloques_continuos(datos, cfg.factor_corte_bloques)
    dt_med = datos["datetime_utc"].diff().dt.total_seconds().median()
    umbral = pd.Timedelta(seconds=cfg.factor_corte_bloques * dt_med)
    huecos = segmentacion.listar_huecos(datos, umbral)
    tabla = segmentacion.tabla_bloques(bloques)
    en_bloques = int(tabla["muestras"].sum()) if len(tabla) else 0
    ctx.reg.cantidades(4, "Filas alineadas -> filas dentro de bloques (>= 20 muestras)", len(datos), en_bloques)

    # ---------- Etapa 5: selección del tramo + limpieza del tramo
    mejor, validos, motivo = seleccionar_mejor_bloque(bloques, cfg.dias_minimos_tramo)
    tramo = limpieza.limpiar_tramo(mejor["datos"], cfg.factor_limpieza)
    info_l = limpieza.resumen_limpieza_tramo(mejor["datos"], tramo)
    elegido = next(i for i, b in enumerate(bloques) if b is mejor)
    for b in bloques:
        b.setdefault("score", np.nan)
    tabla["puntaje"] = [b.get("score", np.nan) for b in bloques]
    tabla["excitacion"] = [b.get("excitacion", np.nan) for b in bloques]

    ctx.etapa(R.Etapa(
        4, "Limpieza y segmentación",
        texto=f"Se corta un bloque cuando dt > {cfg.factor_corte_bloques:g} x dt mediano "
              f"({umbral}). No se interpola: los bloques conservan las muestras originales; "
              f"se descartan bloques con < 20 muestras.\n\n"
              f"Huecos: {len(huecos)}   Bloques: {len(bloques)}\n"
              f"Filas alineadas: {len(datos)}   dentro de bloques: {en_bloques}   "
              f"descartadas: {len(datos) - en_bloques}\nInterpoladas: 0\n\n"
              f"Huecos más largos:\n{huecos.sort_values('duracion_h', ascending=False).head(10).to_string(index=False)}",
        datos={"umbral_hueco": str(umbral), "huecos": len(huecos), "bloques": len(bloques),
               "filas_en_bloques": en_bloques, "interpoladas": 0},
        tablas={"huecos": huecos, "bloques": tabla},
        graficas={"huecos": partial(VD.grafica_huecos, datos=datos, bloques=bloques, huecos=huecos,
                                    umbral_txt=str(umbral))},
    ))
    ctx.reg.cantidades(5, "limpiar_tramo() sobre el tramo elegido", info_l["filas_antes"], info_l["filas_despues"])
    ctx.etapa(R.Etapa(
        5, "Selección del tramo",
        texto=f"{motivo}\n\nTramo: {mejor['inicio']} -> {mejor['fin']}\n\n"
              f"Limpieza del tramo (limpiar_tramo):\n{R.texto_dict(info_l)}\n\n"
              f"Candidatos:\n{tabla.to_string(index=False)}",
        datos={"motivo": motivo, "inicio": mejor["inicio"], "fin": mejor["fin"],
               "duracion_dias": mejor["duracion"] / 86400, "puntaje": mejor["score"],
               "excitacion": mejor["excitacion"], "limpieza": info_l},
        tablas={"candidatos": tabla, "tramo_elegido": tramo},
        graficas={"seleccion": partial(VD.grafica_seleccion_tramo, tabla=tabla, elegido=elegido,
                                       minimo_dias=cfg.dias_minimos_tramo, criterio="mayor puntaje"),
                  "tramo": partial(VD.grafica_variables, datos=tramo,
                                   titulo="Etapa 5 - Tramo elegido (después de limpiar_tramo)")},
    ))

    # ---------- Etapa 6: preparación (partición, derivada, normalización)
    train, val = P.dividir_train_validacion(tramo, cfg.dias_entrenamiento)
    if cfg.seleccion_threshold == "separado":
        ajuste, seleccion = P.separar_seleccion(train, cfg.fraccion_seleccion)
        conjuntos = {"ajuste": ajuste, "seleccion": seleccion, "validacion": val}
        nombre_sel, independiente = "seleccion (final del entrenamiento)", True
    elif cfg.seleccion_threshold == "validacion":
        ajuste, seleccion = train, val
        conjuntos = {"entrenamiento": train, "validacion": val}
        nombre_sel, independiente = "validacion (la misma que se reporta)", False
    else:
        raise ValueError(f"seleccion_threshold desconocido: {cfg.seleccion_threshold}")

    derivada = DerivadaProfesor(cfg.ventana_savgol, cfg.polyorder_savgol, cfg.unidad_tiempo)
    dTdt, _ = derivada(ajuste)
    esc = Escalador().fit(ajuste)
    if derivada.unidad_tiempo != "s":
        ctx.reg.info(6, f"ATENCIÓN: unidad_tiempo='{cfg.unidad_tiempo}' produce tiempo en "
                        f"{derivada.unidad_tiempo} con esta versión de pandas (comportamiento original).")
    tabla_part = pd.DataFrame([P.describir(k, v) for k, v in conjuntos.items()])
    ctx.etapa(R.Etapa(
        6, "Preparación del modelo",
        texto=f"Partición:\n{tabla_part.to_string(index=False)}\n\n"
              f"Derivada: {derivada.parametros()}\n"
              f"dT/dt [°C/{derivada.unidad_tiempo}]: min {dTdt.min():.4g}, max {dTdt.max():.4g}\n\n"
              f"Normalización (con {list(conjuntos)[0]}):\n{esc.tabla().to_string()}",
        datos={"particion": tabla_part.to_dict("records"), "derivada": derivada.parametros(),
               "normalizacion": {"media": esc.media, "desviacion": esc.std}},
        tablas={"particion": tabla_part, "normalizacion": esc.tabla().reset_index(),
                "derivada": pd.DataFrame({"datetime_utc": ajuste["datetime_utc"].values,
                                          "Temperatura": ajuste["Temperatura"].values, "dTdt": dTdt})},
        graficas={"particion": partial(VT.grafica_particion, conjuntos=conjuntos),
                  "derivada": partial(VT.grafica_derivada, train=ajuste, dTdt=dTdt, unidad=derivada.unidad_tiempo),
                  "normalizacion": partial(VT.grafica_normalizacion, train=ajuste, escalador=esc)},
    ))

    # ---------- Etapa 7: identificación (estrategia C sobre A)
    estrategia = BarridoThreshold(
        cfg.thresholds,
        crear_base=lambda th: IdentificacionBase(th, derivada, cfg.alpha),
        simular=lambda res, d: simular_modelo(res.modelo, res.escalador, d, cfg.unidad_tiempo),
        aviso=lambda s: ctx.reg.info(7, s),
    )
    res = estrategia.ejecutar(ajuste, seleccion, nombre_sel, independiente)
    elegido_th = res.parametros["threshold_elegido"]
    ctx.etapa(R.Etapa(
        7, "Identificación",
        texto=f"{res.resumen()}\n\nModelo PySINDy:\n{texto_modelo(res.modelo)}\n"
              "(La constante física puede ser distinta de 0 aunque c0 = 0: aparece al\n"
              " des-normalizar T, L y H, que tienen media distinta de 0.)\n\n"
              f"{R.texto_barrido(res.detalles['barrido'])}\n\n"
              + ("Selección independiente de la validación." if independiente else
                 "AVISO: el threshold se eligió con la validación; sus métricas NO son una evaluación independiente."),
        datos={"estrategia": res.estrategia, "parametros": res.parametros, "coef": res.coef,
               "unidad_tiempo": res.unidad_tiempo, "terminos_activos": res.terminos_activos,
               "ecuacion": res.ecuacion(), "datos_identificacion": res.datos_identificacion,
               "datos_seleccion": res.datos_seleccion, "barrido": res.detalles["barrido"]},
        tablas={"coeficientes": tabla_coeficientes(res.coef, res.coef_normalizados),
                "barrido": pd.DataFrame(res.detalles["barrido"])},
        graficas={"barrido": partial(VM.grafica_barrido, tabla=res.detalles["barrido"],
                                     elegido=elegido_th, conjunto=nombre_sel),
                  "coeficientes": partial(VM.grafica_coeficientes, resultados={f"threshold {elegido_th:g}": res})},
    ))

    # ---------- Etapa 8: simulación y evaluación
    if independiente:
        pred = simular_modelo(res.modelo, res.escalador, val, cfg.unidad_tiempo)
    else:
        pred = res.detalles["pred_seleccion"]
    metricas = calcular_metricas(val["Temperatura"].values, pred)
    tray = pd.DataFrame({"datetime_utc": val["datetime_utc"].values,
                         "T_real": val["Temperatura"].values, "T_simulada": pred,
                         "error": pred - val["Temperatura"].values})
    aviso_ind = "" if independiente else "  (threshold elegido con estos mismos datos)"
    ctx.etapa(R.Etapa(
        8, "Simulación y evaluación",
        texto=f"Integrador: solve_ivp RK45 (variables normalizadas)\n"
              f"Validación: {val['datetime_utc'].iloc[0]} -> {val['datetime_utc'].iloc[-1]} ({len(val)} muestras)\n"
              f"{R.texto_metricas(metricas)}{aviso_ind}",
        datos={"metricas_validacion": metricas, "seleccion_independiente": independiente},
        tablas={"trayectoria_validacion": tray},
        graficas={"validacion": partial(VM.grafica_validacion_profesor, val=val, pred=pred, metricas=metricas,
                                        titulo_extra="" if independiente else " - threshold elegido con esta validación")},
    ))

    ctx.ejec.resultado.update({
        "estrategia": res.estrategia, "threshold": elegido_th, "coef": res.coef,
        "unidad_tiempo": res.unidad_tiempo, "ecuacion": res.ecuacion(), "metricas": metricas,
        "seleccion_independiente": independiente, "barrido": res.detalles["barrido"],
        "tramo": {"inicio": mejor["inicio"], "fin": mejor["fin"]},
        "conteos": {"filas_alineadas": len(datos), "bloques": len(bloques),
                    "tramo": int(mejor["n"]), "tramo_limpio": len(tramo),
                    "train": len(train), "validacion": len(val)},
    })
    ctx.ejec.memoria.update({"pred": pred, "resultado_modelo": res, "train": train, "val": val})
    texto = (f"BARRIDO DE THRESHOLD (invernadero_profesor.py)\n\nThreshold elegido: {elegido_th}\n"
             f"{res.ecuacion()}\n{constante_tiempo(res.coef, res.unidad_tiempo)}\n\n"
             f"Validación: {R.texto_metricas(metricas)}{aviso_ind}\n\n{R.texto_barrido(res.detalles['barrido'])}")
    return ctx.cerrar(texto)
