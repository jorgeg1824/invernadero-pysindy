"""
Funcionalidad 'prediccion' = invernadero_20dias.py

Tramos rellenados en malla uniforme, últimos N días para predecir,
estrategia A (PySINDy) + B (ajuste por simulación) y simulación de
todo el archivo.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

import config as C
from analisis import particion as P
from analisis.seleccion_tramo import elegir_bloque
from datos import limpieza, segmentacion
from evaluacion import reportes as R
from evaluacion.metricas import calcular_metricas, mae_por_dia, metricas_archivo
from modelos.coeficientes import tabla_coeficientes
from modelos.estrategias.ajuste_simulacion import AjusteSimulacion
from modelos.estrategias.identificacion_base import (DerivadaMalla, IdentificacionBase)
from modelos.sindy import texto_modelo
from simulacion.integrador import simular_lsim
from simulacion.prediccion import perfil_diario, simular_archivo
from visualizacion import datos as VD
from visualizacion import modelos as VM
from visualizacion import transformaciones as VT

from flujos.comun import Contexto, etapas_datos, paso_malla


def preparar_bloques(datos, cfg: C.ConfigPrediccion):
    return segmentacion.bloques_rellenados(datos, cfg.max_hueco_horas, paso_malla(cfg))


def evaluar_prediccion(bloque, cfg: C.ConfigPrediccion) -> dict:
    """evaluar() de invernadero_20dias.py: identifica (A [+B]) y predice los últimos N días."""
    dt_h = paso_malla(cfg).total_seconds() / 3600
    train, val = P.dividir_ultimos_dias(bloque, cfg.dias_prediccion)

    base = IdentificacionBase(cfg.threshold, DerivadaMalla(cfg.ventana_savgol, cfg.polyorder_savgol, dt_h),
                              cfg.alpha).ejecutar(train)
    simular = lambda coef, d: simular_lsim(coef, d, dt_h)
    if cfg.ajustar_por_simulacion:
        final = AjusteSimulacion(simular).ejecutar(train, base=base)
    else:
        final = base

    real = val["Temperatura"].values
    with np.errstate(over="ignore", invalid="ignore"):
        pred = simular(final.coef, val)
    pred_sindy = simular(base.coef, val) if base.estable else None
    base_perfil = perfil_diario(train, val)

    return {
        "train": train, "val": val, "modelo": base.modelo,
        "coef_sindy": base.coef, "coef": final.coef,
        "coef_ajustado": final.coef if cfg.ajustar_por_simulacion else None,
        "resultado_base": base, "resultado_final": final, "estrategia": final.estrategia,
        "pred": pred, "pred_sindy": pred_sindy, "base": base_perfil,
        "metricas": calcular_metricas(real, pred),
        "metricas_base": calcular_metricas(real, base_perfil),
        "mae_dias": mae_por_dia(real, pred, int(pd.Timedelta(days=1) / paso_malla(cfg))),
        "dias_prediccion": cfg.dias_prediccion, "max_hueco_horas": cfg.max_hueco_horas,
        "paso": paso_malla(cfg), "zona_horaria": cfg.zona_horaria,
    }


def etapa4_relleno(ctx, datos, bloques, bloque, cfg):
    paso = paso_malla(cfg)
    info = limpieza.resumen_relleno(datos, bloques, paso)
    huecos = segmentacion.listar_huecos(datos, pd.Timedelta(hours=cfg.max_hueco_horas))
    tabla = segmentacion.tabla_bloques(bloques)
    ctx.reg.cantidades(4, f"Mediciones alineadas -> puntos de malla de {paso.total_seconds() / 60:g} min",
                       info["mediciones_originales"], info["puntos_malla"])
    ctx.reg.info(4, f"Puntos rellenados en huecos: {info['puntos_rellenados_en_huecos']}; "
                    f"mediciones fuera de tramos: {info['mediciones_fuera_de_tramos (descartadas)']}")
    ctx.etapa(R.Etapa(
        4, "Limpieza y relleno de huecos",
        texto=f"Huecos > {cfg.max_hueco_horas:g} h separan tramos; los huecos menores se rellenan "
              f"interpolando en el tiempo sobre una malla de {paso.total_seconds() / 60:g} min.\n\n"
              f"{R.texto_dict(info)}\n\nTramos:\n{tabla.to_string(index=False)}\n\n"
              f"Huecos que separan tramos:\n{huecos.to_string(index=False)}",
        datos={**info, "huecos": len(huecos), "tramos": len(bloques)},
        tablas={"huecos": huecos, "tramos": tabla},
        graficas={"huecos": partial(VD.grafica_huecos, datos=datos, bloques=bloques, huecos=huecos,
                                    umbral_txt=f"{cfg.max_hueco_horas:g} h"),
                  "relleno": partial(VD.grafica_relleno, datos=datos, bloque=bloque, paso=paso)},
    ))
    return tabla


def ejecutar_prediccion(ruta_csv=None, cfg: C.ConfigPrediccion | None = None, guardar=True,
                        carpeta=None, aviso=print, graficas=True) -> R.Ejecucion:
    cfg = cfg or C.ConfigPrediccion()
    ctx = Contexto("prediccion", cfg, guardar, carpeta, aviso, graficas)
    ruta_csv = ruta_csv or C.CSV_EJEMPLO
    datos, _ = etapas_datos(ctx, ruta_csv, cfg.tolerancia_alineacion)
    dt_h = paso_malla(cfg).total_seconds() / 3600
    por_dia = pd.Timedelta(days=1) / paso_malla(cfg)

    bloques = preparar_bloques(datos, cfg)
    bloque, motivo = elegir_bloque(bloques, cfg.dias_prediccion, cfg.dias_entrenamiento_min)
    tabla = etapa4_relleno(ctx, datos, bloques, bloque, cfg)

    # ---------- Etapa 5
    elegido = next(i for i, b in enumerate(bloques) if b is bloque)
    ctx.etapa(R.Etapa(
        5, "Selección del tramo",
        texto=f"{motivo}\n\nTramo: {bloque.index[0]} -> {bloque.index[-1]}",
        datos={"motivo": motivo, "inicio": bloque.index[0], "fin": bloque.index[-1], "muestras": len(bloque)},
        tablas={"candidatos": tabla, "tramo_elegido": bloque},
        graficas={"seleccion": partial(VD.grafica_seleccion_tramo, tabla=tabla, elegido=elegido,
                                       minimo_dias=cfg.dias_prediccion + cfg.dias_entrenamiento_min,
                                       criterio="más muestras")},
    ))

    # ---------- Etapas 6-7: identificación
    r = evaluar_prediccion(bloque, cfg)
    base, final = r["resultado_base"], r["resultado_final"]
    tabla_part = pd.DataFrame([P.describir("entrenamiento", r["train"]), P.describir("prediccion", r["val"])])
    ctx.etapa(R.Etapa(
        6, "Preparación del modelo",
        texto=f"Partición: los últimos {cfg.dias_prediccion:g} días se predicen; todo lo anterior entrena.\n"
              f"{tabla_part.to_string(index=False)}\n\nDerivada: {base.parametros}\n\n"
              f"Normalización (con el entrenamiento):\n{base.escalador.tabla().to_string()}",
        datos={"particion": tabla_part.to_dict("records"), "derivada": base.parametros,
               "normalizacion": {"media": base.escalador.media, "desviacion": base.escalador.std}},
        tablas={"particion": tabla_part, "normalizacion": base.escalador.tabla().reset_index(),
                "derivada": pd.DataFrame({"Temperatura": r["train"]["Temperatura"],
                                          "dTdt_C_por_h": base.detalles["dTdt"]})},
        graficas={"particion": partial(VT.grafica_particion, conjuntos={"entrenamiento": r["train"], "prediccion": r["val"]}),
                  "derivada": partial(VT.grafica_derivada, train=r["train"], dTdt=base.detalles["dTdt"], unidad="h"),
                  "normalizacion": partial(VT.grafica_normalizacion, train=r["train"], escalador=base.escalador)},
    ))

    tablas7 = {"coeficientes_base": tabla_coeficientes(base.coef, base.coef_normalizados)}
    texto7 = f"--- Estrategia A ---\n{base.resumen()}\n{texto_modelo(base.modelo)}"
    modelos_graf = {"A: PySINDy base": base}
    if cfg.ajustar_por_simulacion:
        tablas7["coeficientes_ajustados"] = tabla_coeficientes(final.coef, final.coef_normalizados)
        texto7 += f"\n\n--- Estrategia B ---\n{final.resumen()}\nDetalles: {final.detalles}"
        modelos_graf["B: ajuste por simulación"] = final
    ctx.etapa(R.Etapa(
        7, "Identificación", texto=texto7,
        datos={"base": {"coef_por_h": base.coef, "terminos": base.terminos_activos, "estable": base.estable,
                        "parametros": base.parametros},
               "final": {"estrategia": final.estrategia, "coef_por_h": final.coef, "terminos": final.terminos_activos,
                         "parametros": final.parametros, "detalles": final.detalles}},
        tablas=tablas7,
        graficas={"coeficientes": partial(VM.grafica_coeficientes, resultados=modelos_graf)},
    ))

    # ---------- Etapa 8: simulación y evaluación
    res = simular_archivo(bloques, r["coef"], r["train"], dt_h, cfg.zona_horaria)
    total, por_mes = metricas_archivo(res)
    texto = R.resumen_prediccion(r, cfg.dias_prediccion) + "\n\n" + R.resumen_archivo(res, por_dia)
    ctx.etapa(R.Etapa(
        8, "Simulación y evaluación", texto=texto,
        datos={"metricas_prediccion": r["metricas"], "metricas_referencia": r["metricas_base"],
               "mae_por_dia": r["mae_dias"], "metricas_archivo_sin_train": total, "metricas_por_mes": por_mes},
        tablas={"trayectoria_prediccion": pd.DataFrame({"T_real": r["val"]["Temperatura"], "T_simulada": r["pred"],
                                                        "T_referencia_perfil": r["base"]}),
                "trayectoria_archivo": res},
        graficas={k: partial(f, r=r, res=res) for k, f in VM.GRAFICAS_PREDICCION.items()},
    ))
    ctx.ejec.resultado.update({
        "estrategia": r["estrategia"], "coef_sindy_por_h": r["coef_sindy"], "coef_por_h": r["coef"],
        "metricas": r["metricas"], "metricas_referencia": r["metricas_base"], "mae_dias": r["mae_dias"],
        "metricas_archivo": total, "metricas_por_mes": por_mes,
        "tramo": {"inicio": bloque.index[0], "fin": bloque.index[-1]},
        "conteos": {"filas_alineadas": len(datos), "bloques": len(bloques), "tramo": len(bloque),
                    "train": len(r["train"]), "validacion": len(r["val"])},
    })
    ctx.ejec.memoria.update({"pred": r["pred"], "r": r, "archivo": res})
    return ctx.cerrar("PREDICCION (invernadero_20dias.py)\n\n" + texto)
