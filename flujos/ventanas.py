"""
Funcionalidad 'ventanas' = validar_modelo.py

Repite la predicción de 'prediccion' en varias ventanas de cada tramo largo.
"""

from __future__ import annotations

from functools import partial

import pandas as pd

import config as C
from analisis import particion as P
from evaluacion import reportes as R
from visualizacion import modelos as VM

from flujos.comun import Contexto, etapas_datos
from flujos.prediccion import etapa4_relleno, evaluar_prediccion, preparar_bloques


def ejecutar_ventanas(ruta_csv=None, cfg: C.ConfigVentanas | None = None, guardar=True,
                      carpeta=None, aviso=print, graficas=True) -> R.Ejecucion:
    cfg = cfg or C.ConfigVentanas()
    cp = cfg.prediccion
    ctx = Contexto("ventanas", cfg, guardar, carpeta, aviso, graficas)
    ruta_csv = ruta_csv or C.CSV_EJEMPLO
    datos, _ = etapas_datos(ctx, ruta_csv, cp.tolerancia_alineacion)

    minimo = cp.dias_prediccion + cp.dias_entrenamiento_min
    todos = preparar_bloques(datos, cp)
    bloques = [b for b in todos if b.index[-1] - b.index[0] >= pd.Timedelta(days=minimo)]
    etapa4_relleno(ctx, datos, todos, bloques[0] if bloques else todos[0], cp)

    cortes = [(b, fin) for b in bloques for fin in P.cortes_ventanas(b, minimo, cfg.paso_ventanas_dias)]
    tabla_v = pd.DataFrame([{"tramo_inicio": b.index[0], "dias_desde_inicio": fin} for b, fin in cortes])
    ctx.etapa(R.Etapa(
        5, "Selección de tramos y ventanas",
        texto=f"Se usan todos los tramos de al menos {minimo:g} días ({len(bloques)}). En cada uno se recorta "
              f"el final cada {cfg.paso_ventanas_dias} días: {len(cortes)} ventanas.\n"
              "Las ventanas de un mismo tramo se solapan y comparten parte del entrenamiento.\n\n"
              f"{tabla_v.to_string(index=False)}",
        datos={"tramos_usados": len(bloques), "ventanas": len(cortes)},
        tablas={"ventanas": tabla_v},
    ))

    resultados = []
    for k, (b, fin) in enumerate(cortes, 1):
        ctx.aviso(f"Ventana {k}/{len(cortes)}")
        r = evaluar_prediccion(P.recortar(b, fin), cp)
        r["nombre"] = f"{r['val'].index[0]:%d-%b} a {r['val'].index[-1]:%d-%b}"
        r["dias_train"] = (r["train"].index[-1] - r["train"].index[0]).total_seconds() / 86400
        resultados.append(r)

    filas = [{"ventana": f"V{i}", "nombre": r["nombre"], **P.describir("entrenamiento", r["train"]),
              "pred_inicio": r["val"].index[0], "pred_fin": r["val"].index[-1]}
             for i, r in enumerate(resultados, 1)]
    ctx.etapa(R.Etapa(6, "Preparación del modelo (por ventana)",
                      texto=pd.DataFrame(filas).to_string(index=False),
                      tablas={"particiones": pd.DataFrame(filas)}))
    coefs = pd.DataFrame([{"ventana": f"V{i}", **{f"sindy_{k}": v for k, v in r["coef_sindy"].items()},
                           "sindy_estable": r["coef_sindy"]["aT"] < 0,
                           **{f"final_{k}": v for k, v in r["coef"].items()}}
                          for i, r in enumerate(resultados, 1)])
    ctx.etapa(R.Etapa(7, "Identificación (por ventana)",
                      texto=f"Estrategia: {resultados[0]['estrategia']}\nCoeficientes por hora:\n{coefs.to_string(index=False)}",
                      tablas={"coeficientes": coefs}))

    texto = R.tabla_ventanas(resultados)
    graf = {f"ventana_{i}": partial(VM.grafica_ventana, r=r, i=i) for i, r in enumerate(resultados, 1)}
    graf.update({
        "1_resumen_ventanas": partial(VM.grafica_resumen_ventanas, resultados=resultados),
        "2_error_por_dia": partial(VM.grafica_error_por_dia_ventanas, resultados=resultados,
                                   dias_prediccion=cp.dias_prediccion),
        "3_real_vs_predicho": partial(VM.grafica_real_vs_predicho_ventanas, resultados=resultados),
        "4_errores": partial(VM.grafica_errores_ventanas, resultados=resultados, zona_horaria=cp.zona_horaria),
    })
    metr = pd.DataFrame([{"ventana": f"V{i}", "nombre": r["nombre"], "dias_train": r["dias_train"],
                          **r["metricas"], **{f"ref_{k}": v for k, v in r["metricas_base"].items()},
                          "peor_dia_MAE": max(r["mae_dias"])} for i, r in enumerate(resultados, 1)])
    ctx.etapa(R.Etapa(8, "Simulación y evaluación (por ventana)", texto=texto,
                      tablas={"metricas_ventanas": metr}, graficas=graf))
    ctx.ejec.resultado.update({"ventanas": metr.to_dict("records")})
    ctx.ejec.memoria.update({"resultados": resultados})
    return ctx.cerrar("VALIDACION EN VENTANAS (validar_modelo.py)\n\n" + texto)
