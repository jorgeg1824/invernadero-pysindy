"""
Gráficas de identificación, simulación y evaluación (etapas 7 y 8).

Origen:
  * grafica_validacion_profesor()  -> gráfica final de ejecutar_sindy()
                                      en invernadero_profesor.py
  * GRAFICAS_PREDICCION            -> las 7 gráficas de invernadero_20dias.py
                                      (mismo contenido; reciben `r` y `res`)
  * grafica_ventana / grafica_resumen_ventanas / grafica_error_por_dia_ventanas /
    grafica_real_vs_predicho_ventanas / grafica_errores_ventanas
                                   -> las de validar_modelo.py
  * grafica_barrido, grafica_coeficientes, grafica_estrategias -> nuevas.

`r` es el diccionario que arma pipeline.py con: train, val, pred, base,
metricas, metricas_base, mae_dias, dias_prediccion, max_hueco_horas, paso,
zona_horaria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluacion.metricas import metricas_archivo
from visualizacion.datos import _fechas


# ============================================================
# invernadero_profesor.py
# ============================================================

def grafica_validacion_profesor(fig, val: pd.DataFrame, pred, metricas: dict, titulo_extra: str = ""):
    ax = fig.add_subplot()
    ax.plot(val["datetime_utc"], val["Temperatura"].values, label="Temperatura real", lw=1.5)
    ax.plot(val["datetime_utc"], pred, label="Temperatura predicha", lw=1.5)
    ax.axvline(val["datetime_utc"].iloc[0], ls="--", lw=1)
    ax.set_title(f"Validación PySINDy  (RMSE {metricas['RMSE']:.3f} °C, R² {metricas['R2']:.3f}){titulo_extra}")
    ax.set_ylabel("Temperatura [°C]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _fechas(ax)


def grafica_barrido(fig, tabla: list, elegido: float, conjunto: str):
    ok = [t for t in tabla if "error" not in t]
    th = [t["threshold"] for t in ok]
    a1, a2 = fig.subplots(1, 2)
    a1.semilogx(th, [t["RMSE"] for t in ok], "o-")
    a1.axvline(elegido, color="C2", ls="--", label=f"Elegido: {elegido:g}")
    a1.set_xlabel("threshold (escala log)")
    a1.set_ylabel(f"RMSE en {conjunto} [°C]")
    a1.grid(alpha=0.3, which="both")
    a1.legend()
    a2.semilogx(th, [t["terminos"] for t in ok], "s-", color="C1")
    a2.set_xlabel("threshold (escala log)")
    a2.set_ylabel("Términos activos")
    a2.set_yticks(range(0, 5))
    a2.grid(alpha=0.3, which="both")
    fig.suptitle(f"Etapa 7 - Barrido de threshold (selección con: {conjunto})")


def grafica_coeficientes(fig, resultados: dict):
    """resultados: nombre -> ResultadoModelo. Barras de coeficientes normalizados."""
    ax = fig.add_subplot()
    nombres = ["constante", "T", "L", "H"]
    x = np.arange(4)
    ancho = 0.8 / max(len(resultados), 1)
    for i, (nombre, r) in enumerate(resultados.items()):
        ax.bar(x + (i - (len(resultados) - 1) / 2) * ancho, r.coef_normalizados, ancho, label=nombre)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, nombres)
    ax.set_ylabel("Coeficiente (variables normalizadas)")
    ax.set_title("Etapa 7 - Coeficientes de dTn/dt = c0 + cT·Tn + cL·Ln + cH·Hn")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()


def grafica_estrategias(fig, val: pd.DataFrame, preds: dict, metricas: dict):
    """Temperatura medida frente a la simulada por cada estrategia."""
    ax = fig.add_subplot()
    t = val.index if "datetime_utc" not in val.columns else val["datetime_utc"]
    ax.plot(t, val["Temperatura"].values, color="k", lw=1, label="Medida")
    for i, (nombre, p) in enumerate(preds.items()):
        if p is None:
            continue
        m = metricas[nombre]
        if np.isfinite(m["RMSE"]) and m["RMSE"] < 1e3:
            texto = f"RMSE {m['RMSE']:.2f} °C, R² {m['R2']:.3f}"
        else:
            texto = "diverge (modelo inestable)"
        ax.plot(t, np.clip(p, -10, 60), lw=0.9, alpha=0.85, color=f"C{i}", label=f"{nombre}: {texto}")
    ax.set_ylim(5, 55)
    ax.set_ylabel("Temperatura [°C]")
    ax.set_title("Comparación de estrategias sobre los mismos datos de validación")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    _fechas(ax)


# ============================================================
# invernadero_20dias.py
# ============================================================

def _por_tramos(ax, d, columna, max_hueco_horas, **kw):
    # Cada tramo se dibuja aparte para no unir con una línea los huecos sin datos.
    etiqueta = kw.pop("label", None)
    tramo = (d.index.to_series().diff() > pd.Timedelta(hours=max_hueco_horas)).cumsum()
    for i, (_, g) in enumerate(d.groupby(tramo)):
        ax.plot(g.index, g[columna], label=etiqueta if i == 0 else None, **kw)


def _sombrear_entrenamiento(ax, train):
    ax.axvspan(train.index[0], train.index[-1], color="green", alpha=0.08, label="Entrenamiento")


def grafica_prediccion(fig, r, res):
    ax = fig.add_subplot()
    val, train = r["val"], r["train"]
    ultimos = train.iloc[-int(pd.Timedelta(days=3) / r["paso"]):]
    ax.plot(ultimos.index, ultimos["Temperatura"], color="gray", lw=0.8, label="Entrenamiento (ultimos 3 dias)")
    ax.plot(val.index, val["Temperatura"], lw=1.0, label="Temperatura real")
    ax.plot(val.index, r["pred"], lw=1.0, label=f"Prediccion {r['dias_prediccion']:g} dias")
    ax.plot(val.index, r["base"], color="gray", lw=0.8, ls=":", label="Referencia (perfil diario)")
    ax.axvline(val.index[0], ls="--", color="k", lw=1)
    ax.set_title(f"Prediccion de {r['dias_prediccion']:g} dias  (MAE = {r['metricas']['MAE']:.2f} C, "
                 f"R2 = {r['metricas']['R2']:.3f})  [{r['estrategia']}]")
    ax.set_ylabel("Temperatura [C]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    _fechas(ax)


def grafica_archivo(fig, r, res):
    ax = fig.add_subplot()
    total, _ = metricas_archivo(res)
    _sombrear_entrenamiento(ax, r["train"])
    _por_tramos(ax, res, "real", r["max_hueco_horas"], color="C0", lw=0.5, label="Temperatura real")
    _por_tramos(ax, res, "simulada", r["max_hueco_horas"], color="C1", lw=0.5, alpha=0.8, label="Ecuacion simulada")
    ax.set_title(f"Ecuacion simulada sobre todo el archivo  |  fuera del entrenamiento: "
                 f"MAE {total['MAE']:.2f} C, R2 {total['R2']:.3f}")
    ax.set_ylabel("Temperatura [C]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    _fechas(ax)


def grafica_por_mes(fig, r, res):
    _, por_mes = metricas_archivo(res)
    meses = sorted(res["mes"].unique())
    for i, mes in enumerate(meses, 1):
        ax = fig.add_subplot(len(meses), 1, i)
        d = res[res["mes"] == mes]
        _sombrear_entrenamiento(ax, r["train"])
        ax.set_xlim(d.index[0], d.index[-1])
        _por_tramos(ax, d, "real", r["max_hueco_horas"], color="C0", lw=0.7, label="Real")
        _por_tramos(ax, d, "simulada", r["max_hueco_horas"], color="C1", lw=0.7, label="Simulada")
        texto = f"MAE {por_mes[mes]['MAE']:.2f} C, R2 {por_mes[mes]['R2']:.3f}" if mes in por_mes else "(entrenamiento)"
        ax.set_title(f"{mes}   {texto}", fontsize=9)
        ax.set_ylabel("T [C]")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        if i == 1:
            ax.legend(loc="upper left", ncol=3, fontsize=8)


def grafica_metricas_mes(fig, r, res):
    _, por_mes = metricas_archivo(res)
    meses = list(por_mes)
    a1, a2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    a1.bar(meses, [por_mes[m]["MAE"] for m in meses], color="C1")
    a1.set_ylabel("MAE [C]  (menor es mejor)")
    a2.bar(meses, [por_mes[m]["R2"] for m in meses], color="C0")
    a2.set_ylabel("R2  (mayor es mejor)")
    a2.set_ylim(0.5, 1)
    for a in (a1, a2):
        a.set_xlabel(f"Mes ({r['zona_horaria']})")
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Error de la ecuacion por mes (sin contar el entrenamiento)")


def grafica_error_por_dia(fig, r, res):
    ax = fig.add_subplot()
    dias = np.arange(1, len(r["mae_dias"]) + 1)
    ax.bar(dias, r["mae_dias"], color="C1")
    ax.axhline(np.mean(r["mae_dias"]), color="k", ls="--", lw=1, label="Promedio")
    ax.set_xlabel("Dia de prediccion")
    ax.set_ylabel("MAE del dia [C]")
    ax.set_xticks(dias)
    ax.set_title("Error de cada dia de la prediccion")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()


def grafica_real_vs_simulado(fig, r, res):
    ax = fig.add_subplot()
    fuera = res[~res["entrenamiento"]]
    total, _ = metricas_archivo(res)
    ax.scatter(fuera["real"], fuera["simulada"], s=2, alpha=0.15)
    lim = [fuera[["real", "simulada"]].min().min(), fuera[["real", "simulada"]].max().max()]
    ax.plot(lim, lim, "k--", lw=1, label="Prediccion perfecta")
    ax.set_xlabel("Temperatura real [C]")
    ax.set_ylabel("Temperatura simulada [C]")
    ax.set_title(f"Todo el archivo sin el entrenamiento  (R2 = {total['R2']:.3f})")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()


def grafica_errores(fig, r, res):
    fuera = res[~res["entrenamiento"]]
    error = fuera["simulada"] - fuera["real"]
    a1, a2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    a1.hist(error, bins=80, color="C1")
    a1.axvline(0, color="k", lw=1)
    a1.set_xlabel("Error = simulado - real [C]")
    a1.set_ylabel("Cantidad de muestras")
    a1.set_title(f"Distribucion del error ({np.mean(np.abs(error) < 2) * 100:.0f}% dentro de +-2 C)")
    por_hora = error.abs().groupby(fuera.index.tz_convert(r["zona_horaria"]).hour).mean()
    a2.bar(por_hora.index, por_hora.values, color="C1")
    a2.set_xlabel(f"Hora del dia ({r['zona_horaria']})")
    a2.set_ylabel("MAE [C]")
    a2.set_title("Error segun la hora del dia")
    for a in (a1, a2):
        a.grid(alpha=0.3)


GRAFICAS_PREDICCION = {
    "1_prediccion": grafica_prediccion,
    "2_todo_el_archivo": grafica_archivo,
    "3_todo_el_archivo_por_mes": grafica_por_mes,
    "4_metricas_por_mes": grafica_metricas_mes,
    "5_error_por_dia_de_prediccion": grafica_error_por_dia,
    "6_real_vs_simulado": grafica_real_vs_simulado,
    "7_distribucion_del_error": grafica_errores,
}


# ============================================================
# validar_modelo.py
# ============================================================

def grafica_ventana(fig, r, i):
    ax = fig.add_subplot()
    val = r["val"]
    ax.plot(val.index, val["Temperatura"], color="C0", lw=1.0, label="Real")
    ax.plot(val.index, r["pred"], color="C1", lw=1.0, label="Modelo")
    ax.plot(val.index, r["base"], color="gray", lw=0.8, ls=":", label="Referencia (perfil diario)")
    m, mb = r["metricas"], r["metricas_base"]
    ax.set_title(f"Ventana {i}: {r['nombre']}  |  entrenado con {r['dias_train']:.0f} dias  |  "
                 f"Modelo: MAE {m['MAE']:.2f} C, R2 {m['R2']:.3f}  |  Referencia: MAE {mb['MAE']:.2f} C, R2 {mb['R2']:.3f}",
                 fontsize=10)
    ax.set_ylabel("Temperatura [C]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    _fechas(ax)


def grafica_resumen_ventanas(fig, resultados):
    nombres = [f"V{i}\n{r['nombre']}" for i, r in enumerate(resultados, 1)]
    x = np.arange(len(resultados))
    a1, a2 = fig.subplots(1, 2)
    a1.bar(x - 0.2, [r["metricas"]["MAE"] for r in resultados], 0.4, label="Modelo")
    a1.bar(x + 0.2, [r["metricas_base"]["MAE"] for r in resultados], 0.4, color="gray", label="Referencia")
    a1.set_ylabel("MAE [C]  (menor es mejor)")
    a2.bar(x - 0.2, [r["metricas"]["R2"] for r in resultados], 0.4, label="Modelo")
    a2.bar(x + 0.2, [r["metricas_base"]["R2"] for r in resultados], 0.4, color="gray", label="Referencia")
    a2.set_ylabel("R2  (mayor es mejor)")
    a2.set_ylim(0.5, 1)
    for a in (a1, a2):
        a.set_xticks(x, nombres, fontsize=8)
        a.grid(axis="y", alpha=0.3)
        a.legend()
    fig.suptitle("Prediccion en cada ventana: modelo vs referencia ingenua")


def grafica_error_por_dia_ventanas(fig, resultados, dias_prediccion):
    ax = fig.add_subplot()
    dias = np.arange(1, int(dias_prediccion) + 1)
    for i, r in enumerate(resultados, 1):
        ax.plot(dias, r["mae_dias"][:len(dias)], marker="o", ms=3, lw=1, alpha=0.7, label=f"V{i}")
    promedio = np.mean([r["mae_dias"][:len(dias)] for r in resultados], axis=0)
    ax.plot(dias, promedio, color="k", lw=2.5, label="Promedio")
    ax.set_xlabel("Dia de prediccion")
    ax.set_ylabel("MAE del dia [C]")
    ax.set_xticks(dias)
    ax.set_title("Error segun el dia de prediccion en cada ventana")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)


def _juntar(resultados):
    real = np.concatenate([r["val"]["Temperatura"].values for r in resultados])
    pred = np.concatenate([r["pred"] for r in resultados])
    return real, pred


def grafica_real_vs_predicho_ventanas(fig, resultados):
    real, pred = _juntar(resultados)
    ax = fig.add_subplot()
    ax.scatter(real, pred, s=2, alpha=0.15)
    lim = [min(real.min(), pred.min()), max(real.max(), pred.max())]
    ax.plot(lim, lim, "k--", lw=1, label="Prediccion perfecta")
    r2 = 1 - np.sum((real - pred) ** 2) / np.sum((real - real.mean()) ** 2)
    ax.set_xlabel("Temperatura real [C]")
    ax.set_ylabel("Temperatura predicha [C]")
    ax.set_title(f"Todas las ventanas juntas  (R2 = {r2:.3f})")
    ax.grid(alpha=0.3)
    ax.legend()


def grafica_errores_ventanas(fig, resultados, zona_horaria):
    real, pred = _juntar(resultados)
    error = pred - real
    hora = np.concatenate([r["val"].index.tz_convert(zona_horaria).hour for r in resultados])
    a1, a2 = fig.subplots(1, 2)
    a1.hist(error, bins=80, color="C1")
    a1.axvline(0, color="k", lw=1)
    a1.set_xlabel("Error = predicho - real [C]")
    a1.set_ylabel("Cantidad de muestras")
    a1.set_title(f"Distribucion del error  (el {np.mean(np.abs(error) < 2) * 100:.0f}% dentro de +-2 C)")
    por_hora = pd.Series(np.abs(error)).groupby(hora).mean()
    a2.bar(por_hora.index, por_hora.values, color="C1")
    a2.set_xlabel(f"Hora del dia ({zona_horaria})")
    a2.set_ylabel("MAE [C]")
    a2.set_title("Error segun la hora del dia")
    for a in (a1, a2):
        a.grid(alpha=0.3)
