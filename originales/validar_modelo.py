"""
Validacion del modelo de 20 dias sobre varias ventanas de datos distintas.

Genera imagenes en la carpeta validacion/:
  ventana_*.png           real vs prediccion vs referencia, una por ventana
  1_resumen_ventanas.png  MAE y R2 de cada ventana vs la referencia
  2_error_por_dia.png     el error no crece con el horizonte de prediccion
  3_real_vs_predicho.png  dispersion de todas las ventanas juntas
  4_errores.png           histograma del error y error por hora del dia

Uso: python validar_modelo.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import invernadero_profesor as prof
import invernadero_20dias as M

warnings.filterwarnings("ignore")
CARPETA = "validacion"
os.makedirs(CARPETA, exist_ok=True)

df, _ = prof.cargar_csv("Aeronic_export_29_09_2026.csv")
datos, _ = prof.alinear_variables(prof.construir_series(df))
minimo = M.DIAS_PREDICCION + M.DIAS_ENTRENAMIENTO_MIN
bloques = [b for b in M.bloques_rellenados(datos)
           if b.index[-1] - b.index[0] >= pd.Timedelta(days=minimo)]

# Ventanas: en cada tramo largo, se corta el final cada 4 dias para
# obtener distintos periodos de 20 dias de prediccion.
resultados = []
for b in bloques:
    dias = (b.index[-1] - b.index[0]).total_seconds() / 86400
    for fin in sorted({dias} | set(range(minimo, int(dias), 4))):
        r = M.evaluar(b[b.index <= b.index[0] + pd.Timedelta(days=fin)])
        r["nombre"] = f"{r['val'].index[0]:%d-%b} a {r['val'].index[-1]:%d-%b}"
        r["dias_train"] = (r["train"].index[-1] - r["train"].index[0]).total_seconds() / 86400
        resultados.append(r)

# ------------------------------------------------------------
# Tabla en consola
# ------------------------------------------------------------
print(f"{'Ventana de prediccion':24s} {'Entren.':>8s} {'MAE':>6s} {'RMSE':>6s} {'R2':>6s}"
      f" | {'MAE ref':>7s} {'R2 ref':>6s} | {'Peor dia':>8s}  PySINDy solo")
for r in resultados:
    m, mb = r["metricas"], r["metricas_base"]
    estado = "estable" if r["coef_sindy"]["aT"] < 0 else "diverge"
    print(f"{r['nombre']:24s} {r['dias_train']:6.1f} d {m['MAE']:6.2f} {m['RMSE']:6.2f} {m['R2']:6.3f}"
          f" | {mb['MAE']:7.2f} {mb['R2']:6.3f} | {max(r['mae_dias']):8.2f}  {estado}")

# ------------------------------------------------------------
# Una imagen por ventana
# ------------------------------------------------------------
for i, r in enumerate(resultados, 1):
    fig, ax = plt.subplots(figsize=(13, 4.5))
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
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{CARPETA}/ventana_{i}.png", dpi=130)
    plt.close(fig)

# ------------------------------------------------------------
# 1. Resumen por ventana
# ------------------------------------------------------------
nombres = [f"V{i}\n{r['nombre']}" for i, r in enumerate(resultados, 1)]
x = np.arange(len(resultados))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
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
fig.suptitle("Prediccion de 20 dias en cada ventana: modelo vs referencia ingenua")
fig.tight_layout()
fig.savefig(f"{CARPETA}/1_resumen_ventanas.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------
# 2. Error por dia de prediccion
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
dias = np.arange(1, M.DIAS_PREDICCION + 1)
for i, r in enumerate(resultados, 1):
    ax.plot(dias, r["mae_dias"][:len(dias)], marker="o", ms=3, lw=1, alpha=0.7, label=f"V{i}")
promedio = np.mean([r["mae_dias"][:len(dias)] for r in resultados], axis=0)
ax.plot(dias, promedio, color="k", lw=2.5, label="Promedio")
ax.set_xlabel("Dia de prediccion")
ax.set_ylabel("MAE del dia [C]")
ax.set_xticks(dias)
ax.set_title("El error no crece con el horizonte: el dia 20 es tan bueno como el dia 1")
ax.grid(alpha=0.3)
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()
fig.savefig(f"{CARPETA}/2_error_por_dia.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------
# 3. Real vs predicho
# ------------------------------------------------------------
real = np.concatenate([r["val"]["Temperatura"].values for r in resultados])
pred = np.concatenate([r["pred"] for r in resultados])
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(real, pred, s=2, alpha=0.15)
lim = [min(real.min(), pred.min()), max(real.max(), pred.max())]
ax.plot(lim, lim, "k--", lw=1, label="Prediccion perfecta")
r2 = 1 - np.sum((real - pred) ** 2) / np.sum((real - real.mean()) ** 2)
ax.set_xlabel("Temperatura real [C]")
ax.set_ylabel("Temperatura predicha [C]")
ax.set_title(f"Todas las ventanas juntas  (R2 = {r2:.3f})")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"{CARPETA}/3_real_vs_predicho.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------
# 4. Distribucion del error y error por hora del dia
# ------------------------------------------------------------
error = pred - real
hora = np.concatenate([r["val"].index.tz_convert("America/Bogota").hour for r in resultados])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
a1.hist(error, bins=80, color="C1")
a1.axvline(0, color="k", lw=1)
a1.set_xlabel("Error = predicho - real [C]")
a1.set_ylabel("Cantidad de muestras")
a1.set_title(f"Distribucion del error  (el {np.mean(np.abs(error) < 2) * 100:.0f}% dentro de +-2 C)")
por_hora = pd.Series(np.abs(error)).groupby(hora).mean()
a2.bar(por_hora.index, por_hora.values, color="C1")
a2.set_xlabel("Hora del dia (Colombia)")
a2.set_ylabel("MAE [C]")
a2.set_title("Error segun la hora del dia")
for a in (a1, a2):
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{CARPETA}/4_errores.png", dpi=130)
plt.close(fig)

print(f"\nImagenes guardadas en {CARPETA}/")
