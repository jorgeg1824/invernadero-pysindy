"""
Gráficas de las etapas de datos (1 a 5): CSV original, conversión de
fechas, alineación, limpieza/huecos y selección del tramo.

Todas reciben una figura de matplotlib (`fig`) y dibujan en ella, para que
la misma función sirva para la interfaz gráfica y para guardar PNG.
Nuevas (los originales no tenían gráficas por etapa), salvo
grafica_variables(), que replica actualizar_grafica() de la interfaz del
profesor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import UNIDADES, VARIABLES

EJE_T = "Tiempo (UTC)"
COLORES = {"Temperatura": "C3", "Luminosidad": "C1", "Humedad": "C0"}


def _fechas(ax):
    ax.set_xlabel(EJE_T)
    for etiqueta in ax.get_xticklabels():
        etiqueta.set_rotation(20)
        etiqueta.set_ha("right")


def _ylabel(v):
    return f"{v} [{UNIDADES[v]}]"


# ------------------------------------------------------------
# Etapa 1: datos originales
# ------------------------------------------------------------

def grafica_registros_por_metrica(fig, df_crudo: pd.DataFrame):
    ax = fig.add_subplot()
    conteo = df_crudo["metric_name"].value_counts()
    colores = ["C2" if m in VARIABLES else "0.6" for m in conteo.index]
    ax.barh(conteo.index, conteo.values, color=colores)
    for i, v in enumerate(conteo.values):
        ax.text(v, i, f" {v}", va="center")
    ax.set_xlabel("Cantidad de filas")
    ax.set_title("Filas del CSV por métrica (verde = usadas por el modelo, gris = ignoradas)")
    ax.grid(axis="x", alpha=0.3)


def grafica_series_originales(fig, series: dict):
    """Cada variable con sus marcas de tiempo originales, sin alinear."""
    ejes = fig.subplots(3, 1, sharex=True)
    for ax, v in zip(ejes, VARIABLES):
        s = series[v]
        ax.plot(s["datetime_utc"], s[v], lw=0.4, color=COLORES[v])
        ax.set_ylabel(_ylabel(v))
        ax.set_title(f"{v}: {len(s)} mediciones", fontsize=10)
        ax.grid(alpha=0.3)
    _fechas(ejes[-1])
    fig.suptitle("Etapa 1 - Series temporales originales (cada variable con sus propias marcas de tiempo)")


# ------------------------------------------------------------
# Etapa 2: transformación
# ------------------------------------------------------------

def grafica_conversion_fechas(fig, df_crudo: pd.DataFrame, df_conv: pd.DataFrame):
    """Antes: valores en el orden de las filas. Después: valores sobre el eje de fechas."""
    m = df_crudo["metric_name"] == "Temperatura"
    a1, a2 = fig.subplots(2, 1)
    a1.plot(np.flatnonzero(m), pd.to_numeric(df_crudo.loc[m, "number_value"], errors="coerce"),
            lw=0.4, color="C3")
    a1.set_xlabel("Número de fila en el CSV")
    a1.set_ylabel("Temperatura [°C]")
    a1.set_title("Antes de convertir: 'datetime_utc' es texto; solo existe el orden de las filas", fontsize=10)
    a1.grid(alpha=0.3)

    t = df_conv.loc[m, "datetime_utc"]
    a2.plot(t, pd.to_numeric(df_conv.loc[m, "number_value"], errors="coerce"), lw=0.4, color="C3")
    retrocesos = int((t.diff().dt.total_seconds() < 0).sum())
    a2.set_ylabel("Temperatura [°C]")
    a2.set_title(f"Después de convertir a fecha UTC  |  saltos hacia atrás en el tiempo: {retrocesos}",
                 fontsize=10)
    a2.grid(alpha=0.3)
    _fechas(a2)
    fig.suptitle("Etapa 2 - Conversión de fechas")


def grafica_construccion_series(fig, resumen: dict):
    ax = fig.add_subplot()
    etiquetas = ["Filas en el CSV", "Sin fecha o valor", "Métricas ignoradas"] + \
        [f"Serie {v}" for v in resumen["muestras_por_variable"]]
    valores = [resumen["filas_entrada"], resumen["filas_sin_fecha_o_valor"],
               sum(resumen["metricas_ignoradas"].values())] + list(resumen["muestras_por_variable"].values())
    colores = ["0.4", "C3", "0.7"] + ["C2"] * len(resumen["muestras_por_variable"])
    ax.barh(etiquetas[::-1], valores[::-1], color=colores[::-1])
    for i, v in enumerate(valores[::-1]):
        ax.text(v, i, f" {v}", va="center")
    ax.set_xlabel("Registros")
    ax.set_title("Etapa 2 - Formato largo -> una serie por variable: registros que entran, se descartan y quedan")
    ax.grid(axis="x", alpha=0.3)


# ------------------------------------------------------------
# Etapa 3: alineación
# ------------------------------------------------------------

def grafica_desfases(fig, difs: pd.DataFrame, tolerancia_s: float):
    ejes = fig.subplots(1, 2)
    for ax, v in zip(ejes, ["Luminosidad", "Humedad"]):
        d = difs[f"desfase_{v}_s"].dropna()
        dentro = d.abs() <= tolerancia_s
        ax.hist(d[dentro], bins=np.linspace(-tolerancia_s, tolerancia_s, 61), color=COLORES[v])
        ax.set_yscale("log")
        for x in (-tolerancia_s, tolerancia_s):
            ax.axvline(x, color="k", ls="--", lw=1)
        ax.set_xlabel(f"Desfase {v} - Temperatura [s]")
        ax.set_ylabel("Instantes de T (escala log)")
        ax.set_title(f"{v}: {(d == 0).mean() * 100:.2f}% exactamente a 0 s; "
                     f"{dentro.mean() * 100:.2f}% dentro de ±{tolerancia_s:.0f} s; {(~dentro).sum()} fuera", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Etapa 3 - Diferencia temporal entre cada medición de T y la L / H más cercana")


def grafica_variables(fig, datos: pd.DataFrame, titulo: str = "Datos alineados"):
    """Las tres variables sobre la misma marca de tiempo."""
    ejes = fig.subplots(3, 1, sharex=True)
    for ax, v in zip(ejes, VARIABLES):
        ax.plot(datos["datetime_utc"], datos[v], lw=0.4, color=COLORES[v])
        ax.set_ylabel(_ylabel(v))
        ax.grid(alpha=0.3)
    _fechas(ejes[-1])
    fig.suptitle(titulo)


# ------------------------------------------------------------
# Etapa 4: limpieza y huecos
# ------------------------------------------------------------

def grafica_huecos(fig, datos: pd.DataFrame, bloques: list, huecos: pd.DataFrame, umbral_txt: str):
    """Temperatura completa con los huecos y los tramos continuos marcados."""
    ax = fig.add_subplot()
    ax.plot(datos["datetime_utc"], datos["Temperatura"], lw=0.3, color="0.5", label="Temperatura medida")
    for i, b in enumerate(bloques):
        ini, fin = (b["inicio"], b["fin"]) if isinstance(b, dict) else (b.index[0], b.index[-1])
        ax.axvspan(ini, fin, color=f"C{i % 10}", alpha=0.12, label="Tramos continuos" if i == 0 else None)
    for j, h in huecos.iterrows():
        ax.axvspan(h["inicio"], h["fin"], color="red", alpha=0.6,
                   label=f"Huecos > {umbral_txt}" if j == 0 else None)
    ax.set_ylabel("Temperatura [°C]")
    ax.set_title(f"Etapa 4 - {len(bloques)} tramos continuos y {len(huecos)} huecos > {umbral_txt}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    _fechas(ax)


def grafica_relleno(fig, datos: pd.DataFrame, bloque: pd.DataFrame, paso: pd.Timedelta):
    """Acercamiento al mayor hueco rellenado dentro del tramo elegido."""
    orig = datos.set_index("datetime_utc")["Temperatura"]
    orig = orig[(orig.index >= bloque.index[0]) & (orig.index <= bloque.index[-1])]
    dt = orig.index.to_series().diff()
    centro = dt.idxmax() if len(dt.dropna()) else bloque.index[len(bloque) // 2]
    ventana = pd.Timedelta(hours=6)
    o = orig[(orig.index > centro - ventana) & (orig.index < centro + ventana)]
    b = bloque[(bloque.index > centro - ventana) & (bloque.index < centro + ventana)]
    from datos.limpieza import puntos_en_huecos
    hueco = puntos_en_huecos(datos, b, paso)

    ax = fig.add_subplot()
    ax.plot(b.index, b["Temperatura"], color="C0", lw=1, label=f"Malla uniforme ({paso.total_seconds() / 60:.0f} min)")
    ax.plot(o.index, o.values, "o", ms=4, color="k", label="Mediciones originales")
    ax.plot(b.index[hueco], b["Temperatura"][hueco], "x", ms=6, color="C3",
            label=f"Rellenados en el hueco (sin medición a ±{paso.total_seconds() / 60:.0f} min)")
    ax.set_ylabel("Temperatura [°C]")
    ax.set_title(f"Etapa 4 - Relleno del mayor hueco del tramo ({dt.max()} sin datos)")
    ax.grid(alpha=0.3)
    ax.legend()
    _fechas(ax)


# ------------------------------------------------------------
# Etapa 5: selección del tramo
# ------------------------------------------------------------

def grafica_seleccion_tramo(fig, tabla: pd.DataFrame, elegido: int, minimo_dias: float, criterio: str):
    ax = fig.add_subplot()
    colores = ["C2" if i == elegido else ("C0" if d >= minimo_dias else "0.75")
               for i, d in zip(tabla["bloque"], tabla["dias"])]
    ax.barh(tabla["bloque"].astype(str), tabla["dias"], color=colores)
    ax.axvline(minimo_dias, color="k", ls="--", lw=1, label=f"Mínimo exigido: {minimo_dias:g} días")
    for i, (d, ini) in enumerate(zip(tabla["dias"], tabla["inicio"])):
        ax.text(d, i, f" {d:.1f} d  ({pd.Timestamp(ini):%Y-%m-%d})", va="center", fontsize=8)
    ax.set_xlabel("Duración del tramo [días]")
    ax.set_ylabel("Tramo")
    ax.set_title(f"Etapa 5 - Tramos candidatos (verde = elegido; criterio: {criterio})")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
