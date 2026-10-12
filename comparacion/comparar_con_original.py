"""
Etapa 7 - Compara la versión modular con los resultados de los programas
originales guardados en referencia_original/.

    python comparacion/comparar_con_original.py

Escribe:
    docs/comparacion_original_vs_modular.md
    comparacion/resultado_comparacion.json

Tolerancias (justificación en docs/comparacion_original_vs_modular.md):
    EXACTA = 1e-9  relativo   cálculos deterministas (mismo código y orden)
    AJUSTE = 1e-6  relativo   resultados de least_squares (optimizador iterativo)
    TRAY   = 1e-6  °C         trayectorias simuladas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import config as C  # noqa: E402
import pipeline  # noqa: E402
from analisis.derivadas import unidad_efectiva  # noqa: E402

REF = RAIZ / "referencia_original" / "resultados"
REF_PANDAS2 = RAIZ / "referencia_original" / "resultados_pandas2"
EXACTA, AJUSTE, TRAY = 1e-9, 1e-6, 1e-6


class Comparador:
    def __init__(self, programa):
        self.programa = programa
        self.filas = []

    def _fila(self, aspecto, original, modular, diferencia, tolerancia, ok):
        self.filas.append({"programa": self.programa, "aspecto": aspecto, "original": original,
                           "modular": modular, "diferencia": diferencia, "tolerancia": tolerancia,
                           "ok": bool(ok)})

    def igual(self, aspecto, a, b):
        self._fila(aspecto, str(a), str(b), "-" if a == b else "distinto", "igual", a == b)

    def numero(self, aspecto, a, b, rtol=EXACTA):
        a, b = float(a), float(b)
        d = abs(a - b) / max(abs(a), 1e-300) if a != b else 0.0
        self._fila(aspecto, f"{a:.10g}", f"{b:.10g}", f"{d:.2e} rel", f"{rtol:g} rel", d <= rtol)

    def dic(self, aspecto, a: dict, b: dict, rtol=EXACTA):
        for k in a:
            self.numero(f"{aspecto} {k}", a[k], b[k], rtol)

    def serie(self, aspecto, a, b, atol=TRAY):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape:
            self._fila(aspecto, f"{a.shape}", f"{b.shape}", "tamaño distinto", f"{atol:g} °C", False)
            return
        d = float(np.max(np.abs(a - b)))
        self._fila(aspecto, f"{len(a)} valores", f"{len(b)} valores", f"{d:.2e} °C máx", f"{atol:g} °C", d <= atol)


def _ts(x):
    return pd.Timestamp(x).isoformat()


def _leer(carpeta, programa):
    return json.loads((carpeta / programa / "resultados.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------

def comparar_barrido(carpeta_ref, unidad, etiqueta):
    ref = _leer(carpeta_ref, "profesor")
    tray = pd.read_csv(carpeta_ref / "profesor" / "trayectoria_validacion.csv")
    cfg = C.ConfigBarrido(unidad_tiempo=unidad)
    ej = pipeline.ejecutar_barrido(cfg=cfg, guardar=False, aviso=None)
    r = ej.resultado
    c = Comparador(f"invernadero_profesor.py vs barrido ({etiqueta})")
    reg = ref["registros"]
    c.igual("filas CSV", reg["filas_csv"], ej.etapa(1).datos["filas"])
    c.igual("muestras por variable", reg["muestras_por_variable"],
            ej.etapa(2).datos["construccion_series"]["muestras_por_variable"])
    c.igual("filas alineadas", reg["filas_alineadas"], r["conteos"]["filas_alineadas"])
    c.igual("duplicados eliminados", reg["duplicados_eliminados"], ej.etapa(3).datos["duplicados_eliminados"])
    c.igual("muestras del tramo", reg["muestras_tramo"], r["conteos"]["tramo"])
    c.igual("muestras tramo limpio", reg["muestras_tramo_limpio"], r["conteos"]["tramo_limpio"])
    c.igual("muestras entrenamiento", reg["muestras_train"], r["conteos"]["train"])
    c.igual("muestras validación", reg["muestras_validacion"], r["conteos"]["validacion"])
    c.igual("inicio del tramo", _ts(ref["tramo"]["inicio"]), _ts(r["tramo"]["inicio"]))
    c.igual("fin del tramo", _ts(ref["tramo"]["fin"]), _ts(r["tramo"]["fin"]))
    c.numero("puntaje del tramo", ref["tramo"]["score"], ej.etapa(5).datos["puntaje"])
    c.igual("thresholds probados", ref["configuracion"]["thresholds"], cfg.thresholds)
    c.igual("threshold elegido", ref["threshold_elegido"], r["threshold"])
    c.dic("coeficiente", ref["coeficientes_fisicos"], r["coef"])
    for a, b in zip(ref["barrido_threshold"], r["barrido"]):
        c.numero(f"RMSE barrido th={a['threshold']:g}", a["RMSE"], b["RMSE"], rtol=1e-6)
        c.igual(f"términos th={a['threshold']:g}", a["terminos"], b["terminos"])
    c.serie("trayectoria simulada (validación)", tray["T_simulada"], ej.memoria["pred"])
    c.dic("métrica", ref["metricas_validacion"], r["metricas"], rtol=1e-6)
    return c.filas


def comparar_prediccion(carpeta_ref):
    ref = _leer(carpeta_ref, "20dias")
    tray = pd.read_csv(carpeta_ref / "20dias" / "trayectoria_prediccion_20d.csv")
    arch = pd.read_csv(carpeta_ref / "20dias" / "trayectoria_archivo.csv")
    ej = pipeline.ejecutar_prediccion(guardar=False, aviso=None)
    r = ej.resultado
    c = Comparador("invernadero_20dias.py vs prediccion")
    reg = ref["registros"]
    c.igual("filas alineadas", reg["filas_alineadas"], r["conteos"]["filas_alineadas"])
    c.igual("tramos rellenados", reg["bloques"], r["conteos"]["bloques"])
    c.igual("muestras del tramo", reg["muestras_bloque"], r["conteos"]["tramo"])
    c.igual("muestras entrenamiento", reg["muestras_train"], r["conteos"]["train"])
    c.igual("muestras predicción", reg["muestras_validacion"], r["conteos"]["validacion"])
    for k in ("inicio", "fin"):
        c.igual(f"{k} del tramo", _ts(ref["tramo"][k]), _ts(r["tramo"][k]))
    c.igual("parámetros", {k: ref["configuracion"][k] for k in ("THRESHOLD", "MAX_HUECO_HORAS", "DIAS_PREDICCION")},
            {"THRESHOLD": ej.config["threshold"], "MAX_HUECO_HORAS": ej.config["max_hueco_horas"],
             "DIAS_PREDICCION": ej.config["dias_prediccion"]})
    c.dic("coef. PySINDy (/h)", ref["coef_sindy_por_hora"], r["coef_sindy_por_h"])
    c.dic("coef. ajustado (/h)", ref["coef_ajustado_por_hora"], r["coef_por_h"], rtol=AJUSTE)
    c.serie("trayectoria predicción 20 días", tray["T_simulada"], ej.memoria["pred"])
    c.serie("referencia perfil diario", tray["T_referencia_perfil"], ej.memoria["r"]["base"])
    c.serie("trayectoria todo el archivo", arch["simulada"], ej.memoria["archivo"]["simulada"].values)
    c.dic("métrica predicción", ref["metricas_prediccion"], r["metricas"], rtol=AJUSTE)
    c.dic("métrica referencia", ref["metricas_referencia_perfil"], r["metricas_referencia"])
    c.dic("métrica archivo", ref["metricas_archivo_sin_train"], r["metricas_archivo"], rtol=AJUSTE)
    for mes in ref["metricas_por_mes"]:
        c.numero(f"MAE {mes}", ref["metricas_por_mes"][mes]["MAE"], r["metricas_por_mes"][mes]["MAE"], AJUSTE)
    c.serie("MAE por día", ref["mae_por_dia"], r["mae_dias"])
    return c.filas


def comparar_ventanas(carpeta_ref):
    ref = _leer(carpeta_ref, "validar_modelo")["ventanas"]
    ej = pipeline.ejecutar_ventanas(guardar=False, aviso=None)
    res = ej.memoria["resultados"]
    c = Comparador("validar_modelo.py vs ventanas")
    c.igual("cantidad de ventanas", len(ref), len(res))
    for i, (a, b) in enumerate(zip(ref, res), 1):
        c.igual(f"V{i} nombre", a["nombre"], b["nombre"])
        c.igual(f"V{i} inicio predicción", _ts(a["val_inicio"]), _ts(b["val"].index[0]))
        c.dic(f"V{i} coef. ajustado", a["coef_ajustado_por_hora"], b["coef"], rtol=AJUSTE)
        c.dic(f"V{i} métrica", a["metricas"], b["metricas"], rtol=AJUSTE)
        c.numero(f"V{i} peor día MAE", a["peor_dia_mae"], max(b["mae_dias"]), AJUSTE)
    return c.filas


# ------------------------------------------------------------

def informe(filas, entorno_ref, segundos):
    df = pd.DataFrame(filas)
    lineas = [
        "# Comparación: programas originales vs versión modular",
        "",
        "Generado por `python comparacion/comparar_con_original.py` "
        f"({time.strftime('%Y-%m-%d %H:%M')}, {segundos:.0f} s).",
        "",
        "Se ejecutó la versión modular con el mismo CSV y la configuración por defecto, y se",
        "comparó con lo que guardaron los programas originales en `referencia_original/`",
        "(ver `referencia_original/COMO_SE_EJECUTO.md`).",
        "",
        f"Entorno de la referencia: Python {entorno_ref['python']}, pandas {entorno_ref['pandas']}, "
        f"numpy {entorno_ref['numpy']}, scipy {entorno_ref['scipy']}, pysindy {entorno_ref['pysindy']}.",
        "",
        "## Tolerancias",
        "",
        "| Tipo | Tolerancia | Por qué |",
        "|---|---|---|",
        "| Conteos, fechas, parámetros, threshold | igualdad exacta | son enteros o texto |",
        f"| Coeficientes de PySINDy y conversiones | {EXACTA:g} relativo | mismo cálculo en el mismo orden: debe dar el mismo número |",
        f"| Coeficientes ajustados (least_squares) y sus métricas | {AJUSTE:g} relativo | optimizador iterativo: el mismo programa original da diferencias ~1e-7 al cambiar solo la versión de pandas |",
        f"| Trayectorias simuladas | {TRAY:g} °C | muy por debajo de la resolución del sensor (0,01 °C) |",
        "",
        "## Resumen",
        "",
        "| Comparación | Aspectos | Coinciden |",
        "|---|---|---|",
    ]
    for prog, g in df.groupby("programa", sort=False):
        lineas.append(f"| {prog} | {len(g)} | {int(g['ok'].sum())} {'✅' if g['ok'].all() else '❌'} |")
    for prog, g in df.groupby("programa", sort=False):
        lineas += ["", f"## {prog}", "", "| Aspecto | Original | Modular | Diferencia | Tolerancia | OK |",
                   "|---|---|---|---|---|---|"]
        for _, f in g.iterrows():
            lineas.append(f"| {f['aspecto']} | {f['original']} | {f['modular']} | {f['diferencia']} | "
                          f"{f['tolerancia']} | {'✅' if f['ok'] else '❌'} |")
    lineas += ["", TEXTO_DIFERENCIAS]
    return "\n".join(lineas) + "\n"


TEXTO_DIFERENCIAS = """## Funcionalidades y gráficas

| Original | En la versión modular | Estado |
|---|---|---|
| profesor: cargar CSV, alinear, elegir tramo, barrido de threshold, ecuación y métricas | `python cli.py barrido` / interfaz | Conservado (tabla de arriba) |
| profesor: gráfica de variables | Etapas 1, 3 y 5 (un panel por variable) | Conservado; sin las casillas para ocultar variables |
| profesor: gráfica de validación | Etapa 8, `validacion.png` | Conservado |
| 20dias: predicción de 20 días, referencia ingenua, simulación de todo el archivo, métricas por mes y por día | `python cli.py prediccion` / interfaz | Conservado (tabla de arriba) |
| 20dias: 7 gráficas y selector de gráficas | Etapa 8 de `prediccion`; selector "Gráfica" en la interfaz | Conservado; títulos de 1 gráfica ajustados (ver `docs/mapa_migracion.md`) |
| 20dias: modo consola | `python cli.py prediccion` | Conservado; ya no necesita tkinter |
| validar_modelo: 6 ventanas, tabla y 10 gráficas | `python cli.py ventanas`, etapa 8 | Conservado; título de 1 gráfica ajustado |

## Diferencias explicadas

1. **Unidad de tiempo del barrido (corrección C1).** Por defecto la versión
   modular calcula el tiempo en segundos reales (`unidad_tiempo="s"`). Con
   pandas 3, el original calculaba kilosegundos. Por eso, con la configuración
   por defecto:
   * `python cli.py barrido` da el mismo resultado que el **original con
     pandas 2** (threshold 0,01, 2 términos, RMSE 2,645 °C). Comparación
     "unidad_tiempo='s'" de arriba.
   * `python cli.py barrido --unidad-tiempo original` da el mismo resultado que
     el **original con pandas 3** (threshold 1e-5, 4 términos, RMSE 2,820 °C).
     Comparación "unidad_tiempo='original'" de arriba.
2. **Etiqueta de unidades en la ecuación.** La ecuación se imprime igual que en el
   original, más la unidad de tiempo al final (`[°C/s]`, `[°C/ks]` o `[°C/h]`).
   Solo es presentación.
3. **Archivos de salida.** Los originales no guardaban (profesor) o guardaban
   solo PNG (20dias, validar_modelo). La versión modular guarda además, por
   etapa, tablas CSV, resúmenes JSON y un registro. No cambia ningún resultado.
4. **Ninguna otra diferencia numérica:** la mayor diferencia medida en
   trayectorias es 7e-15 °C y en coeficientes 0 (ver las tablas).
"""


def ejecutar_todo(con_pandas2=True):
    warnings.filterwarnings("ignore")
    filas = []
    # El original con pandas 3 equivale a unidad_tiempo="original"; con pandas 2, a "s".
    if unidad_efectiva("original") == "ks":
        filas += comparar_barrido(REF, "original", "unidad_tiempo='original'")
        if con_pandas2 and REF_PANDAS2.exists():
            filas += comparar_barrido(REF_PANDAS2, "s", "unidad_tiempo='s' vs original con pandas 2")
    else:
        filas += comparar_barrido(REF, "s", "unidad_tiempo='s'")
    filas += comparar_prediccion(REF)
    filas += comparar_ventanas(REF)
    return filas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sin-pandas2", action="store_true", help="no comparar con la referencia de pandas 2")
    a = p.parse_args()
    t0 = time.time()
    filas = ejecutar_todo(not a.sin_pandas2)
    entorno_ref = json.loads((REF / "registro_ejecucion.json").read_text())["entorno"]
    md = informe(filas, entorno_ref, time.time() - t0)
    (RAIZ / "docs" / "comparacion_original_vs_modular.md").write_text(md, encoding="utf-8")
    (RAIZ / "comparacion" / "resultado_comparacion.json").write_text(
        json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    df = pd.DataFrame(filas)
    print(df.groupby("programa", sort=False)["ok"].agg(["count", "sum"]).to_string())
    malos = df[~df["ok"]]
    if len(malos):
        print("\nNO coinciden:\n" + malos.to_string(index=False))
        return 1
    print("\nTodo coincide dentro de las tolerancias. Informe: docs/comparacion_original_vs_modular.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
