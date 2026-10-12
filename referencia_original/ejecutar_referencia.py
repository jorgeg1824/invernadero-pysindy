"""
Etapa 2 - Ejecuta los TRES programas originales SIN modificarlos y guarda
sus resultados como referencia para comparar la versión modular.

Los originales se importan desde ../originales/ tal cual están. Este script
solo:
  * simula los clics de la interfaz de invernadero_profesor.py
    (Abrir CSV -> Buscar mejor tramo -> Ejecutar PySINDy),
  * llama al modo consola de invernadero_20dias.py,
  * ejecuta validar_modelo.py como script,
y vuelca en disco lo que cada programa calcula (ecuaciones, coeficientes,
métricas, trayectorias simuladas y gráficas).

Uso (desde la raíz del repositorio):

    python referencia_original/ejecutar_referencia.py
    python referencia_original/ejecutar_referencia.py --salida referencia_original/pandas2

La interfaz del profesor necesita una pantalla. En Linux sin pantalla se
puede usar:  xvfb-run python referencia_original/ejecutar_referencia.py
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import runpy
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGINALES = RAIZ / "originales"
CSV = RAIZ / "datos_ejemplo" / "Aeronic_export_29_09_2026.csv"

sys.path.insert(0, str(ORIGINALES))
os.environ.setdefault("MPLBACKEND", "Agg")


def md5(ruta):
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


def a_json(x):
    """Convierte numpy / pandas a tipos serializables."""
    import numpy as np
    import pandas as pd
    if isinstance(x, dict):
        return {str(k): a_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [a_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


def entorno():
    import numpy, pandas, scipy, matplotlib, sklearn, pysindy
    return {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
        "pysindy": pysindy.__version__,
        "md5_csv": md5(CSV),
        "md5_originales": {p.name: md5(p) for p in sorted(ORIGINALES.glob("*.py"))},
    }


# ============================================================
# 1. invernadero_profesor.py  (solo tiene interfaz gráfica)
# ============================================================

def referencia_profesor(salida):
    import numpy as np
    import pandas as pd
    import invernadero_profesor as prof

    salida.mkdir(parents=True, exist_ok=True)
    errores = []

    # Se reemplazan los diálogos para poder "hacer clic" sin intervención.
    prof.filedialog.askopenfilename = lambda **kw: str(CSV)
    prof.messagebox.showerror = lambda t, m: errores.append(f"{t}: {m}")
    prof.messagebox.showwarning = lambda t, m: errores.append(f"{t}: {m}")

    root = prof.tk.Tk()
    app = prof.Aplicacion(root)

    pasos = {}
    t0 = time.time()

    app.abrir_csv()
    pasos["1_abrir_csv"] = app.info.get("1.0", "end")
    app.fig.savefig(salida / "1_datos_cargados.png", dpi=110)

    app.buscar_tramo()
    pasos["2_buscar_mejor_tramo"] = app.info.get("1.0", "end")
    app.fig.savefig(salida / "2_mejor_tramo.png", dpi=110)

    salida_consola = io.StringIO()
    with contextlib.redirect_stdout(salida_consola):
        app.ejecutar_sindy()
    pasos["3_ejecutar_pysindy"] = app.info.get("1.0", "end")
    app.fig.savefig(salida / "3_validacion_pysindy.png", dpi=110)

    duracion = time.time() - t0

    # La interfaz no guarda el resultado; se recalcula con las MISMAS
    # funciones y argumentos que usa ejecutar_sindy() para poder volcarlo.
    datos = prof.limpiar_tramo(app.mejor_bloque["datos"])
    train, val = prof.dividir_train_validacion(datos, dias_entrenamiento=5)
    with contextlib.redirect_stdout(io.StringIO()):
        mejor, resultados = prof.buscar_mejor_modelo(train, val)
    coef = prof.obtener_coeficientes_fisicos(mejor["modelo"], mejor["escalador"])

    pd.DataFrame({
        "datetime_utc": val["datetime_utc"].values,
        "T_real": val["Temperatura"].values,
        "T_simulada": mejor["pred"],
    }).to_csv(salida / "trayectoria_validacion.csv", index=False)

    bloque = app.mejor_bloque
    resumen = {
        "programa": "invernadero_profesor.py",
        "como_se_ejecuto": "Interfaz: Abrir CSV -> Buscar mejor tramo -> Ejecutar PySINDy",
        "configuracion": {
            "dias_minimos_tramo": 6,
            "dias_entrenamiento": 5,
            "factor_corte_bloques": 4.0,
            "ventana_savgol": 21,
            "thresholds": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "alpha_stlsq": 0.001,
            "integrador": "solve_ivp RK45",
            "criterio_seleccion": "menor RMSE en validacion",
        },
        "registros": {
            "filas_csv": int(len(app.df)),
            "muestras_por_variable": {v: int(len(app.series[v])) for v in prof.VARIABLES},
            "filas_alineadas": int(len(app.datos)),
            "duplicados_eliminados": int(app.datos.attrs.get("timestamps_duplicados_eliminados", 0)),
            "bloques_validos": int(len(app.bloques)),
            "muestras_tramo": int(bloque["n"]),
            "muestras_tramo_limpio": int(len(datos)),
            "muestras_train": int(len(train)),
            "muestras_validacion": int(len(val)),
        },
        "tramo": {
            "inicio": bloque["inicio"],
            "fin": bloque["fin"],
            "duracion_h": bloque["duracion"] / 3600,
            "excitacion": bloque["excitación"],
            "score": bloque["score"],
            "train_inicio": train["datetime_utc"].iloc[0],
            "train_fin": train["datetime_utc"].iloc[-1],
            "val_inicio": val["datetime_utc"].iloc[0],
            "val_fin": val["datetime_utc"].iloc[-1],
        },
        "threshold_elegido": mejor["threshold"],
        "coeficientes_normalizados": dict(zip(
            mejor["modelo"].get_feature_names(),
            np.asarray(mejor["modelo"].coefficients()).ravel().tolist(),
        )),
        "coeficientes_fisicos": coef,
        "ecuacion": prof.ecuacion_fisica(coef),
        "metricas_validacion": mejor["metricas"],
        "barrido_threshold": resultados,
        "errores_interfaz": errores,
        "duracion_s": duracion,
    }
    (salida / "resultados.json").write_text(
        json.dumps(a_json(resumen), indent=2, ensure_ascii=False, default=str))
    (salida / "interfaz_texto.txt").write_text(
        "\n\n".join(f"===== {k} =====\n{v}" for k, v in pasos.items()))
    (salida / "consola.txt").write_text(salida_consola.getvalue())

    root.destroy()
    return resumen


# ============================================================
# 2. invernadero_20dias.py  (tiene modo consola)
# ============================================================

def referencia_20dias(salida):
    import numpy as np
    import pandas as pd
    import invernadero_profesor as prof
    import invernadero_20dias as M

    salida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    consola = io.StringIO()
    with contextlib.redirect_stdout(consola):
        M.modo_consola(str(CSV), carpeta=str(salida / "graficas"))
    duracion = time.time() - t0
    (salida / "consola.txt").write_text(consola.getvalue())

    # Mismo procedimiento que modo_consola() para volcar los números.
    df, _ = prof.cargar_csv(str(CSV))
    datos, _ = prof.alinear_variables(prof.construir_series(df))
    bloques = M.bloques_rellenados(datos)
    bloque = M.elegir_bloque(bloques)
    r = M.evaluar(bloque)
    res = M.simular_archivo(bloques, r["coef"], r["train"])
    total, por_mes = M.metricas_archivo(res)

    pd.DataFrame({
        "datetime_utc": r["val"].index,
        "T_real": r["val"]["Temperatura"].values,
        "T_simulada": r["pred"],
        "T_simulada_sindy": r["pred_sindy"] if r["pred_sindy"] is not None else np.nan,
        "T_referencia_perfil": r["base"],
    }).to_csv(salida / "trayectoria_prediccion_20d.csv", index=False)
    res.rename_axis("datetime_utc").to_csv(salida / "trayectoria_archivo.csv")

    resumen = {
        "programa": "invernadero_20dias.py",
        "como_se_ejecuto": "python invernadero_20dias.py --consola <csv>",
        "configuracion": {
            "DIAS_PREDICCION": M.DIAS_PREDICCION,
            "DIAS_ENTRENAMIENTO_MIN": M.DIAS_ENTRENAMIENTO_MIN,
            "MAX_HUECO_HORAS": M.MAX_HUECO_HORAS,
            "PASO_min": M.PASO.total_seconds() / 60,
            "THRESHOLD": M.THRESHOLD,
            "ventana_savgol": 21,
            "integrador": "scipy.signal.lsim",
            "ZONA_HORARIA": M.ZONA_HORARIA,
        },
        "registros": {
            "filas_alineadas": int(len(datos)),
            "bloques": int(len(bloques)),
            "duracion_bloques_dias": [
                (b.index[-1] - b.index[0]).total_seconds() / 86400 for b in bloques],
            "muestras_bloque": int(len(bloque)),
            "muestras_train": int(len(r["train"])),
            "muestras_validacion": int(len(r["val"])),
        },
        "tramo": {
            "inicio": bloque.index[0], "fin": bloque.index[-1],
            "train_inicio": r["train"].index[0], "train_fin": r["train"].index[-1],
            "val_inicio": r["val"].index[0], "val_fin": r["val"].index[-1],
        },
        "coeficientes_normalizados_sindy": dict(zip(
            r["modelo"].get_feature_names(),
            np.asarray(r["modelo"].coefficients()).ravel().tolist())),
        "coef_sindy_por_hora": r["coef_sindy"],
        "coef_ajustado_por_hora": r["coef"],
        "sindy_estable": bool(r["coef_sindy"]["aT"] < 0),
        "metricas_prediccion": r["metricas"],
        "metricas_referencia_perfil": r["metricas_base"],
        "mae_por_dia": r["mae_dias"],
        "metricas_archivo_sin_train": total,
        "metricas_por_mes": por_mes,
        "duracion_s": duracion,
    }
    (salida / "resultados.json").write_text(
        json.dumps(a_json(resumen), indent=2, ensure_ascii=False, default=str))
    return resumen


# ============================================================
# 3. validar_modelo.py  (script que se ejecuta al importarlo)
# ============================================================

def referencia_validar(salida):
    salida.mkdir(parents=True, exist_ok=True)
    # El script usa rutas relativas: lee el CSV y escribe validacion/
    # en el directorio actual.
    shutil.copy(CSV, salida / CSV.name)
    anterior = os.getcwd()
    os.chdir(salida)
    consola = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(consola):
            g = runpy.run_path(str(ORIGINALES / "validar_modelo.py"), run_name="__main__")
    finally:
        os.chdir(anterior)
        (salida / CSV.name).unlink(missing_ok=True)
    duracion = time.time() - t0
    (salida / "consola.txt").write_text(consola.getvalue())

    ventanas = [{
        "nombre": r["nombre"],
        "dias_train": r["dias_train"],
        "train_inicio": r["train"].index[0], "train_fin": r["train"].index[-1],
        "val_inicio": r["val"].index[0], "val_fin": r["val"].index[-1],
        "coef_ajustado_por_hora": r["coef"],
        "coef_sindy_por_hora": r["coef_sindy"],
        "metricas": r["metricas"],
        "metricas_referencia": r["metricas_base"],
        "peor_dia_mae": max(r["mae_dias"]),
    } for r in g["resultados"]]

    resumen = {
        "programa": "validar_modelo.py",
        "como_se_ejecuto": "python validar_modelo.py (desde la carpeta con el CSV)",
        "ventanas": ventanas,
        "duracion_s": duracion,
    }
    (salida / "resultados.json").write_text(
        json.dumps(a_json(resumen), indent=2, ensure_ascii=False, default=str))
    return resumen


PROGRAMAS = {
    "profesor": referencia_profesor,
    "20dias": referencia_20dias,
    "validar_modelo": referencia_validar,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salida", default=str(RAIZ / "referencia_original" / "resultados"))
    p.add_argument("--programas", nargs="+", default=list(PROGRAMAS), choices=list(PROGRAMAS))
    a = p.parse_args()

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    registro = {"entorno": entorno(), "programas": {}}
    for nombre in a.programas:
        print(f"== Ejecutando {nombre} ...", flush=True)
        try:
            r = PROGRAMAS[nombre](salida / nombre)
            registro["programas"][nombre] = {"ok": True, "duracion_s": r["duracion_s"]}
        except Exception:
            registro["programas"][nombre] = {"ok": False, "error": traceback.format_exc()}
            print(traceback.format_exc())
    (salida / "registro_ejecucion.json").write_text(
        json.dumps(a_json(registro), indent=2, ensure_ascii=False, default=str))
    print("Referencia guardada en", salida)


if __name__ == "__main__":
    main()
