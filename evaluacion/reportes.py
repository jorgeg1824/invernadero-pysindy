"""
Informes de resultados y registro de cada ejecución.

  * Etapa / Ejecucion: lo que produce cada etapa (texto, tablas, gráficas)
    en memoria; la interfaz gráfica y la consola leen de aquí.
  * RegistroEjecucion: escribe todo en resultados/<fecha>_<comando>/,
    una carpeta por etapa, más configuracion.json y registro.log.
  * Funciones de texto: resumen() y resumen_archivo() de
    invernadero_20dias.py, tabla de ventanas de validar_modelo.py y el
    texto del barrido del profesor.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluacion.metricas import metricas_archivo
from modelos.coeficientes import ecuacion_fisica, por_segundo

NOMBRES_ETAPAS = {
    1: "datos_originales",
    2: "transformacion",
    3: "alineacion",
    4: "limpieza",
    5: "seleccion_tramo",
    6: "preparacion_modelo",
    7: "identificacion",
    8: "simulacion_evaluacion",
}


# ============================================================
# Estructuras en memoria
# ============================================================

@dataclass
class Etapa:
    numero: int
    titulo: str
    texto: str = ""
    datos: dict = field(default_factory=dict)          # resumen JSON
    tablas: dict = field(default_factory=dict)         # nombre -> DataFrame
    graficas: dict = field(default_factory=dict)       # nombre -> función(fig)

    @property
    def clave(self) -> str:
        return f"{self.numero:02d}_{NOMBRES_ETAPAS.get(self.numero, 'extra')}"


@dataclass
class Ejecucion:
    comando: str
    config: dict
    etapas: list = field(default_factory=list)
    resultado: dict = field(default_factory=dict)     # resumen serializable (resultado.json)
    memoria: dict = field(default_factory=dict)       # objetos en memoria (DataFrames, arreglos)
    carpeta: Path | None = None

    def etapa(self, numero: int) -> Etapa:
        for e in self.etapas:
            if e.numero == numero:
                return e
        raise KeyError(numero)


# ============================================================
# Escritura en disco
# ============================================================

def _a_json(x):
    if isinstance(x, dict):
        return {str(k): _a_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_a_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _a_json(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, (pd.Timedelta,)):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    return x


def a_json(x) -> str:
    return json.dumps(_a_json(x), indent=2, ensure_ascii=False, default=str)


class RegistroEjecucion:
    """
    Crea resultados/<AAAAMMDD-HHMMSS>_<comando>/ y guarda en ella:
        configuracion.json, entorno.json, registro.log, resumen.txt
        NN_<etapa>/  resumen.json, resumen.txt, *.csv, *.png
    """

    def __init__(self, carpeta_base, comando: str, config: dict, guardar: bool = True,
                 guardar_graficas: bool = True):
        self.guardar = guardar
        self.guardar_graficas = guardar_graficas
        self.lineas = []
        self.carpeta = None
        self.log = logging.getLogger(f"invernadero.{comando}")
        if guardar:
            marca = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.carpeta = Path(carpeta_base) / f"{marca}_{comando}"
            n = 1
            while self.carpeta.exists():
                n += 1
                self.carpeta = Path(carpeta_base) / f"{marca}_{comando}_{n}"
            self.carpeta.mkdir(parents=True)
            (self.carpeta / "configuracion.json").write_text(a_json(config), encoding="utf-8")
            (self.carpeta / "entorno.json").write_text(a_json(entorno()), encoding="utf-8")

    def info(self, etapa: int | str, mensaje: str):
        linea = f"{datetime.now():%H:%M:%S} [etapa {etapa}] {mensaje}"
        self.lineas.append(linea)
        self.log.info(linea)
        if self.carpeta:
            with open(self.carpeta / "registro.log", "a", encoding="utf-8") as f:
                f.write(linea + "\n")

    def cantidades(self, etapa: int, operacion: str, antes: int, despues: int):
        self.info(etapa, f"{operacion}: {antes} -> {despues} registros ({despues - antes:+d})")

    def guardar_etapa(self, e: Etapa):
        if not self.carpeta:
            return
        from matplotlib.figure import Figure

        d = self.carpeta / e.clave
        d.mkdir(exist_ok=True)
        (d / "resumen.json").write_text(a_json(e.datos), encoding="utf-8")
        (d / "resumen.txt").write_text(f"{e.titulo}\n{'=' * len(e.titulo)}\n\n{e.texto}\n",
                                       encoding="utf-8")
        for nombre, tabla in e.tablas.items():
            tabla.to_csv(d / f"{nombre}.csv", index=isinstance(tabla.index, pd.DatetimeIndex))
        if self.guardar_graficas:
            for nombre, dibujar in e.graficas.items():
                fig = Figure(figsize=(13, 6.5), layout="constrained")
                dibujar(fig)
                fig.savefig(d / f"{nombre}.png", dpi=110)

    def cerrar(self, texto_resumen: str, resultado: dict):
        if not self.carpeta:
            return
        (self.carpeta / "resumen.txt").write_text(texto_resumen + "\n", encoding="utf-8")
        (self.carpeta / "resultado.json").write_text(a_json(resultado), encoding="utf-8")


def entorno() -> dict:
    import matplotlib, numpy, pandas, pysindy, scipy, sklearn
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "sistema": platform.platform(),
        "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "scikit-learn": sklearn.__version__,
        "pysindy": pysindy.__version__,
    }


# ============================================================
# Textos
# ============================================================

def texto_dict(d: dict, sangria: str = "  ") -> str:
    """Diccionario -> líneas 'clave: valor' legibles."""
    lineas = []
    for k, v in d.items():
        if isinstance(v, dict):
            lineas.append(f"{sangria}{k}:")
            lineas.append(texto_dict(v, sangria + "  "))
        else:
            if isinstance(v, (float, np.floating)):
                v = f"{v:.6g}"
            lineas.append(f"{sangria}{k}: {v}")
    return "\n".join(lineas)


def texto_metricas(m: dict) -> str:
    return f"RMSE = {m['RMSE']:.4f} °C   MAE = {m['MAE']:.4f} °C   R² = {m['R2']:.4f}"


def texto_barrido(tabla: list) -> str:
    lineas = ["THRESHOLDS PROBADOS", "--------------------"]
    for r in tabla:
        if "error" in r:
            lineas.append(f"{r['threshold']:.5g}   falló: {r['error']}")
        else:
            estado = "" if abs(r["RMSE"]) < 1e3 else "   (diverge)"
            lineas.append(f"{r['threshold']:<8.5g} RMSE={r['RMSE']:<10.5g} R²={r['R2']:<10.5g} "
                          f"términos={r['terminos']}{estado}")
    return "\n".join(lineas)


def resumen_prediccion(r: dict, dias_prediccion: float) -> str:
    """resumen() de invernadero_20dias.py (mismo contenido)."""
    dias = lambda d: (d.index[-1] - d.index[0]).total_seconds() / 86400
    m, mb = r["metricas"], r["metricas_base"]
    lineas = [
        f"Entrenamiento: {r['train'].index[0]:%Y-%m-%d} -> {r['train'].index[-1]:%Y-%m-%d} ({dias(r['train']):.1f} dias)",
        f"Prediccion:    {r['val'].index[0]:%Y-%m-%d} -> {r['val'].index[-1]:%Y-%m-%d} ({dias(r['val']):.1f} dias)",
        "",
        "Ecuacion PySINDy (solo regresion de la derivada):",
        ecuacion_fisica(por_segundo(r["coef_sindy"])),
        "  -> " + ("estable" if r["coef_sindy"]["aT"] < 0 else "INESTABLE (aT > 0), diverge al simular"),
    ]
    if r.get("coef_ajustado") is not None:
        lineas += ["", "Ecuacion final (ajustada por simulacion, t en segundos):",
                   ecuacion_fisica(por_segundo(r["coef"]))]
    lineas += [
        "",
        f"Prediccion de {dias_prediccion:g} dias  [estrategia: {r['estrategia']}]:",
        f"  RMSE = {m['RMSE']:.3f} C   MAE = {m['MAE']:.3f} C   R2 = {m['R2']:.4f}",
        "Referencia (perfil diario promedio):",
        f"  RMSE = {mb['RMSE']:.3f} C   MAE = {mb['MAE']:.3f} C   R2 = {mb['R2']:.4f}",
        "",
        "MAE por dia de prediccion (C):",
        "  " + " ".join(f"{e:.2f}" for e in r["mae_dias"]),
    ]
    return "\n".join(lineas)


def resumen_archivo(res: pd.DataFrame, muestras_por_dia: float) -> str:
    """resumen_archivo() de invernadero_20dias.py."""
    total, por_mes = metricas_archivo(res)
    dias = (~res["entrenamiento"]).sum() / muestras_por_dia
    lineas = [
        "COMPARACION CON TODO EL ARCHIVO",
        f"(sin contar el entrenamiento: {dias:.0f} dias)",
        f"  RMSE = {total['RMSE']:.3f} C   MAE = {total['MAE']:.3f} C   R2 = {total['R2']:.4f}",
        "",
        "Por mes:",
    ]
    lineas += [f"  {mes}:  MAE = {m['MAE']:.2f} C   R2 = {m['R2']:.3f}" for mes, m in por_mes.items()]
    return "\n".join(lineas)


def tabla_ventanas(resultados: list) -> str:
    """Tabla en consola de validar_modelo.py."""
    lineas = [f"{'Ventana de prediccion':24s} {'Entren.':>8s} {'MAE':>6s} {'RMSE':>6s} {'R2':>6s}"
              f" | {'MAE ref':>7s} {'R2 ref':>6s} | {'Peor dia':>8s}  PySINDy solo"]
    for r in resultados:
        m, mb = r["metricas"], r["metricas_base"]
        estado = "estable" if r["coef_sindy"]["aT"] < 0 else "diverge"
        lineas.append(f"{r['nombre']:24s} {r['dias_train']:6.1f} d {m['MAE']:6.2f} {m['RMSE']:6.2f} {m['R2']:6.3f}"
                      f" | {mb['MAE']:7.2f} {mb['R2']:6.3f} | {max(r['mae_dias']):8.2f}  {estado}")
    return "\n".join(lineas)
