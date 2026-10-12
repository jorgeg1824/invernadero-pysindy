"""
Parámetros, rutas y opciones configurables del proyecto.

Todos los valores por defecto reproducen los programas originales.
Las únicas diferencias de comportamiento son correcciones documentadas en
docs/informe_final.md; cada una tiene una opción aquí para volver al
comportamiento original y poder compararlo.

Una configuración se puede guardar/cargar como JSON:

    python cli.py barrido --config mi_config.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# ============================================================
# RUTAS
# ============================================================

RAIZ = Path(__file__).resolve().parent
CSV_EJEMPLO = RAIZ / "datos_ejemplo" / "Aeronic_export_29_09_2026.csv"
CARPETA_RESULTADOS = RAIZ / "resultados"

# ============================================================
# VARIABLES DEL CSV
# ============================================================

VARIABLES = ["Temperatura", "Luminosidad", "Humedad"]

VARIABLES_ALIASES = {
    "Temperatura": ["Temperatura", "temperatura", "temperature"],
    "Luminosidad": ["Luminosidad", "luminosidad", "luz", "lux"],
    "Humedad": ["Humedad", "humedad", "HR", "hr"],
}

UNIDADES = {
    "Temperatura": "°C",
    "Luminosidad": "lux",
    "Humedad": "%",
}

ENTRADAS = ["Luminosidad", "Humedad"]   # u0, u1
ESTADO = "Temperatura"                   # x0


# ============================================================
# FLUJO "barrido"  (invernadero_profesor.py)
# ============================================================

@dataclass
class ConfigBarrido:
    """Selección de tramo por puntaje + barrido de threshold (programa del profesor)."""

    tolerancia_alineacion: float = 2.0        # x dt típico (merge_asof nearest)
    factor_corte_bloques: float = 4.0          # salto > factor * dt mediano corta un bloque
    factor_limpieza: float = 5.0               # limpiar_tramo descarta dt > factor * mediana
    dias_minimos_tramo: float = 6.0
    dias_entrenamiento: float = 5.0
    ventana_savgol: int = 21
    polyorder_savgol: int = 2
    thresholds: list = field(default_factory=lambda: [
        0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01])
    alpha: float = 0.001

    # Corrección C1 (docs/informe_final.md).
    # "s"        -> tiempo en segundos reales (igual al original con pandas 2.x).
    # "original" -> reproduce la expresión original astype('int64')/1e9,
    #               cuyo resultado depende de la versión de pandas
    #               (con pandas 3.x da kilosegundos).
    unidad_tiempo: str = "s"

    # Mejora M2: con qué datos se elige el threshold.
    # "validacion" -> como el original (el mismo conjunto que se reporta).
    # "separado"   -> se reserva el final del entrenamiento para elegir el
    #                 threshold; la validación queda como prueba independiente.
    seleccion_threshold: str = "validacion"
    fraccion_seleccion: float = 0.2


# ============================================================
# FLUJO "prediccion"  (invernadero_20dias.py)
# ============================================================

@dataclass
class ConfigPrediccion:
    """Tramos rellenados + PySINDy + ajuste por simulación (predicción a N días)."""

    tolerancia_alineacion: float = 2.0
    dias_prediccion: float = 20.0
    dias_entrenamiento_min: float = 5.0
    max_hueco_horas: float = 2.0
    paso_minutos: float = 5.0
    threshold: float = 0.001
    alpha: float = 0.001
    ventana_savgol: int = 21
    polyorder_savgol: int = 2
    zona_horaria: str = "America/Bogota"     # solo para agrupar por mes / hora local

    # True = base PySINDy + ajuste por simulación (original).
    # False = solo la identificación base de PySINDy.
    ajustar_por_simulacion: bool = True

    # Solo afecta la presentación: el original imprime la ecuación
    # convirtiendo los coeficientes a segundos.
    mostrar_en_segundos: bool = True


# ============================================================
# FLUJO "ventanas"  (validar_modelo.py)
# ============================================================

@dataclass
class ConfigVentanas:
    """Repite la predicción del flujo 'prediccion' sobre varias ventanas."""

    paso_ventanas_dias: int = 4
    prediccion: ConfigPrediccion = field(default_factory=ConfigPrediccion)


# ============================================================
# Utilidades
# ============================================================

CONFIGS = {
    "barrido": ConfigBarrido,
    "prediccion": ConfigPrediccion,
    "ventanas": ConfigVentanas,
}


def a_dict(cfg) -> dict:
    return asdict(cfg)


def desde_dict(clase, datos: dict):
    """Crea la configuración ignorando claves desconocidas (con aviso)."""
    validos = {f.name for f in fields(clase)}
    desconocidas = set(datos) - validos
    if desconocidas:
        raise ValueError(f"Parámetros desconocidos para {clase.__name__}: {sorted(desconocidas)}")
    datos = dict(datos)
    if clase is ConfigVentanas and isinstance(datos.get("prediccion"), dict):
        datos["prediccion"] = desde_dict(ConfigPrediccion, datos["prediccion"])
    return clase(**datos)


def cargar_config(flujo: str, ruta: str | Path | None = None):
    """Configuración por defecto del flujo, sobrescrita con un JSON opcional."""
    clase = CONFIGS[flujo]
    if ruta is None:
        return clase()
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return desde_dict(clase, datos)


def guardar_config(cfg, ruta: str | Path):
    Path(ruta).write_text(json.dumps(a_dict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
