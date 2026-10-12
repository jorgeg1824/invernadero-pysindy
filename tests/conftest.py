"""Configuración común de las pruebas."""

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import config as C  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "lenta: usa el CSV completo (varios segundos)")


@pytest.fixture
def csv_largo(tmp_path):
    """
    CSV mínimo en formato largo (tabulado) escrito a mano:
    3 instantes x 3 variables + una métrica que debe ignorarse + una fila inválida.
    """
    filas = [
        ("Monitoreo", "Temperatura", "2026-06-15T10:00:00Z", 20.0),
        ("Monitoreo", "Luminosidad", "2026-06-15T10:00:00Z", 100.0),
        ("Monitoreo", "Humedad", "2026-06-15T10:00:00Z", 80.0),
        ("Monitoreo", "bomba", "2026-06-15T10:00:00Z", 1.0),
        ("Monitoreo", "Temperatura", "2026-06-15T10:05:00Z", 21.0),
        ("Monitoreo", "Luminosidad", "2026-06-15T10:05:00Z", 200.0),
        ("Monitoreo", "Humedad", "2026-06-15T10:05:00Z", 70.0),
        ("Monitoreo", "Temperatura", "2026-06-15T10:10:00Z", 22.0),
        ("Monitoreo", "Luminosidad", "2026-06-15T10:10:00Z", 300.0),
        ("Monitoreo", "Humedad", "2026-06-15T10:10:00Z", 60.0),
        ("Monitoreo", "Temperatura", "no-es-fecha", 99.0),
    ]
    df = pd.DataFrame(filas, columns=["device_name", "metric_name", "datetime_utc", "number_value"])
    ruta = tmp_path / "pequeno.csv"
    df.to_csv(ruta, sep="\t", index=False)
    return ruta


def sistema_sintetico(coef, dias=8, paso_min=5, ruido=0.0, semilla=0):
    """
    Datos en malla uniforme generados con una ecuación CONOCIDA
        dT/dt = a0 + aT·T + aL·L + aH·H   (t en horas)
    con entradas tipo día/noche. Devuelve un DataFrame indexado por fecha.
    """
    from simulacion.integrador import simular_lsim

    rng = np.random.default_rng(semilla)
    n = int(dias * 24 * 60 / paso_min) + 1
    idx = pd.date_range("2026-01-01", periods=n, freq=f"{paso_min}min", tz="UTC")
    h = idx.hour + idx.minute / 60
    sol = np.clip(np.sin((h - 6) / 12 * np.pi), 0, None)
    L = 8000 * sol * (1 + 0.2 * rng.standard_normal(n).cumsum() / np.sqrt(n))
    L = np.clip(L, 0, None)
    H = 95 - 50 * sol + 3 * np.sin(np.arange(n) / 37)
    d = pd.DataFrame({"Temperatura": 16.0, "Luminosidad": L, "Humedad": H}, index=idx)
    d["Temperatura"] = simular_lsim(coef, d, paso_min / 60)
    if ruido:
        d["Temperatura"] += ruido * rng.standard_normal(n)
    return d


COEF_CONOCIDO = {"a0": 12.0, "aT": -0.8, "aL": 0.0012, "aH": -0.05}


@pytest.fixture(scope="session")
def datos_reales():
    """CSV de ejemplo cargado y alineado (una vez por sesión)."""
    from datos.alineacion import alinear_variables
    from datos.carga import cargar_csv
    from datos.transformacion import construir_series

    df, _ = cargar_csv(C.CSV_EJEMPLO)
    datos, _ = alinear_variables(construir_series(df))
    return datos
