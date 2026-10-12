"""Prueba 2 de la guía (parte 2): alineación temporal con una tabla hecha a mano."""

import numpy as np
import pandas as pd
import pytest

from datos.alineacion import alinear_variables, diferencias_temporales, estimar_dt_median


def _serie(nombre, tiempos, valores):
    return pd.DataFrame({"datetime_utc": pd.to_datetime(tiempos, utc=True), nombre: valores})


@pytest.fixture
def series():
    # T cada 5 min. L llega 30 s tarde. H llega 10 s antes, salvo la última,
    # que está 20 min después (fuera de la tolerancia 2 x 300 s = 600 s).
    T = _serie("Temperatura", ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10",
                               "2026-01-01 00:15"], [10.0, 11.0, 12.0, 13.0])
    L = _serie("Luminosidad", ["2026-01-01 00:00:30", "2026-01-01 00:05:30", "2026-01-01 00:10:30",
                               "2026-01-01 00:15:30"], [0.0, 1.0, 2.0, 3.0])
    H = _serie("Humedad", ["2025-12-31 23:59:50", "2026-01-01 00:04:50", "2026-01-01 00:09:50",
                           "2026-01-01 00:35:00"], [90.0, 80.0, 70.0, 60.0])
    return {"Temperatura": T, "Luminosidad": L, "Humedad": H}


def test_dt_mediano(series):
    assert estimar_dt_median(series["Temperatura"]) == 300.0


def test_asigna_la_medicion_mas_cercana(series):
    datos, dt = alinear_variables(series)
    assert dt == 300.0
    assert datos["Luminosidad"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert datos["Humedad"].tolist()[:3] == [90.0, 80.0, 70.0]


def test_nearest_puede_reutilizar_una_medicion(series):
    """Para T 00:15 la H más cercana es la de 00:09:50 (310 s), que ya se usó a las 00:10."""
    datos, _ = alinear_variables(series)
    assert datos["Humedad"].iloc[3] == 70.0


def test_fuera_de_tolerancia_queda_nan(series):
    # Con tolerancia 1 x dt = 300 s, los 310 s de 00:15 -> 00:09:50 ya no alcanzan.
    datos, _ = alinear_variables(series, factor_tolerancia=1)
    assert np.isnan(datos["Humedad"].iloc[3])
    assert datos["Humedad"].iloc[2] == 70.0


def test_elimina_marcas_duplicadas(series):
    T = pd.concat([series["Temperatura"], series["Temperatura"].iloc[[1]]])
    datos, _ = alinear_variables({**series, "Temperatura": T})
    assert len(datos) == 4
    assert datos.attrs["timestamps_duplicados_eliminados"] == 1
    assert datos["datetime_utc"].is_monotonic_increasing


def test_diferencias_temporales(series):
    d = diferencias_temporales(series)
    assert d["desfase_Luminosidad_s"].tolist() == [30.0] * 4
    assert d["desfase_Humedad_s"].tolist()[:3] == [-10.0] * 3
