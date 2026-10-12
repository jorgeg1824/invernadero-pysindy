"""Prueba 5 de la guía: simulación."""

import numpy as np
import pandas as pd
import pytest

from conftest import COEF_CONOCIDO, sistema_sintetico
from modelos.estrategias.identificacion_base import DerivadaMalla, IdentificacionBase
from simulacion.integrador import simular_lsim, simular_modelo
from simulacion.prediccion import perfil_diario, simular_archivo


def test_lsim_coincide_con_la_solucion_analitica():
    """Entradas constantes: T(t) = T* + (T0 - T*)·e^(aT·t),  T* = -(a0 + aL·L + aH·H)/aT."""
    c = {"a0": 10.0, "aT": -0.5, "aL": 0.001, "aH": -0.02}
    idx = pd.date_range("2026-01-01", periods=13, freq="5min", tz="UTC")
    d = pd.DataFrame({"Temperatura": 30.0, "Luminosidad": 1000.0, "Humedad": 50.0}, index=idx)
    t = np.arange(13) * 5 / 60
    T_eq = -(10 + 1.0 - 1.0) / -0.5
    esperado = T_eq + (30 - T_eq) * np.exp(-0.5 * t)
    assert np.allclose(simular_lsim(c, d, 5 / 60), esperado, atol=1e-9)


def test_rk45_y_lsim_dan_la_misma_trayectoria():
    """Los dos integradores de los originales coinciden para el mismo modelo."""
    d = sistema_sintetico(COEF_CONOCIDO, dias=2)
    res = IdentificacionBase(1e-3, DerivadaMalla(5, 2, 5 / 60)).ejecutar(d)
    p_lsim = simular_lsim(res.coef, d, 5 / 60)
    # simular_modelo mide el tiempo en segundos y el modelo está en horas:
    # se usan marcas separadas 5/60 s para que "1 s" del eje equivalga a 1 h.
    dd = d.reset_index(drop=True)
    dd.insert(0, "datetime_utc", pd.Timestamp("2026-01-01", tz="UTC")
              + pd.to_timedelta(np.arange(len(d)) * 5 / 60, unit="s"))
    p_rk45 = simular_modelo(res.modelo, res.escalador, dd, "s")
    error = np.abs(p_lsim - p_rk45)
    # RK45 usa la tolerancia por defecto de solve_ivp (rtol = 1e-3): en promedio
    # coincide con lsim (error medio ~0.015 °C), pero se desvía hasta ~0.13 °C en los cambios bruscos de luz.
    assert error.mean() < 0.03
    assert error.max() < 0.2


def test_simular_archivo_marca_entrenamiento_y_arranca_de_la_medicion():
    d = sistema_sintetico(COEF_CONOCIDO, dias=2)
    b1, b2 = d.iloc[:300], d.iloc[400:]
    res = simular_archivo([b1, b2], COEF_CONOCIDO, b1, 5 / 60)
    assert res["entrenamiento"].sum() == len(b1)
    assert res.loc[b2.index[0], "simulada"] == pytest.approx(b2["Temperatura"].iloc[0])
    assert np.allclose(res["simulada"], res["real"], atol=1e-6)


def test_perfil_diario():
    idx = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    train = pd.DataFrame({"Temperatura": np.r_[np.arange(24), np.arange(24) + 2.0]}, index=idx)
    val = pd.DataFrame(index=pd.date_range("2026-01-03", periods=3, freq="1h", tz="UTC"))
    assert perfil_diario(train, val).tolist() == [1.0, 2.0, 3.0]
