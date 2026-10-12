"""Prueba 3 de la guía: cálculo (derivadas, eje de tiempo) y normalización."""

import numpy as np
import pandas as pd
import pytest

from analisis.derivadas import (calcular_derivada_temperatura, derivada_savgol_directa,
                                tiempo_absoluto, tiempo_relativo, unidad_efectiva)
from analisis.normalizacion import Escalador


def test_escalador_media_cero_desviacion_uno():
    d = pd.DataFrame({"Temperatura": [10.0, 20.0, 30.0], "Luminosidad": [0.0, 0.0, 300.0],
                      "Humedad": [50.0, 50.0, 50.0]})
    esc = Escalador().fit(d)
    n = esc.transform(d)
    assert np.allclose(n["Temperatura"], [-np.sqrt(1.5), 0, np.sqrt(1.5)])
    assert np.isclose(n["Luminosidad"].mean(), 0) and np.isclose(n["Luminosidad"].std(ddof=0), 1)
    # Variable constante: desviación 0 -> se usa 1 para no dividir por cero.
    assert esc.std["Humedad"] == 1.0 and (n["Humedad"] == 0).all()
    assert np.isclose(esc.inversa("Temperatura", n["Temperatura"].iloc[2]), 30.0)


def test_derivada_malla_de_una_rampa():
    T = 15 + 2.0 * np.arange(100) * (5 / 60)        # 2 °C/h
    assert np.allclose(derivada_savgol_directa(T, 21, 2, 5 / 60), 2.0)


def test_derivada_profesor_en_segundos():
    t = pd.date_range("2026-01-01", periods=200, freq="300s", tz="UTC")
    d = pd.DataFrame({"datetime_utc": t, "Temperatura": 20 + 0.001 * np.arange(200) * 300})
    _, dTdt = calcular_derivada_temperatura(d)
    assert np.allclose(dTdt, 0.001)                  # °C/s


@pytest.mark.parametrize("resolucion", ["ns", "us", "ms", "s"])
def test_tiempo_en_segundos_no_depende_de_la_resolucion(resolucion):
    """Corrección C1: el resultado no cambia con la resolución interna de pandas."""
    t = pd.Series(pd.to_datetime(["1970-01-01T00:00:00Z", "1970-01-01T00:05:00Z"], utc=True)
                  ).astype(f"datetime64[{resolucion}, UTC]")
    assert tiempo_absoluto(t, "s").tolist() == [0.0, 300.0]
    assert tiempo_relativo(t, "s").tolist() == [0.0, 300.0]


def test_expresion_original_depende_de_la_resolucion():
    """Documenta el error C1: int64/1e9 solo da segundos si la resolución es ns."""
    t = pd.Series(pd.to_datetime(["1970-01-01T00:05:00Z"], utc=True))
    assert tiempo_absoluto(t.astype("datetime64[ns, UTC]"), "original")[0] == 300.0
    assert tiempo_absoluto(t.astype("datetime64[us, UTC]"), "original")[0] == 0.3   # kilosegundos


def test_unidad_efectiva_conocida():
    assert unidad_efectiva("s") == "s"
    assert unidad_efectiva("original") in {"s", "ks"}
