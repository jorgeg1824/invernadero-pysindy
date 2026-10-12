"""Prueba 2 de la guía (parte 1): transformación de fechas y construcción de series."""

import pandas as pd

from datos.carga import leer_csv_crudo
from datos.transformacion import (analizar_zona_horaria, construir_series, convertir_fechas,
                                  resumen_construccion)


def test_zona_horaria_explicita_utc():
    info = analizar_zona_horaria(pd.Series(["2026-06-15T10:00:00Z", "2026-06-15T10:05:00Z"]))
    assert info["con_zona"] == 2
    assert "UTC" in info["conclusion"]


def test_zona_horaria_ausente_se_advierte():
    info = analizar_zona_horaria(pd.Series(["2026-06-15 10:00:00", "2026-06-15 10:05:00"]))
    assert info["con_zona"] == 0
    assert info["conclusion"].startswith("ADVERTENCIA")


def test_offset_se_convierte_a_utc():
    df = pd.DataFrame({"datetime_utc": ["2026-06-15T05:00:00-05:00"]})
    conv, info = convertir_fechas(df)
    assert conv["datetime_utc"].iloc[0] == pd.Timestamp("2026-06-15T10:00:00Z")
    assert info["fechas_no_convertibles"] == 0


def test_convertir_fechas_registra_invalidas(csv_largo):
    conv, info = convertir_fechas(leer_csv_crudo(csv_largo))
    assert info["fechas_no_convertibles"] == 1
    assert info["tipo_despues"].startswith("datetime64")


def test_construir_series_tabla_conocida(csv_largo):
    conv, _ = convertir_fechas(leer_csv_crudo(csv_largo))
    series = construir_series(conv)
    assert set(series) == {"Temperatura", "Luminosidad", "Humedad"}
    # La fila con fecha inválida se descarta; quedan 3 temperaturas ordenadas.
    assert series["Temperatura"]["Temperatura"].tolist() == [20.0, 21.0, 22.0]
    assert series["Humedad"]["Humedad"].tolist() == [80.0, 70.0, 60.0]
    assert series["Luminosidad"]["datetime_utc"].is_monotonic_increasing

    r = resumen_construccion(conv, series)
    assert r["filas_sin_fecha_o_valor"] == 1
    assert r["metricas_ignoradas"] == {"bomba": 1}
    assert r["muestras_por_variable"] == {"Temperatura": 3, "Luminosidad": 3, "Humedad": 3}


def test_construir_series_ordena_filas_desordenadas():
    df = pd.DataFrame({
        "metric_name": ["Temperatura", "Temperatura", "Luminosidad", "Humedad"],
        "datetime_utc": pd.to_datetime(["2026-01-01T01:00Z", "2026-01-01T00:00Z",
                                        "2026-01-01T00:00Z", "2026-01-01T00:00Z"], utc=True),
        "number_value": [2.0, 1.0, 5.0, 9.0],
    })
    s = construir_series(df)
    assert s["Temperatura"]["Temperatura"].tolist() == [1.0, 2.0]
