"""Prueba 1 de la guía: carga y validación básica de datos."""

import pandas as pd
import pytest

from datos.carga import cargar_csv, detectar_columnas, leer_csv_crudo, validar_estructura


def test_lee_csv_tabulado_sin_transformar(csv_largo):
    df = leer_csv_crudo(csv_largo)
    assert list(df.columns) == ["device_name", "metric_name", "datetime_utc", "number_value"]
    assert len(df) == 11
    assert pd.api.types.is_string_dtype(df["datetime_utc"])     # todavía es texto


def test_lee_csv_separado_por_comas(csv_largo, tmp_path):
    """Corrección C2: el original no podía leer un CSV con comas."""
    ruta = tmp_path / "comas.csv"
    pd.read_csv(csv_largo, sep="\t").to_csv(ruta, index=False)
    df = leer_csv_crudo(ruta)
    assert df.shape == (11, 4)


def test_validar_estructura_ok(csv_largo):
    info = validar_estructura(leer_csv_crudo(csv_largo))
    assert info["formato"].startswith("largo")
    assert info["metricas"]["Temperatura"] == 4
    assert info["metricas"]["bomba"] == 1


def test_validar_estructura_sin_fecha():
    with pytest.raises(ValueError, match="datetime_utc"):
        validar_estructura(pd.DataFrame({"a": [1]}))


def test_validar_estructura_falta_variable(csv_largo):
    df = leer_csv_crudo(csv_largo)
    df = df[df["metric_name"] != "Humedad"]
    with pytest.raises(ValueError, match="Humedad"):
        validar_estructura(df)


def test_cargar_csv_convierte_a_utc(csv_largo):
    df, columnas = cargar_csv(csv_largo)
    assert str(df["datetime_utc"].dt.tz) == "UTC"
    assert df["datetime_utc"].isna().sum() == 1          # 'no-es-fecha'
    assert columnas == {"Temperatura": "number_value", "Luminosidad": "number_value",
                        "Humedad": "number_value"}


def test_detectar_columnas_formato_ancho_con_alias():
    df = pd.DataFrame(columns=["datetime_utc", "temperature", "LUX", "hr"])
    assert detectar_columnas(df) == {"Temperatura": "temperature", "Luminosidad": "LUX", "Humedad": "hr"}
