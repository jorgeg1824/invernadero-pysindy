"""Limpieza, relleno de huecos, segmentación, selección de tramo y partición."""

import numpy as np
import pandas as pd

from analisis.particion import (cortes_ventanas, dividir_train_validacion, dividir_ultimos_dias,
                                separar_seleccion)
from analisis.seleccion_tramo import elegir_bloque, seleccionar_mejor_bloque
from datos.limpieza import limpiar_tramo, puntos_en_huecos, rellenar_en_malla
from datos.segmentacion import bloques_rellenados, listar_huecos, obtener_bloques_continuos


def _datos(tiempos):
    t = pd.DatetimeIndex(tiempos)
    n = len(t)
    return pd.DataFrame({"datetime_utc": t, "Temperatura": np.arange(n, dtype=float),
                         "Luminosidad": 1.0, "Humedad": 2.0})


def _malla(inicio, n, paso="5min"):
    return list(pd.date_range(inicio, periods=n, freq=paso, tz="UTC"))


def test_bloques_continuos_cortan_en_hueco():
    # 30 muestras, hueco de 1 h, 25 muestras, hueco, 5 muestras (< 20 -> descartado)
    t = _malla("2026-01-01", 30) + _malla("2026-01-01 04:00", 25) + _malla("2026-01-02", 5)
    bloques = obtener_bloques_continuos(_datos(t), factor=4)
    assert [b["n"] for b in bloques] == [30, 25]


def test_rellenar_en_malla_interpola_linealmente():
    g = pd.DataFrame({"Temperatura": [0.0, 10.0], "Luminosidad": [0.0, 0.0], "Humedad": [0.0, 0.0]},
                     index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:20"], utc=True))
    b = rellenar_en_malla(g, pd.Timedelta(minutes=5))
    assert b["Temperatura"].tolist() == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_bloques_rellenados_rellena_corto_y_corta_largo():
    t = _malla("2026-01-01", 12) + _malla("2026-01-01 01:30", 12) + _malla("2026-01-01 06:00", 12)
    bloques = bloques_rellenados(_datos(t), max_hueco_horas=2, paso=pd.Timedelta(minutes=5))
    # Hueco de 35 min -> se rellena;  hueco de 3,4 h -> nuevo tramo.
    assert len(bloques) == 2
    assert (bloques[0].index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all()
    huecos = puntos_en_huecos(_datos(t), bloques[0], pd.Timedelta(minutes=5))
    # Última medición 00:55, siguiente 01:30: 01:05, 01:10, 01:15 y 01:20 quedan a > 5 min.
    assert huecos.sum() == 4


def test_listar_huecos():
    t = _malla("2026-01-01", 10) + _malla("2026-01-01 03:00", 10)
    h = listar_huecos(_datos(t), pd.Timedelta(hours=1))
    assert len(h) == 1 and round(h["duracion_h"].iloc[0], 3) == round(2 + 15 / 60, 3)


def test_limpiar_tramo_quita_nan_y_duplicados():
    d = _datos(_malla("2026-01-01", 10))
    d.loc[3, "Humedad"] = np.nan
    d = pd.concat([d, d.iloc[[5]]])
    limpio = limpiar_tramo(d)
    assert len(limpio) == 9 and limpio["datetime_utc"].is_unique


def test_seleccion_por_puntaje_y_por_longitud():
    t = _malla("2026-01-01", 24 * 12 * 7) + _malla("2026-01-09", 24 * 12 * 3)
    bloques = obtener_bloques_continuos(_datos(t))
    mejor, validos, motivo = seleccionar_mejor_bloque(bloques, dias_minimos=6)
    assert len(validos) == 1 and mejor["n"] == 24 * 12 * 7 and "puntaje" in motivo

    mallas = [pd.DataFrame(index=pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"))
              for n in (288 * 26, 288 * 30)]
    elegido, _ = elegir_bloque(mallas, dias_prediccion=20, dias_entrenamiento_min=5)
    assert len(elegido) == 288 * 30


def test_particiones():
    d = _datos(_malla("2026-01-01", 288 * 8))
    train, val = dividir_train_validacion(d, dias_entrenamiento=5)
    assert train["datetime_utc"].max() < val["datetime_utc"].min()
    assert len(train) == 288 * 5

    ajuste, sel = separar_seleccion(train, 0.2)
    assert ajuste["datetime_utc"].max() < sel["datetime_utc"].min()
    assert abs(len(sel) / len(train) - 0.2) < 0.01

    b = d.set_index("datetime_utc")
    tr, pr = dividir_ultimos_dias(b, dias_prediccion=3)
    assert (pr.index[-1] - pr.index[0]) < pd.Timedelta(days=3)
    assert len(tr) + len(pr) == len(b)


def test_cortes_de_ventanas_como_validar_modelo():
    b = pd.DataFrame(index=pd.date_range("2026-01-01", periods=288 * 34 + 1, freq="5min", tz="UTC"))
    assert cortes_ventanas(b, 25, 4) == [25, 29, 33, 34.0]
