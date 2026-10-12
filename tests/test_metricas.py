"""Prueba 6 de la guía: métricas con valores conocidos."""

import numpy as np
import pandas as pd
import pytest

from evaluacion.metricas import calcular_metricas, mae_por_dia, metricas_archivo


def test_metricas_valores_conocidos():
    m = calcular_metricas([1, 2, 3, 4], [1, 2, 3, 5])
    # error = [0,0,0,-1] -> RMSE = sqrt(1/4) = 0.5 ; MAE = 0.25
    # SS_tot = 5 ; SS_res = 1 -> R2 = 0.8
    assert m["RMSE"] == pytest.approx(0.5)
    assert m["MAE"] == pytest.approx(0.25)
    assert m["R2"] == pytest.approx(0.8)


def test_prediccion_perfecta_y_constante():
    assert calcular_metricas([1, 2, 3], [1, 2, 3]) == {"RMSE": 0, "MAE": 0, "R2": 1}
    m = calcular_metricas([2, 2, 2], [2, 2, 2])
    assert np.isnan(m["R2"])                          # varianza cero: R2 no definido


def test_mae_por_dia():
    real = np.zeros(6)
    pred = np.array([1, 1, 2, 2, 3, 3.0])
    assert mae_por_dia(real, pred, muestras_por_dia=2) == [1.0, 2.0, 3.0]


def test_metricas_archivo_excluye_entrenamiento():
    res = pd.DataFrame({"real": [0, 0, 0, 0.0], "simulada": [9, 9, 1, -1.0],
                        "entrenamiento": [True, True, False, False], "mes": ["a", "a", "a", "b"]})
    total, por_mes = metricas_archivo(res)
    assert total["MAE"] == 1.0
    assert set(por_mes) == {"a", "b"}
