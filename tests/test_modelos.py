"""Prueba 4 de la guía: identificación / recuperación de una ecuación conocida."""

import numpy as np
import pytest

from conftest import COEF_CONOCIDO, sistema_sintetico
from evaluacion.metricas import calcular_metricas
from modelos.coeficientes import a_fisico, ecuacion_fisica, obtener_coeficientes_fisicos, por_segundo
from modelos.estrategias.ajuste_simulacion import AjusteSimulacion
from modelos.estrategias.barrido_threshold import BarridoThreshold
from modelos.estrategias.identificacion_base import DerivadaMalla, IdentificacionBase
from simulacion.integrador import simular_lsim

DT = 5 / 60


@pytest.fixture(scope="module")
def datos():
    return sistema_sintetico(COEF_CONOCIDO, dias=8)


def _cerca(coef, rel=0.03):
    for k, v in COEF_CONOCIDO.items():
        assert coef[k] == pytest.approx(v, rel=rel), (k, coef[k], v)


def test_identificacion_base_recupera_la_ecuacion(datos):
    res = IdentificacionBase(1e-3, DerivadaMalla(5, 2, DT)).ejecutar(datos)
    _cerca(res.coef, rel=0.03)
    assert res.estable
    assert res.terminos_activos == ["constante", "T", "L", "H"]
    assert res.estrategia == "A_identificacion_base"
    assert res.datos_identificacion["muestras"] == len(datos)


def test_ajuste_por_simulacion_recupera_la_ecuacion(datos):
    base = IdentificacionBase(1e-3, DerivadaMalla(21, 2, DT)).ejecutar(datos)
    aj = AjusteSimulacion(lambda c, d: simular_lsim(c, d, DT)).ejecutar(datos, base=base)
    _cerca(aj.coef, rel=1e-4)
    assert aj.estrategia == "B_ajuste_por_simulacion" and aj.origen is base
    assert aj.coef["aT"] < 0


def test_ajuste_necesita_resultado_base(datos):
    with pytest.raises(ValueError):
        AjusteSimulacion(lambda c, d: simular_lsim(c, d, DT)).ejecutar(datos)


def test_barrido_elige_menor_rmse_y_registra_datos(datos):
    mitad = len(datos) // 2
    train, sel = datos.iloc[:mitad], datos.iloc[mitad:]
    estr = BarridoThreshold([1e-3, 5.0], lambda th: IdentificacionBase(th, DerivadaMalla(5, 2, DT)),
                            lambda r, d: simular_lsim(r.coef, d, DT), aviso=lambda s: None)
    res = estr.ejecutar(train, sel, "seleccion", independiente=True)
    tabla = {f["threshold"]: f for f in res.detalles["barrido"]}
    # Con threshold 5 se eliminan términos y el error crece: debe ganar 1e-3.
    assert tabla[5.0]["terminos"] < tabla[1e-3]["terminos"]
    assert res.parametros["threshold_elegido"] == 1e-3
    assert res.datos_seleccion["conjunto"] == "seleccion"
    assert res.detalles["seleccion_independiente"] is True


def test_conversion_de_coeficientes_equivalente(datos):
    """obtener_coeficientes_fisicos (profesor) == a_fisico (20dias)."""
    res = IdentificacionBase(1e-3, DerivadaMalla(5, 2, DT)).ejecutar(datos)
    a = obtener_coeficientes_fisicos(res.modelo, res.escalador)
    b = a_fisico(np.asarray(res.modelo.coefficients()).ravel(), res.escalador)
    assert a == pytest.approx(b, rel=1e-12)


def test_conversion_de_unidades_y_formato():
    c = por_segundo({"a0": 3600.0, "aT": -3600.0, "aL": 0.0, "aH": 36.0})
    assert c == {"a0": 1.0, "aT": -1.0, "aL": 0.0, "aH": 0.01}
    assert ecuacion_fisica(c) == "dT/dt = 1 + (-1)·T + (0)·L + (0.01)·H"


def test_modelo_identificado_predice_bien(datos):
    res = IdentificacionBase(1e-3, DerivadaMalla(5, 2, DT)).ejecutar(datos)
    m = calcular_metricas(datos["Temperatura"], simular_lsim(res.coef, datos, DT))
    assert m["R2"] > 0.99
