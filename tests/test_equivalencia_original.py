"""
La versión modular reproduce los resultados guardados de los programas
originales (referencia_original/). Usa el CSV completo: ~1 minuto.

    pytest -m lenta          solo estas
    pytest -m "not lenta"    todo menos estas
"""

import pandas as pd
import pytest

from comparacion import comparar_con_original as comp

pytestmark = pytest.mark.lenta


def _sin_fallas(filas):
    malas = [f for f in filas if not f["ok"]]
    assert not malas, pd.DataFrame(malas).to_string()


@pytest.mark.skipif(comp.unidad_efectiva("original") != "ks",
                    reason="la referencia principal se generó con pandas 3")
def test_barrido_igual_al_programa_del_profesor():
    _sin_fallas(comp.comparar_barrido(comp.REF, "original", "original"))


def test_barrido_corregido_igual_al_profesor_con_pandas2():
    _sin_fallas(comp.comparar_barrido(comp.REF_PANDAS2, "s", "s"))


def test_prediccion_igual_a_invernadero_20dias():
    _sin_fallas(comp.comparar_prediccion(comp.REF))


def test_ventanas_igual_a_validar_modelo():
    _sin_fallas(comp.comparar_ventanas(comp.REF))
