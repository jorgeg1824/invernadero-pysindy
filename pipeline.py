"""
Coordina las funcionalidades del proyecto. Cada una es un flujo de
etapas que solo llama a los módulos de datos/, analisis/, modelos/,
simulacion/, evaluacion/ y visualizacion/; los flujos están en flujos/:

  barrido      flujos/barrido.py      invernadero_profesor.py   estrategia C (+A)
  prediccion   flujos/prediccion.py   invernadero_20dias.py     estrategias A + B
  ventanas     flujos/ventanas.py     validar_modelo.py         A + B en varias ventanas
  estrategias  flujos/estrategias.py  nuevo                     A, B y C con los mismos datos

Las 8 etapas de la guía son:
  1 datos originales  2 transformación  3 alineación  4 limpieza
  5 selección del tramo  6 preparación del modelo  7 identificación
  8 simulación y evaluación
Las etapas 1 a 3 son comunes (flujos/comun.py).

Cada función devuelve una evaluacion.reportes.Ejecucion con las etapas y,
si guardar=True, escribe todo en resultados/<fecha>_<comando>/.

Uso desde Python:

    import pipeline
    ej = pipeline.ejecutar("prediccion")
    print(ej.resultado["resumen"])
    ej.etapa(5).texto            # por qué se eligió el tramo
"""

from __future__ import annotations

import config as C
from flujos.barrido import ejecutar_barrido
from flujos.estrategias import comparar_estrategias
from flujos.prediccion import evaluar_prediccion, ejecutar_prediccion
from flujos.ventanas import ejecutar_ventanas

FUNCIONALIDADES = {
    "barrido": ejecutar_barrido,
    "prediccion": ejecutar_prediccion,
    "ventanas": ejecutar_ventanas,
    "estrategias": comparar_estrategias,
}

DESCRIPCIONES = {
    "barrido": "Barrido de threshold (invernadero_profesor.py)",
    "prediccion": "PySINDy + ajuste por simulación, predicción a N días (invernadero_20dias.py)",
    "ventanas": "Predicción en varias ventanas (validar_modelo.py)",
    "estrategias": "Compara las estrategias A, B y C con los mismos datos (nuevo)",
}


def ejecutar(nombre: str, ruta_csv=None, cfg=None, **kwargs):
    """Ejecuta una funcionalidad por su nombre con la configuración por defecto o la dada."""
    if nombre not in FUNCIONALIDADES:
        raise ValueError(f"Funcionalidad desconocida: {nombre}. Opciones: {list(FUNCIONALIDADES)}")
    return FUNCIONALIDADES[nombre](ruta_csv=ruta_csv, cfg=cfg, **kwargs)


__all__ = ["FUNCIONALIDADES", "DESCRIPCIONES", "ejecutar", "ejecutar_barrido", "ejecutar_prediccion",
           "ejecutar_ventanas", "comparar_estrategias", "evaluar_prediccion", "C"]
