"""
Normalización z-score de T, L y H.

Origen: clase Escalador de invernadero_profesor.py. invernadero_20dias.py
usaba la misma clase (prof.Escalador), por eso es compartida.

Media y desviación se calculan SOLO con el entrenamiento y se reutilizan
para validar/simular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import VARIABLES


class Escalador:

    def __init__(self):
        self.media = {}
        self.std = {}

    def fit(self, datos):
        for variable in VARIABLES:
            x = datos[variable].values.astype(float)
            self.media[variable] = np.mean(x)
            s = np.std(x)
            if s < 1e-12:
                s = 1.0
            self.std[variable] = s
        return self

    def transform(self, datos):
        resultado = datos.copy()
        for variable in VARIABLES:
            resultado[variable] = (datos[variable] - self.media[variable]) / self.std[variable]
        return resultado

    # Nombres alternativos del original
    def transformar(self, datos):
        return self.transform(datos)

    def transformar_valor(self, variable, x):
        return (x - self.media[variable]) / self.std[variable]

    def inversa(self, variable, x):
        return x * self.std[variable] + self.media[variable]

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame({"media": self.media, "desviacion": self.std}).rename_axis("variable")
