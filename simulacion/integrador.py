"""
Integración de la ecuación identificada usando las L y H medidas.

Dos integradores, conservados por separado porque cada flujo usa el suyo:

  simular_modelo()   invernadero_profesor.py
      solve_ivp (RK45) en variables NORMALIZADAS, evaluando el lado derecho
      con modelo.predict() e interpolando linealmente L y H entre muestras.
      Acepta muestreo irregular.

  simular_lsim()     invernadero_20dias.py (antes: simular)
      scipy.signal.lsim del sistema lineal en variables FÍSICAS
      (A = aT, B = [a0, aL, aH], entradas [1, L, H]). lsim también
      interpola linealmente las entradas. Supone malla UNIFORME de paso dt.

Para el mismo modelo las dos dan la misma trayectoria salvo error numérico
(tests/test_simulacion.py lo comprueba), pero no son intercambiables: RK45
con un modelo rígido o inestable es mucho más lento, y lsim exige paso fijo.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import lsim

from analisis.derivadas import tiempo_relativo


def simular_modelo(modelo, escalador, datos, unidad: str = "s") -> np.ndarray:
    """datos: DataFrame con 'datetime_utc', Temperatura, Luminosidad, Humedad."""
    tiempo = tiempo_relativo(datos["datetime_utc"], unidad)

    L = datos["Luminosidad"].values.astype(float)
    H = datos["Humedad"].values.astype(float)
    T_real = datos["Temperatura"].values.astype(float)

    Ln = (L - escalador.media["Luminosidad"]) / escalador.std["Luminosidad"]
    Hn = (H - escalador.media["Humedad"]) / escalador.std["Humedad"]
    T0n = (T_real[0] - escalador.media["Temperatura"]) / escalador.std["Temperatura"]

    def rhs(t, y):
        u = np.array([[np.interp(t, tiempo, Ln), np.interp(t, tiempo, Hn)]])
        derivada = modelo.predict(np.array([[y[0]]]), u=u)
        return [float(derivada[0, 0])]

    solucion = solve_ivp(rhs, (tiempo[0], tiempo[-1]), [T0n], t_eval=tiempo, method="RK45")
    if not solucion.success:
        raise RuntimeError("La integración del modelo falló:\n" + solucion.message)

    return solucion.y[0] * escalador.std["Temperatura"] + escalador.media["Temperatura"]


def simular_lsim(coef: dict, bloque, dt: float) -> np.ndarray:
    """bloque: DataFrame en malla uniforme de paso dt (misma unidad que coef)."""
    U = np.c_[np.ones(len(bloque)), bloque["Luminosidad"], bloque["Humedad"]]
    t = np.arange(len(bloque)) * dt
    sistema = (
        np.array([[coef["aT"]]]),
        np.array([[coef["a0"], coef["aL"], coef["aH"]]]),
        np.array([[1.0]]),
        np.zeros((1, 3)),
    )
    _, T, _ = lsim(sistema, U, t, X0=[bloque["Temperatura"].iloc[0]])
    return T
