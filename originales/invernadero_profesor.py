import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector

from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

from zoneinfo import ZoneInfo

import pysindy as ps


# ============================================================
# CONFIGURACIÓN
# ============================================================

VARIABLES = ["Temperatura", "Luminosidad", "Humedad"]

VARIABLES_ALIASES = {
    "Temperatura": ["Temperatura", "temperatura", "temperature"],
    "Luminosidad": ["Luminosidad", "luminosidad", "luz", "lux"],
    "Humedad": ["Humedad", "humedad", "HR", "hr"],
}

# Modelo:
#
#       dT/dt = f(T, L, H)
#
# Primer orden -> derivada primera
# Primer grado -> polinomio lineal
#
#       dT/dt = c0 + cT*T + cL*L + cH*H


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def encontrar_columna(df, variable):
    """
    Busca la columna real correspondiente a una variable.
    """

    if variable in df.columns:
        return variable

    for alias in VARIABLES_ALIASES[variable]:
        if alias in df.columns:
            return alias

    # búsqueda ignorando mayúsculas/minúsculas
    columnas_lower = {str(c).lower(): c for c in df.columns}

    for alias in VARIABLES_ALIASES[variable]:
        if alias.lower() in columnas_lower:
            return columnas_lower[alias.lower()]

    return None


def detectar_columnas(df):
    """
    Detecta las columnas necesarias.
    """

    resultado = {}

    for variable in VARIABLES:
        columna = encontrar_columna(df, variable)

        if columna is None:
            raise ValueError(
                f"No se encontró la variable '{variable}'.\n\n"
                f"Columnas disponibles:\n{list(df.columns)}"
            )

        resultado[variable] = columna

    return resultado


def cargar_csv(ruta):
    """
    Carga el CSV.
    """

    # Intentamos primero TSV porque ese era el formato observado
    try:
        df = pd.read_csv(ruta, sep="\t")
    except Exception:
        df = pd.read_csv(ruta)

    if "datetime_utc" not in df.columns:
        raise ValueError(
            "No se encontró la columna 'datetime_utc'."
        )

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"],
        utc=True,
        errors="coerce"
    )

    if "metric_name" in df.columns and "number_value" in df.columns:
        columnas = {variable: "number_value" for variable in VARIABLES}
    else:
        columnas = detectar_columnas(df)

    # Si existe number_value usamos esa columna
    if "number_value" in df.columns:

        for variable, columna in columnas.items():

            if columna == "number_value":
                continue

            # Caso normal: metric_name + number_value
            pass

    return df, columnas


# ============================================================
# CONVERSIÓN DEL FORMATO LONG A SERIES
# ============================================================

def construir_series(df):
    """
    Convierte el formato:

        datetime_utc | metric_name | number_value

    en:

        datetime_utc | Temperatura | Humedad | Luminosidad
    """

    if "metric_name" not in df.columns:
        raise ValueError(
            "Se esperaba una columna 'metric_name'."
        )

    if "number_value" not in df.columns:
        raise ValueError(
            "Se esperaba una columna 'number_value'."
        )

    datos = df.copy()

    datos["datetime_utc"] = pd.to_datetime(
        datos["datetime_utc"],
        utc=True,
        errors="coerce"
    )

    datos["number_value"] = pd.to_numeric(
        datos["number_value"],
        errors="coerce"
    )

    datos = datos.dropna(
        subset=["datetime_utc", "number_value"]
    )

    resultado = {}

    for variable in VARIABLES:

        aliases = VARIABLES_ALIASES[variable]

        mascara = datos["metric_name"].astype(str).str.lower().isin(
            [a.lower() for a in aliases]
        )

        temp = datos.loc[
            mascara,
            ["datetime_utc", "number_value"]
        ].copy()

        if temp.empty:
            raise ValueError(
                f"No se encontraron datos para {variable}"
            )

        temp = temp.sort_values("datetime_utc")

        temp = temp.rename(
            columns={
                "number_value": variable
            }
        )

        resultado[variable] = temp

    return resultado


# ============================================================
# ALINEACIÓN TEMPORAL
# ============================================================

def estimar_dt_median(series):
    """
    Calcula el intervalo temporal típico.
    """

    t = series["datetime_utc"].sort_values()

    dt = t.diff().dt.total_seconds()

    dt = dt[
        (dt > 0) &
        np.isfinite(dt)
    ]

    if len(dt) == 0:
        return np.nan

    return float(dt.median())


def alinear_variables(series):
    """
    Usa Temperatura como referencia temporal.

    Humedad y Luminosidad se asocian mediante nearest timestamp.
    No se genera una nueva malla temporal.
    """

    T = series["Temperatura"].copy()
    L = series["Luminosidad"].copy()
    H = series["Humedad"].copy()

    T = T.sort_values("datetime_utc")
    L = L.sort_values("datetime_utc")
    H = H.sort_values("datetime_utc")

    dt_T = estimar_dt_median(T)
    dt_L = estimar_dt_median(L)
    dt_H = estimar_dt_median(H)

    dt_referencia = np.nanmedian(
        [dt_T, dt_L, dt_H]
    )

    if not np.isfinite(dt_referencia):
        raise ValueError(
            "No fue posible determinar el intervalo temporal."
        )

    # Tolerancia relativamente conservadora
    tolerancia = pd.Timedelta(
        seconds=2 * dt_referencia
    )

    datos = pd.merge_asof(
        T,
        L,
        on="datetime_utc",
        direction="nearest",
        tolerance=tolerancia
    )

    datos = pd.merge_asof(
        datos.sort_values("datetime_utc"),
        H,
        on="datetime_utc",
        direction="nearest",
        tolerance=tolerancia
    )

    datos = datos.sort_values("datetime_utc")

    # Elimina timestamps duplicados: PySINDy y np.gradient requieren
    # un vector temporal estrictamente creciente.
    duplicados = datos["datetime_utc"].duplicated().sum()
    datos = datos.drop_duplicates(
        subset=["datetime_utc"],
        keep="first"
    ).reset_index(drop=True)
    datos.attrs["timestamps_duplicados_eliminados"] = int(duplicados)

    return datos, dt_referencia


# ============================================================
# DETECCIÓN DE CORTES
# ============================================================

def obtener_bloques_continuos(datos, factor=4.0):
    """
    Divide los datos cuando aparece un salto temporal grande.

    Un salto se considera grande si:

        dt > factor * dt_mediano
    """

    datos = datos.copy()

    datos["dt"] = (
        datos["datetime_utc"]
        .diff()
        .dt.total_seconds()
    )

    dt_validos = datos["dt"].dropna()

    if len(dt_validos) == 0:
        return []

    dt_mediana = dt_validos.median()

    datos["corte"] = (
        datos["dt"] > factor * dt_mediana
    )

    datos["bloque"] = datos["corte"].cumsum()

    bloques = []

    for _, grupo in datos.groupby("bloque"):

        grupo = grupo.drop(columns=["dt", "corte", "bloque"])

        if len(grupo) < 20:
            continue

        inicio = grupo["datetime_utc"].iloc[0]
        fin = grupo["datetime_utc"].iloc[-1]

        duracion = (
            fin - inicio
        ).total_seconds()

        bloques.append(
            {
                "datos": grupo,
                "inicio": inicio,
                "fin": fin,
                "duracion": duracion,
                "n": len(grupo),
            }
        )

    return bloques


# ============================================================
# MÉTRICAS DE EXCITACIÓN
# ============================================================

def calcular_excitación(datos):
    """
    Mide cuánto cambian las variables.

    No queremos solamente un tramo largo.
    Queremos un tramo donde el sistema tenga dinámica.
    """

    resultado = {}

    for variable in VARIABLES:

        x = pd.to_numeric(
            datos[variable],
            errors="coerce"
        ).dropna()

        if len(x) < 2:
            resultado[variable] = 0.0
            continue

        rango = x.max() - x.min()
        std = x.std()

        # Coeficiente de variación robusto
        escala = max(abs(x.median()), 1e-9)

        resultado[variable] = (
            0.5 * std / escala +
            0.5 * rango / escala
        )

    # Damos mayor peso a T porque es nuestro estado
    score = (
        0.5 * resultado["Temperatura"] +
        0.25 * resultado["Luminosidad"] +
        0.25 * resultado["Humedad"]
    )

    return score


def puntuar_bloque(bloque):
    """
    Combina duración y excitación.
    """

    datos = bloque["datos"]

    if len(datos) < 20:
        return -np.inf

    excitacion = calcular_excitación(datos)

    duracion_horas = (
        bloque["duracion"] / 3600.0
    )

    # Log evita que un tramo extremadamente largo
    # domine completamente.
    score_duracion = np.log1p(
        max(duracion_horas, 0)
    )

    score = (
        0.55 * score_duracion +
        0.45 * excitacion
    )

    return score


# ============================================================
# BUSCAR MEJOR TRAMO
# ============================================================

def seleccionar_mejor_bloque(datos, dias_minimos=6):
    """
    Busca el mejor bloque continuo.
    """

    bloques = obtener_bloques_continuos(datos)

    if not bloques:
        raise ValueError(
            "No se encontraron bloques continuos."
        )

    bloques_validos = [
        bloque for bloque in bloques
        if bloque["duracion"] >= dias_minimos * 86400
    ]

    if not bloques_validos:
        max_dias = max(b["duracion"] for b in bloques) / 86400.0
        raise ValueError(
            f"No existe un bloque continuo de {dias_minimos} días.\n\n"
            f"El bloque continuo más largo tiene {max_dias:.2f} días."
        )

    bloques = bloques_validos

    for bloque in bloques:
        bloque["excitación"] = calcular_excitación(
            bloque["datos"]
        )

        bloque["score"] = puntuar_bloque(
            bloque
        )

    bloques = sorted(
        bloques,
        key=lambda x: x["score"],
        reverse=True
    )

    return bloques[0], bloques


# ============================================================
# LIMPIEZA DEL TRAMO
# ============================================================

def limpiar_tramo(datos):
    """
    Elimina NaN y saltos temporales excesivos.
    """

    datos = datos.copy()

    datos = datos.dropna(
        subset=VARIABLES
    )

    datos = datos.sort_values(
        "datetime_utc"
    )

    datos = datos.drop_duplicates(
        subset=["datetime_utc"],
        keep="first"
    ).reset_index(drop=True)

    t = (
        datos["datetime_utc"]
        .astype("int64") / 1e9
    )

    dt = np.diff(t)

    if len(dt) == 0:
        return datos

    dt_mediana = np.median(dt)

    mascara = np.ones(
        len(datos),
        dtype=bool
    )

    mascara[1:] = (
        dt <= 5 * dt_mediana
    )

    return datos.loc[mascara].copy()


# ============================================================
# NORMALIZACIÓN
# ============================================================

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

    def transform(self, datos):

        resultado = datos.copy()

        for variable in VARIABLES:

            resultado[variable] = (
                datos[variable]
                - self.media[variable]
            ) / self.std[variable]

        return resultado

    def transformar(self, datos):
        return self.transform(datos)

    def transformar_valor(self, variable, x):

        return (
            x - self.media[variable]
        ) / self.std[variable]

    def inversa(self, variable, x):

        return (
            x * self.std[variable]
            + self.media[variable]
        )


# ============================================================
# DERIVADA SUAVIZADA
# ============================================================

def calcular_derivada_temperatura(datos):
    """
    Estima dT/dt con Savitzky-Golay.

    Como el muestreo puede no ser perfectamente uniforme,
    utilizamos el tiempo real posteriormente para convertir
    la derivada respecto al índice en derivada respecto al tiempo.
    """

    T = datos["Temperatura"].values.astype(float)

    t = (
        datos["datetime_utc"]
        .astype("int64")
        .values / 1e9
    )

    n = len(T)

    if n < 7:
        raise ValueError(
            "Muy pocos datos para calcular la derivada."
        )

    # Ventana impar
    ventana = min(21, n)

    if ventana % 2 == 0:
        ventana -= 1

    if ventana < 5:
        ventana = 5

    # Suavizamos temperatura
    T_suave = savgol_filter(
        T,
        window_length=ventana,
        polyorder=2
    )

    # Derivada respecto al tiempo real
    orden = np.argsort(t, kind="stable")
    t = t[orden]
    T_suave = T_suave[orden]
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("El tiempo no es estrictamente creciente después de limpiar duplicados.")

    dTdt = np.gradient(
        T_suave,
        t
    )

    return T_suave, dTdt


# ============================================================
# PREPARAR TRAIN / VALIDACIÓN
# ============================================================

def dividir_train_validacion(datos, dias_entrenamiento=5):

    datos = datos.sort_values("datetime_utc").copy()
    datos = datos.drop_duplicates(
        subset=["datetime_utc"], keep="first"
    ).reset_index(drop=True)

    inicio = datos["datetime_utc"].iloc[0]
    limite = inicio + pd.Timedelta(days=dias_entrenamiento)

    train = datos[datos["datetime_utc"] < limite].copy()
    validacion = datos[datos["datetime_utc"] >= limite].copy()

    if len(train) < 20:
        raise ValueError("El tramo de entrenamiento contiene muy pocos datos.")
    if len(validacion) < 20:
        raise ValueError(
            f"No quedan suficientes datos posteriores a los {dias_entrenamiento} días de entrenamiento para validar."
        )

    return train, validacion

# ============================================================
# CREAR MODELO SINDY
# ============================================================

def crear_modelo(threshold):

    library = ps.PolynomialLibrary(
        degree=1,
        include_bias=True
    )

    optimizer = ps.STLSQ(
        threshold=threshold,
        alpha=0.001,
        normalize_columns=True
    )

    # No usamos la diferenciación interna porque calcularemos
    # dT/dt explícitamente con Savitzky-Golay.
    modelo = ps.SINDy(
        optimizer=optimizer,
        feature_library=library
    )

    return modelo


# ============================================================
# IDENTIFICAR MODELO
# ============================================================

def identificar_modelo(train, threshold):

    escalador = Escalador()
    escalador.fit(train)

    train_n = escalador.transform(
        train
    )

    # Estado
    T = train_n[
        ["Temperatura"]
    ].values

    # Entradas
    U = train_n[
        ["Luminosidad", "Humedad"]
    ].values

    # Derivada física de T
    _, dTdt = calcular_derivada_temperatura(
        train
    )

    # Como T fue normalizada:
    #
    # dT_normalizada/dt = dT/dt / std(T)

    dTdt_n = (
        dTdt /
        escalador.std["Temperatura"]
    )

    modelo = crear_modelo(
        threshold
    )

    # Le damos explícitamente la derivada.
    #
    # X contiene T
    # u contiene L y H
    #
    tiempo_train = (
        train["datetime_utc"]
        .astype("int64")
        .values / 1e9
    )

    if np.any(np.diff(tiempo_train) <= 0):
        raise ValueError("El tiempo de entrenamiento no es estrictamente creciente.")

    modelo.fit(
        T,
        u=U,
        t=tiempo_train,
        x_dot=dTdt_n
    )

    return modelo, escalador


# ============================================================
# ECUACIÓN FÍSICA
# ============================================================

def obtener_coeficientes_fisicos(
    modelo,
    escalador
):
    """
    Convierte:

        dTn/dt =
            c0 + cT*Tn + cL*Ln + cH*Hn

    en:

        dT/dt =
            a0 + aT*T + aL*L + aH*H
    """

    coef = np.asarray(
        modelo.coefficients()
    ).flatten()

    nombres = modelo.get_feature_names()

    coef_dict = dict(
        zip(nombres, coef)
    )

    c0 = coef_dict.get("1", 0.0)

    cT = coef_dict.get(
        "x0",
        0.0
    )

    cL = coef_dict.get(
        "u0",
        0.0
    )

    cH = coef_dict.get(
        "u1",
        0.0
    )

    sT = escalador.std["Temperatura"]
    sL = escalador.std["Luminosidad"]
    sH = escalador.std["Humedad"]

    mT = escalador.media["Temperatura"]
    mL = escalador.media["Luminosidad"]
    mH = escalador.media["Humedad"]

    # Transformación:
    #
    # dT/dt = sT * dTn/dt

    aT = cT

    aL = (
        sT * cL / sL
    )

    aH = (
        sT * cH / sH
    )

    a0 = (
        sT * c0
        - cT * mT
        - sT * cL * mL / sL
        - sT * cH * mH / sH
    )

    return {
        "a0": a0,
        "aT": aT,
        "aL": aL,
        "aH": aH
    }


def ecuacion_fisica(coeficientes):

    a0 = coeficientes["a0"]
    aT = coeficientes["aT"]
    aL = coeficientes["aL"]
    aH = coeficientes["aH"]

    return (
        f"dT/dt = "
        f"{a0:.8g} "
        f"+ ({aT:.8g})·T "
        f"+ ({aL:.8g})·L "
        f"+ ({aH:.8g})·H"
    )


# ============================================================
# SIMULACIÓN DEL MODELO
# ============================================================

def simular_modelo(
    modelo,
    escalador,
    datos
):
    """
    Simula:

        dT/dt = f(T,L,H)

    usando luminosidad y humedad medidas.
    """

    tiempo = (
        datos["datetime_utc"]
        .astype("int64")
        .values / 1e9
    )

    tiempo = tiempo - tiempo[0]

    L = datos["Luminosidad"].values.astype(float)
    H = datos["Humedad"].values.astype(float)

    T_real = datos["Temperatura"].values.astype(float)

    # Normalización
    Ln = (
        L - escalador.media["Luminosidad"]
    ) / escalador.std["Luminosidad"]

    Hn = (
        H - escalador.media["Humedad"]
    ) / escalador.std["Humedad"]

    T0n = (
        T_real[0]
        - escalador.media["Temperatura"]
    ) / escalador.std["Temperatura"]

    def interpolar(t, valores):

        return np.interp(
            t,
            tiempo,
            valores
        )

    def rhs(t, y):

        Tn = y[0]

        Ln_t = interpolar(
            t,
            Ln
        )

        Hn_t = interpolar(
            t,
            Hn
        )

        entrada = np.array(
            [[Tn]]
        )

        u = np.array(
            [[Ln_t, Hn_t]]
        )

        derivada = modelo.predict(
            entrada,
            u=u
        )

        return [
            float(derivada[0, 0])
        ]

    solucion = solve_ivp(
        rhs,
        (
            tiempo[0],
            tiempo[-1]
        ),
        [T0n],
        t_eval=tiempo,
        method="RK45"
    )

    if not solucion.success:
        raise RuntimeError(
            "La integración del modelo falló:\n"
            + solucion.message
        )

    T_pred_n = solucion.y[0]

    T_pred = (
        T_pred_n *
        escalador.std["Temperatura"]
        + escalador.media["Temperatura"]
    )

    return T_pred


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(real, pred):

    real = np.asarray(real)
    pred = np.asarray(pred)

    error = real - pred

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    mae = np.mean(
        np.abs(error)
    )

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (real - np.mean(real)) ** 2
    )

    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# ============================================================
# PROBAR VARIOS THRESHOLDS
# ============================================================

def buscar_mejor_modelo(train, validacion):

    thresholds = [
        0.00001,
        0.00003,
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01
    ]

    resultados = []

    mejor = None

    for threshold in thresholds:

        try:

            modelo, escalador = identificar_modelo(
                train,
                threshold
            )

            T_pred = simular_modelo(
                modelo,
                escalador,
                validacion
            )

            T_real = validacion[
                "Temperatura"
            ].values

            metricas = calcular_metricas(
                T_real,
                T_pred
            )

            # Número de términos activos
            coef = np.asarray(
                modelo.coefficients()
            )

            n_activos = np.sum(
                np.abs(coef) > 1e-10
            )

            resultado = {
                "threshold": threshold,
                "RMSE": metricas["RMSE"],
                "MAE": metricas["MAE"],
                "R2": metricas["R2"],
                "terminos": int(n_activos)
            }

            resultados.append(
                resultado
            )

            print(
                f"threshold={threshold:.5g} "
                f"RMSE={metricas['RMSE']:.5f} "
                f"R2={metricas['R2']:.5f} "
                f"terminos={n_activos}"
            )

            # Preferimos menor RMSE
            if (
                mejor is None
                or metricas["RMSE"] < mejor["metricas"]["RMSE"]
            ):

                mejor = {
                    "modelo": modelo,
                    "escalador": escalador,
                    "threshold": threshold,
                    "pred": T_pred,
                    "metricas": metricas
                }

        except Exception as e:

            print(
                f"Threshold {threshold} falló: {e}"
            )

    if mejor is None:
        raise RuntimeError(
            "Ninguno de los modelos pudo ser identificado."
        )

    return mejor, resultados


# ============================================================
# INTERFAZ
# ============================================================

class Aplicacion:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Identificación PySINDy - Invernadero aeropónico"
        )

        self.root.geometry(
            "1300x850"
        )

        self.df = None
        self.series = None
        self.datos = None

        self.mejor_bloque = None
        self.bloques = None

        self.fig = None
        self.ax = None
        self.canvas = None
        self.span = None

        self.variables_var = {}

        self.crear_interfaz()


    # ========================================================
    # INTERFAZ
    # ========================================================

    def crear_interfaz(self):

        panel = ttk.Frame(
            self.root,
            padding=10
        )

        panel.pack(
            side=tk.LEFT,
            fill=tk.Y
        )

        grafico = ttk.Frame(
            self.root
        )

        grafico.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # Botón abrir
        # ----------------------------------------------------

        ttk.Button(
            panel,
            text="Abrir CSV",
            command=self.abrir_csv
        ).pack(
            fill=tk.X,
            pady=5
        )

        ttk.Button(
            panel,
            text="Mostrar todo",
            command=self.mostrar_todo
        ).pack(
            fill=tk.X,
            pady=5
        )

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        ttk.Label(
            panel,
            text="Variables"
        ).pack(
            pady=(20, 5)
        )

        for variable in VARIABLES:

            var = tk.BooleanVar(
                value=True
            )

            self.variables_var[
                variable
            ] = var

            ttk.Checkbutton(
                panel,
                text=variable,
                variable=var,
                command=self.actualizar_grafica
            ).pack(
                anchor="w"
            )

        # ----------------------------------------------------
        # Detección
        # ----------------------------------------------------

        ttk.Label(
            panel,
            text="Identificación"
        ).pack(
            pady=(20, 5)
        )

        ttk.Button(
            panel,
            text="Buscar mejor tramo",
            command=self.buscar_tramo
        ).pack(
            fill=tk.X,
            pady=5
        )

        ttk.Button(
            panel,
            text="Ejecutar PySINDy",
            command=self.ejecutar_sindy
        ).pack(
            fill=tk.X,
            pady=5
        )

        # ----------------------------------------------------
        # Información
        # ----------------------------------------------------

        self.info = tk.Text(
            panel,
            width=42,
            height=25
        )

        self.info.pack(
            fill=tk.BOTH,
            expand=True,
            pady=10
        )

        # ----------------------------------------------------
        # Gráfico
        # ----------------------------------------------------

        self.fig, self.ax = plt.subplots(
            figsize=(9, 6)
        )

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=grafico
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )


    # ========================================================
    # ABRIR CSV
    # ========================================================

    def abrir_csv(self):

        ruta = filedialog.askopenfilename(
            filetypes=[
                ("CSV", "*.csv"),
                ("Todos", "*.*")
            ]
        )

        if not ruta:
            return

        try:

            df, columnas = cargar_csv(
                ruta
            )

            self.df = df

            self.series = construir_series(
                df
            )

            self.datos, dt = alinear_variables(
                self.series
            )

            self.info.delete(
                "1.0",
                tk.END
            )

            self.info.insert(
                tk.END,
                "ARCHIVO CARGADO\n\n"
            )

            self.info.insert(
                tk.END,
                f"Filas originales: {len(df)}\n"
            )

            self.info.insert(
                tk.END,
                f"Filas alineadas: {len(self.datos)}\n"
            )

            duplicados = self.datos.attrs.get("timestamps_duplicados_eliminados", 0)
            self.info.insert(
                tk.END,
                f"dt típico: {dt:.3f} s\n"
                f"Timestamps duplicados eliminados: {duplicados}\n\n"
            )

            for variable in VARIABLES:

                self.info.insert(
                    tk.END,
                    f"{variable}: "
                    f"{len(self.series[variable])} muestras\n"
                )

            self.actualizar_grafica()

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )


    # ========================================================
    # MOSTRAR TODO
    # ========================================================

    def mostrar_todo(self):

        if self.datos is None:
            return

        self.actualizar_grafica(
            self.datos
        )


    # ========================================================
    # ACTUALIZAR GRÁFICA
    # ========================================================

    def actualizar_grafica(
        self,
        datos=None
    ):

        if self.datos is None:
            return

        if datos is None:
            datos = self.datos

        self.ax.clear()

        for variable in VARIABLES:

            if not self.variables_var[
                variable
            ].get():
                continue

            self.ax.plot(
                datos["datetime_utc"],
                datos[variable],
                label=variable,
                linewidth=0.8
            )

        self.ax.set_xlabel(
            "Tiempo"
        )

        self.ax.set_ylabel(
            "Valor"
        )

        self.ax.grid(
            True,
            alpha=0.3
        )

        self.ax.legend()

        self.fig.autofmt_xdate()

        self.canvas.draw()


    # ========================================================
    # BUSCAR TRAMO
    # ========================================================

    def buscar_tramo(self):

        if self.datos is None:
            messagebox.showwarning(
                "Aviso",
                "Primero abra un CSV."
            )
            return

        try:

            mejor, bloques = seleccionar_mejor_bloque(
                self.datos,
                dias_minimos=6
            )

            self.mejor_bloque = mejor
            self.bloques = bloques

            datos = mejor["datos"]

            self.actualizar_grafica(
                datos
            )

            self.info.delete(
                "1.0",
                tk.END
            )

            self.info.insert(
                tk.END,
                "MEJOR TRAMO ENCONTRADO\n"
                "=======================\n\n"
            )

            self.info.insert(
                tk.END,
                f"Inicio:\n"
                f"{mejor['inicio']}\n\n"
            )

            self.info.insert(
                tk.END,
                f"Fin:\n"
                f"{mejor['fin']}\n\n"
            )

            self.info.insert(
                tk.END,
                f"Duración:\n"
                f"{mejor['duracion']/3600:.2f} horas\n\n"
            )

            self.info.insert(
                tk.END,
                f"Muestras:\n"
                f"{mejor['n']}\n\n"
            )

            self.info.insert(
                tk.END,
                f"Excitación:\n"
                f"{mejor['excitación']:.5f}\n\n"
            )

            self.info.insert(
                tk.END,
                f"Score:\n"
                f"{mejor['score']:.5f}\n\n"
            )

            self.info.insert(
                tk.END,
                f"Bloques encontrados: "
                f"{len(bloques)}\n"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )


    # ========================================================
    # EJECUTAR SINDY
    # ========================================================

    def ejecutar_sindy(self):

        if self.mejor_bloque is None:

            messagebox.showwarning(
                "Aviso",
                "Primero ejecuta 'Buscar mejor tramo'."
            )

            return

        try:

            datos = limpiar_tramo(
                self.mejor_bloque["datos"]
            )

            train, validacion = dividir_train_validacion(
                datos,
                dias_entrenamiento=5
            )

            self.info.delete(
                "1.0",
                tk.END
            )

            self.info.insert(
                tk.END,
                "EJECUTANDO PYSINDY\n"
                "===================\n\n"
            )

            self.info.insert(
                tk.END,
                f"Train: {len(train)} muestras\n"
            )

            duracion_train = (train["datetime_utc"].iloc[-1] - train["datetime_utc"].iloc[0]).total_seconds() / 86400.0
            duracion_validacion = (validacion["datetime_utc"].iloc[-1] - validacion["datetime_utc"].iloc[0]).total_seconds() / 86400.0
            self.info.insert(
                tk.END,
                f"Validación: {len(validacion)} muestras\n"
                f"Duración entrenamiento: {duracion_train:.2f} días\n"
                f"Duración validación: {duracion_validacion:.2f} días\n\n"
            )

            self.info.insert(
                tk.END,
                "Buscando threshold...\n\n"
            )

            self.root.update()

            mejor, resultados = buscar_mejor_modelo(
                train,
                validacion
            )

            modelo = mejor["modelo"]
            escalador = mejor["escalador"]

            coef = obtener_coeficientes_fisicos(
                modelo,
                escalador
            )

            ecuacion = ecuacion_fisica(
                coef
            )

            metricas = mejor["metricas"]

            # ------------------------------------------------
            # RESULTADOS
            # ------------------------------------------------

            self.info.insert(
                tk.END,
                "MODELO SELECCIONADO\n"
                "--------------------\n\n"
            )

            self.info.insert(
                tk.END,
                f"Threshold: "
                f"{mejor['threshold']}\n\n"
            )

            self.info.insert(
                tk.END,
                "Ecuación física:\n\n"
            )

            self.info.insert(
                tk.END,
                ecuacion + "\n\n"
            )

            self.info.insert(
                tk.END,
                "Validación:\n\n"
            )

            self.info.insert(
                tk.END,
                f"RMSE = "
                f"{metricas['RMSE']:.5f}\n"
            )

            self.info.insert(
                tk.END,
                f"MAE = "
                f"{metricas['MAE']:.5f}\n"
            )

            self.info.insert(
                tk.END,
                f"R² = "
                f"{metricas['R2']:.5f}\n\n"
            )

            self.info.insert(
                tk.END,
                "Modelo PySINDy:\n\n"
            )

            self.info.insert(
                tk.END,
                str(modelo)
                + "\n\n"
            )

            # ------------------------------------------------
            # TABLA DE THRESHOLDS
            # ------------------------------------------------

            self.info.insert(
                tk.END,
                "THRESHOLDS PROBADOS\n"
                "--------------------\n"
            )

            for resultado in resultados:

                self.info.insert(
                    tk.END,
                    f"{resultado['threshold']:.5g}   "
                    f"RMSE={resultado['RMSE']:.5f}   "
                    f"R²={resultado['R2']:.5f}   "
                    f"términos={resultado['terminos']}\n"
                )

            # ------------------------------------------------
            # GRÁFICA VALIDACIÓN
            # ------------------------------------------------

            T_pred = mejor["pred"]

            T_real = validacion[
                "Temperatura"
            ].values

            tiempo = validacion[
                "datetime_utc"
            ]

            self.ax.clear()

            self.ax.plot(
                tiempo,
                T_real,
                label="Temperatura real",
                linewidth=1.5
            )

            self.ax.plot(
                tiempo,
                T_pred,
                label="Temperatura predicha",
                linewidth=1.5
            )

            self.ax.axvline(
                tiempo.iloc[0],
                linestyle="--",
                linewidth=1
            )

            self.ax.set_title(
                "Validación PySINDy"
            )

            self.ax.set_xlabel(
                "Tiempo"
            )

            self.ax.set_ylabel(
                "Temperatura"
            )

            self.ax.grid(
                True,
                alpha=0.3
            )

            self.ax.legend()

            self.fig.autofmt_xdate()

            self.canvas.draw()

        except Exception as e:

            messagebox.showerror(
                "Error PySINDy",
                str(e)
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = Aplicacion(
        root
    )

    root.mainloop()
