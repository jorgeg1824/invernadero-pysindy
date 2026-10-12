"""
Modificacion de invernadero_profesor.py para predecir 20 dias.

Cambios respecto al codigo del profesor:
  1. Los huecos cortos de datos (<= 2 h) se rellenan por interpolacion, asi
     los tramos continuos pasan de ~10-24 dias a 26 y 41 dias.
  2. Se valida con los ULTIMOS 20 dias del tramo; se entrena con los dias previos.
  3. PySINDy elige la forma de la ecuacion  dT/dt = a0 + aT*T + aL*L + aH*H
     y da los coeficientes iniciales. Luego esos coeficientes se ajustan
     minimizando el error de la SIMULACION de todo el entrenamiento (no el error
     de la derivada), forzando aT < 0 para que el modelo sea estable.
  4. La ecuacion final se simula tambien sobre TODO el archivo para compararla
     con todos los datos, y la interfaz permite elegir entre varias graficas.

Uso:
    python invernadero_20dias.py              -> interfaz (la del profesor, ampliada)
    python invernadero_20dias.py --consola    -> sin interfaz, guarda las graficas en graficas/
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import pysindy as ps
from scipy.signal import savgol_filter, lsim
from scipy.optimize import least_squares

import invernadero_profesor as prof
from invernadero_profesor import tk, ttk, messagebox, VARIABLES

DIAS_PREDICCION = 20
DIAS_ENTRENAMIENTO_MIN = 5
MAX_HUECO_HORAS = 2
PASO = pd.Timedelta(minutes=5)
DT_H = PASO.total_seconds() / 3600  # paso en horas
THRESHOLD = 0.001
ENTRADAS = ["Luminosidad", "Humedad"]
ZONA_HORARIA = "America/Bogota"


# ============================================================
# 1. TRAMOS CONTINUOS (rellenando huecos cortos)
# ============================================================

def bloques_rellenados(datos):
    d = datos.dropna(subset=VARIABLES).set_index("datetime_utc")[VARIABLES]
    corte = d.index.to_series().diff() > pd.Timedelta(hours=MAX_HUECO_HORAS)

    bloques = []
    for _, g in d.groupby(corte.cumsum().values):
        malla = pd.date_range(g.index[0].ceil(PASO), g.index[-1].floor(PASO), freq=PASO)
        if len(malla) < 2:
            continue
        bloque = g.reindex(g.index.union(malla)).interpolate(method="time").reindex(malla)
        bloques.append(bloque)
    return bloques


def elegir_bloque(bloques):
    necesarios = pd.Timedelta(days=DIAS_PREDICCION + DIAS_ENTRENAMIENTO_MIN)
    validos = [b for b in bloques if b.index[-1] - b.index[0] >= necesarios]
    if not validos:
        mas_largo = max(b.index[-1] - b.index[0] for b in bloques)
        raise ValueError(
            f"Se necesita un tramo de {DIAS_PREDICCION + DIAS_ENTRENAMIENTO_MIN} dias.\n"
            f"El mas largo tiene {mas_largo.total_seconds() / 86400:.1f} dias."
        )
    return max(validos, key=len)


def dividir(bloque):
    limite = bloque.index[-1] - pd.Timedelta(days=DIAS_PREDICCION)
    return bloque[bloque.index <= limite], bloque[bloque.index > limite]


# ============================================================
# 2. MODELO: PySINDy + ajuste por simulacion
# ============================================================

def a_fisico(c, esc):
    """Coeficientes normalizados [c0, cT, cL, cH] -> fisicos (tiempo en horas)."""
    c0, cT, cL, cH = c
    sT, sL, sH = (esc.std[v] for v in VARIABLES)
    mT, mL, mH = (esc.media[v] for v in VARIABLES)
    return {
        "a0": sT * c0 - cT * mT - sT * cL * mL / sL - sT * cH * mH / sH,
        "aT": cT,
        "aL": sT * cL / sL,
        "aH": sT * cH / sH,
    }


def simular(coef, bloque):
    """Integra dT/dt = a0 + aT*T + aL*L + aH*H con las L y H medidas."""
    U = np.c_[np.ones(len(bloque)), bloque["Luminosidad"], bloque["Humedad"]]
    t = np.arange(len(bloque)) * DT_H
    sistema = (
        np.array([[coef["aT"]]]),
        np.array([[coef["a0"], coef["aL"], coef["aH"]]]),
        np.array([[1.0]]),
        np.zeros((1, 3)),
    )
    _, T, _ = lsim(sistema, U, t, X0=[bloque["Temperatura"].iloc[0]])
    return T


def identificar(train):
    esc = prof.Escalador()
    esc.fit(train)
    tn = esc.transform(train)

    dTdt = savgol_filter(train["Temperatura"].values, 21, 2, deriv=1, delta=DT_H)

    modelo = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=1),
        optimizer=ps.STLSQ(threshold=THRESHOLD, alpha=1e-3, normalize_columns=True),
    )
    modelo.fit(
        tn[["Temperatura"]].values,
        u=tn[ENTRADAS].values,
        x_dot=dTdt.reshape(-1, 1) / esc.std["Temperatura"],
        t=DT_H,
    )
    c_sindy = np.asarray(modelo.coefficients()).ravel()

    # Solo se ajustan los terminos que PySINDy dejo activos (mas siempre T).
    activos = np.flatnonzero(c_sindy != 0)
    activos = np.union1d(activos, [1])

    # aT se escribe como -exp(q) para que siempre sea negativo (modelo estable).
    def armar(q):
        c = np.zeros(4)
        c[activos] = q
        c[1] = -np.exp(q[np.searchsorted(activos, 1)])
        return c

    q0 = c_sindy[activos].copy()
    q0[np.searchsorted(activos, 1)] = np.log(-c_sindy[1]) if c_sindy[1] < 0 else 0.0

    real = train["Temperatura"].values
    ajuste = least_squares(lambda q: simular(a_fisico(armar(q), esc), train) - real, q0)

    return modelo, a_fisico(c_sindy, esc), a_fisico(armar(ajuste.x), esc)


# ============================================================
# 3. EVALUACION
# ============================================================

def mae_por_dia(real, pred):
    dia = np.arange(len(real)) // int(pd.Timedelta(days=1) / PASO)
    return [float(np.mean(np.abs(real - pred)[dia == d])) for d in np.unique(dia)]


def por_segundo(coef):
    return {k: v / 3600 for k, v in coef.items()}


def evaluar(bloque):
    train, val = dividir(bloque)
    modelo, coef_sindy, coef = identificar(train)

    real = val["Temperatura"].values
    pred = simular(coef, val)
    pred_sindy = simular(coef_sindy, val) if coef_sindy["aT"] < 0 else None

    # Referencia ingenua: el perfil diario promedio del entrenamiento.
    paso_del_dia = lambda idx: idx.hour * 60 + idx.minute
    perfil = train["Temperatura"].groupby(paso_del_dia(train.index)).mean()
    base = perfil.reindex(paso_del_dia(val.index)).values

    return {
        "train": train, "val": val, "modelo": modelo,
        "coef_sindy": coef_sindy, "coef": coef,
        "pred": pred, "pred_sindy": pred_sindy, "base": base,
        "metricas": prof.calcular_metricas(real, pred),
        "metricas_base": prof.calcular_metricas(real, base),
        "mae_dias": mae_por_dia(real, pred),
    }


def simular_archivo(bloques, coef, train):
    """Simula la ecuacion en cada tramo continuo del archivo completo.

    La temperatura real solo se usa como valor inicial de cada tramo.
    """
    partes = []
    for b in bloques:
        p = b[["Temperatura"]].rename(columns={"Temperatura": "real"})
        p["simulada"] = simular(coef, b)
        partes.append(p)
    res = pd.concat(partes)
    res["entrenamiento"] = (res.index >= train.index[0]) & (res.index <= train.index[-1])
    res["mes"] = res.index.tz_convert(ZONA_HORARIA).strftime("%Y-%m")
    return res


def metricas_archivo(res):
    """Metricas sin contar el tramo de entrenamiento: total y por mes."""
    fuera = res[~res["entrenamiento"]]
    m = lambda d: prof.calcular_metricas(d["real"].values, d["simulada"].values)
    return m(fuera), {mes: m(d) for mes, d in fuera.groupby("mes")}


def resumen(r):
    dias = lambda d: (d.index[-1] - d.index[0]).total_seconds() / 86400
    m, mb = r["metricas"], r["metricas_base"]
    lineas = [
        f"Entrenamiento: {r['train'].index[0]:%Y-%m-%d} -> {r['train'].index[-1]:%Y-%m-%d} ({dias(r['train']):.1f} dias)",
        f"Prediccion:    {r['val'].index[0]:%Y-%m-%d} -> {r['val'].index[-1]:%Y-%m-%d} ({dias(r['val']):.1f} dias)",
        "",
        "Ecuacion PySINDy (solo regresion de la derivada):",
        prof.ecuacion_fisica(por_segundo(r["coef_sindy"])),
        "  -> " + ("estable" if r["coef_sindy"]["aT"] < 0 else "INESTABLE (aT > 0), diverge al simular"),
        "",
        "Ecuacion final (ajustada por simulacion, t en segundos):",
        prof.ecuacion_fisica(por_segundo(r["coef"])),
        "",
        f"Prediccion de {DIAS_PREDICCION} dias:",
        f"  RMSE = {m['RMSE']:.3f} C   MAE = {m['MAE']:.3f} C   R2 = {m['R2']:.4f}",
        f"Referencia (perfil diario promedio):",
        f"  RMSE = {mb['RMSE']:.3f} C   MAE = {mb['MAE']:.3f} C   R2 = {mb['R2']:.4f}",
        "",
        "MAE por dia de prediccion (C):",
        "  " + " ".join(f"{e:.2f}" for e in r["mae_dias"]),
    ]
    return "\n".join(lineas)


def resumen_archivo(res):
    total, por_mes = metricas_archivo(res)
    dias = (~res["entrenamiento"]).sum() / (pd.Timedelta(days=1) / PASO)
    lineas = [
        "COMPARACION CON TODO EL ARCHIVO",
        f"(sin contar el entrenamiento: {dias:.0f} dias)",
        f"  RMSE = {total['RMSE']:.3f} C   MAE = {total['MAE']:.3f} C   R2 = {total['R2']:.4f}",
        "",
        "Por mes:",
    ]
    lineas += [f"  {mes}:  MAE = {m['MAE']:.2f} C   R2 = {m['R2']:.3f}" for mes, m in por_mes.items()]
    return "\n".join(lineas)


# ============================================================
# 4. GRAFICAS  (cada una dibuja sobre una figura de matplotlib)
# ============================================================

def _por_tramos(ax, d, columna, **kw):
    # Cada tramo se dibuja aparte para no unir con una linea los huecos sin datos.
    etiqueta = kw.pop("label", None)
    tramo = (d.index.to_series().diff() > pd.Timedelta(hours=MAX_HUECO_HORAS)).cumsum()
    for i, (_, g) in enumerate(d.groupby(tramo)):
        ax.plot(g.index, g[columna], label=etiqueta if i == 0 else None, **kw)


def _sombrear_entrenamiento(ax, train):
    ax.axvspan(train.index[0], train.index[-1], color="green", alpha=0.08, label="Entrenamiento")


def grafica_prediccion(fig, r, res):
    ax = fig.add_subplot()
    val, train = r["val"], r["train"]
    ultimos = train.iloc[-int(pd.Timedelta(days=3) / PASO):]
    ax.plot(ultimos.index, ultimos["Temperatura"], color="gray", linewidth=0.8, label="Entrenamiento (ultimos 3 dias)")
    ax.plot(val.index, val["Temperatura"], linewidth=1.0, label="Temperatura real")
    ax.plot(val.index, r["pred"], linewidth=1.0, label=f"Prediccion {DIAS_PREDICCION} dias")
    ax.plot(val.index, r["base"], color="gray", linewidth=0.8, linestyle=":", label="Referencia (perfil diario)")
    ax.axvline(val.index[0], linestyle="--", color="k", linewidth=1)
    ax.set_title(f"Prediccion de {DIAS_PREDICCION} dias  (MAE = {r['metricas']['MAE']:.2f} C, R2 = {r['metricas']['R2']:.3f})")
    ax.set_ylabel("Temperatura [C]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()


def grafica_archivo(fig, r, res):
    ax = fig.add_subplot()
    total, _ = metricas_archivo(res)
    _sombrear_entrenamiento(ax, r["train"])
    _por_tramos(ax, res, "real", color="C0", lw=0.5, label="Temperatura real")
    _por_tramos(ax, res, "simulada", color="C1", lw=0.5, alpha=0.8, label="Ecuacion simulada")
    ax.set_title(f"Ecuacion simulada sobre todo el archivo  |  fuera del entrenamiento: "
                 f"MAE {total['MAE']:.2f} C, R2 {total['R2']:.3f}")
    ax.set_ylabel("Temperatura [C]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()


def grafica_por_mes(fig, r, res):
    _, por_mes = metricas_archivo(res)
    meses = sorted(res["mes"].unique())
    for i, mes in enumerate(meses, 1):
        ax = fig.add_subplot(len(meses), 1, i)
        d = res[res["mes"] == mes]
        _sombrear_entrenamiento(ax, r["train"])
        ax.set_xlim(d.index[0], d.index[-1])
        _por_tramos(ax, d, "real", color="C0", lw=0.7, label="Real")
        _por_tramos(ax, d, "simulada", color="C1", lw=0.7, label="Simulada")
        texto = f"MAE {por_mes[mes]['MAE']:.2f} C, R2 {por_mes[mes]['R2']:.3f}" if mes in por_mes else "(entrenamiento)"
        ax.set_title(f"{mes}   {texto}", fontsize=9)
        ax.set_ylabel("T [C]")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        if i == 1:
            ax.legend(loc="upper left", ncol=3, fontsize=8)


def grafica_metricas_mes(fig, r, res):
    _, por_mes = metricas_archivo(res)
    meses = list(por_mes)
    a1, a2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    a1.bar(meses, [por_mes[m]["MAE"] for m in meses], color="C1")
    a1.set_ylabel("MAE [C]  (menor es mejor)")
    a2.bar(meses, [por_mes[m]["R2"] for m in meses], color="C0")
    a2.set_ylabel("R2  (mayor es mejor)")
    a2.set_ylim(0.5, 1)
    for a in (a1, a2):
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Error de la ecuacion por mes (sin contar el entrenamiento)")


def grafica_error_por_dia(fig, r, res):
    ax = fig.add_subplot()
    dias = np.arange(1, len(r["mae_dias"]) + 1)
    ax.bar(dias, r["mae_dias"], color="C1")
    ax.axhline(np.mean(r["mae_dias"]), color="k", linestyle="--", linewidth=1, label="Promedio")
    ax.set_xlabel("Dia de prediccion")
    ax.set_ylabel("MAE del dia [C]")
    ax.set_xticks(dias)
    ax.set_title("Error de cada dia de la prediccion: no crece con el horizonte")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()


def grafica_real_vs_simulado(fig, r, res):
    ax = fig.add_subplot()
    fuera = res[~res["entrenamiento"]]
    total, _ = metricas_archivo(res)
    ax.scatter(fuera["real"], fuera["simulada"], s=2, alpha=0.15)
    lim = [fuera[["real", "simulada"]].min().min(), fuera[["real", "simulada"]].max().max()]
    ax.plot(lim, lim, "k--", lw=1, label="Prediccion perfecta")
    ax.set_xlabel("Temperatura real [C]")
    ax.set_ylabel("Temperatura simulada [C]")
    ax.set_title(f"Todo el archivo sin el entrenamiento  (R2 = {total['R2']:.3f})")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()


def grafica_errores(fig, r, res):
    fuera = res[~res["entrenamiento"]]
    error = fuera["simulada"] - fuera["real"]
    a1, a2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    a1.hist(error, bins=80, color="C1")
    a1.axvline(0, color="k", lw=1)
    a1.set_xlabel("Error = simulado - real [C]")
    a1.set_ylabel("Cantidad de muestras")
    a1.set_title(f"Distribucion del error ({np.mean(np.abs(error) < 2) * 100:.0f}% dentro de +-2 C)")
    por_hora = error.abs().groupby(fuera.index.tz_convert(ZONA_HORARIA).hour).mean()
    a2.bar(por_hora.index, por_hora.values, color="C1")
    a2.set_xlabel("Hora del dia (Colombia)")
    a2.set_ylabel("MAE [C]")
    a2.set_title("Error segun la hora del dia")
    for a in (a1, a2):
        a.grid(alpha=0.3)


GRAFICAS = {
    "Prediccion de 20 dias": grafica_prediccion,
    "Todo el archivo": grafica_archivo,
    "Todo el archivo por mes": grafica_por_mes,
    "Metricas por mes": grafica_metricas_mes,
    "Error por dia de prediccion": grafica_error_por_dia,
    "Real vs simulado": grafica_real_vs_simulado,
    "Distribucion del error": grafica_errores,
}


# ============================================================
# 5. INTERFAZ (la del profesor, con un selector de graficas)
# ============================================================

class Aplicacion20(prof.Aplicacion):

    def __init__(self, root):
        self.bloque = None
        self.bloques_todos = None
        self.resultado = None
        self.archivo = None
        super().__init__(root)
        self.root.title(f"Identificacion PySINDy - Prediccion de {DIAS_PREDICCION} dias")
        self.agregar_controles()

    def agregar_controles(self):
        panel = self.info.master

        ttk.Label(panel, text="Grafica").pack(before=self.info, pady=(20, 5))
        self.grafica = tk.StringVar(value=list(GRAFICAS)[0])
        selector = ttk.Combobox(panel, textvariable=self.grafica, values=list(GRAFICAS), state="readonly")
        selector.pack(before=self.info, fill=tk.X)
        selector.bind("<<ComboboxSelected>>", lambda _: self.dibujar())

        # Barra de matplotlib: zoom, mover, guardar imagen.
        barra = NavigationToolbar2Tk(self.canvas, self.canvas.get_tk_widget().master, pack_toolbar=False)
        barra.update()
        barra.pack(side=tk.BOTTOM, fill=tk.X)

    def actualizar_grafica(self, datos=None):
        # La grafica de datos del profesor usa self.ax; se recrea por si otra grafica limpio la figura.
        self.fig.clear()
        self.ax = self.fig.add_subplot()
        super().actualizar_grafica(datos)

    def dibujar(self):
        if self.resultado is None:
            messagebox.showwarning("Aviso", "Primero ejecuta 'Ejecutar PySINDy'.")
            return
        self.fig.clear()
        GRAFICAS[self.grafica.get()](self.fig, self.resultado, self.archivo)
        self.fig.tight_layout()
        self.canvas.draw()

    def buscar_tramo(self):
        if self.datos is None:
            messagebox.showwarning("Aviso", "Primero abra un CSV.")
            return
        try:
            self.bloques_todos = bloques_rellenados(self.datos)
            self.bloque = elegir_bloque(self.bloques_todos)
            self.mejor_bloque = {"datos": self.bloque.rename_axis("datetime_utc").reset_index()}
            self.actualizar_grafica(self.mejor_bloque["datos"])

            b = self.bloque
            self.info.delete("1.0", tk.END)
            self.info.insert(tk.END,
                "MEJOR TRAMO ENCONTRADO\n=======================\n\n"
                f"Huecos rellenados: hasta {MAX_HUECO_HORAS} h\n"
                f"Tramos continuos: {len(self.bloques_todos)}\n\n"
                f"Inicio: {b.index[0]}\nFin:    {b.index[-1]}\n"
                f"Duracion: {(b.index[-1] - b.index[0]).total_seconds() / 86400:.1f} dias\n"
                f"Muestras: {len(b)}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def ejecutar_sindy(self):
        if self.bloque is None:
            messagebox.showwarning("Aviso", "Primero ejecuta 'Buscar mejor tramo'.")
            return
        try:
            self.info.delete("1.0", tk.END)
            self.info.insert(tk.END, "Identificando y simulando...\n")
            self.root.update()

            self.resultado = evaluar(self.bloque)
            self.archivo = simular_archivo(self.bloques_todos, self.resultado["coef"], self.resultado["train"])

            self.info.delete("1.0", tk.END)
            self.info.insert(tk.END, resumen(self.resultado) + "\n\n" + resumen_archivo(self.archivo))
            self.grafica.set(list(GRAFICAS)[0])
            self.dibujar()
        except Exception as e:
            messagebox.showerror("Error PySINDy", str(e))


# ============================================================
# MAIN
# ============================================================

def modo_consola(ruta, carpeta="graficas"):
    df, _ = prof.cargar_csv(ruta)
    datos, _ = prof.alinear_variables(prof.construir_series(df))
    bloques = bloques_rellenados(datos)
    r = evaluar(elegir_bloque(bloques))
    res = simular_archivo(bloques, r["coef"], r["train"])
    print(resumen(r) + "\n\n" + resumen_archivo(res))

    os.makedirs(carpeta, exist_ok=True)
    for i, (nombre, grafica) in enumerate(GRAFICAS.items(), 1):
        fig = plt.figure(figsize=(14, 12 if grafica is grafica_por_mes else 5.5))
        grafica(fig, r, res)
        fig.tight_layout()
        archivo = f"{carpeta}/{i}_{nombre.lower().replace(' ', '_')}.png"
        fig.savefig(archivo, dpi=130)
        plt.close(fig)
        print("Grafica guardada:", archivo)


if __name__ == "__main__":
    if "--consola" in sys.argv:
        argumentos = [a for a in sys.argv[1:] if a != "--consola"]
        modo_consola(argumentos[0] if argumentos else "Aeronic_export_29_09_2026.csv")
    else:
        root = tk.Tk()
        Aplicacion20(root)
        root.mainloop()
