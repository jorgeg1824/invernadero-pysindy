"""
Interfaz gráfica: capa de presentación sobre pipeline.py.

No hace cálculos: arma la configuración, llama a una funcionalidad del
pipeline en un hilo aparte y muestra lo que devolvió cada etapa
(texto, tablas y gráficas). Reemplaza a las interfaces de
invernadero_profesor.py y invernadero_20dias.py (ver docs/mapa_migracion.md).

    python cli.py gui
"""

from __future__ import annotations

import dataclasses
import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import config as C

FUNCIONES = {
    "Barrido de threshold  (invernadero_profesor.py)": "barrido",
    "Predicción a 20 días  (invernadero_20dias.py)": "prediccion",
    "Validación en ventanas  (validar_modelo.py)": "ventanas",
    "Comparar estrategias A / B / C  (nuevo)": "estrategias",
}
OPCIONES = {
    "unidad_tiempo": ["s", "original"],
    "seleccion_threshold": ["validacion", "separado"],
}
FILAS_TABLA = 500


class Aplicacion:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Invernadero - identificación con PySINDy")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 700)

        self.ruta_csv = tk.StringVar(value=str(C.CSV_EJEMPLO) if C.CSV_EJEMPLO.exists() else "")
        self.funcion = tk.StringVar(value=list(FUNCIONES)[0])
        self.guardar = tk.BooleanVar(value=True)
        self.estado = tk.StringVar(value="Elija un CSV y una funcionalidad, y pulse Ejecutar.")
        self.campos = {}
        self.ejecucion = None
        self.etapa = None
        self.cola = queue.Queue()
        self._auto_resumen = False

        self._construir()
        self._formulario()
        self.root.after(100, self._revisar_cola)

    # ========================================================
    # Construcción de la ventana
    # ========================================================

    def _construir(self):
        izq = ttk.Frame(self.root, padding=8)
        izq.pack(side=tk.LEFT, fill=tk.Y)
        der = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Datos
        ttk.Label(izq, text="1. Datos", font=("", 10, "bold")).pack(anchor="w")
        f = ttk.Frame(izq)
        f.pack(fill=tk.X, pady=(2, 10))
        ttk.Entry(f, textvariable=self.ruta_csv, width=34).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(f, text="Abrir CSV…", command=self._abrir_csv).pack(side=tk.LEFT, padx=(4, 0))

        # --- Funcionalidad
        ttk.Label(izq, text="2. Funcionalidad", font=("", 10, "bold")).pack(anchor="w")
        cb = ttk.Combobox(izq, textvariable=self.funcion, values=list(FUNCIONES), state="readonly", width=44)
        cb.pack(fill=tk.X, pady=(2, 10))
        cb.bind("<<ComboboxSelected>>", lambda _: self._formulario())

        # --- Parámetros
        ttk.Label(izq, text="3. Parámetros", font=("", 10, "bold")).pack(anchor="w")
        marco = ttk.Frame(izq)
        marco.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
        lienzo = tk.Canvas(marco, width=330, highlightthickness=0)
        barra = ttk.Scrollbar(marco, orient=tk.VERTICAL, command=lienzo.yview)
        self.form = ttk.Frame(lienzo)
        self.form.bind("<Configure>", lambda _: lienzo.configure(scrollregion=lienzo.bbox("all")))
        lienzo.create_window((0, 0), window=self.form, anchor="nw")
        lienzo.configure(yscrollcommand=barra.set)
        lienzo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barra.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(izq, text="Restablecer valores originales", command=self._formulario).pack(fill=tk.X)

        # --- Ejecutar
        ttk.Checkbutton(izq, text="Guardar resultados en resultados/", variable=self.guardar).pack(anchor="w", pady=(8, 0))
        self.boton = ttk.Button(izq, text="Ejecutar", command=self._ejecutar)
        self.boton.pack(fill=tk.X, pady=(4, 2), ipady=4)
        self.progreso = ttk.Progressbar(izq, mode="indeterminate")
        self.progreso.pack(fill=tk.X)
        self.boton_carpeta = ttk.Button(izq, text="Abrir carpeta de resultados", command=self._abrir_carpeta,
                                        state=tk.DISABLED)
        self.boton_carpeta.pack(fill=tk.X, pady=(6, 0))

        # --- Etapas (derecha)
        arriba = ttk.Frame(der)
        arriba.pack(fill=tk.X)
        ttk.Label(arriba, text="Etapas:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.botones_etapa = []
        for n, nombre in enumerate(["1 Datos originales", "2 Transformación", "3 Alineación", "4 Limpieza",
                                    "5 Selección del tramo", "6 Preparación", "7 Identificación",
                                    "8 Simulación y evaluación"], 1):
            b = ttk.Button(arriba, text=nombre, state=tk.DISABLED, command=lambda n=n: self._mostrar_etapa(n))
            b.pack(side=tk.LEFT, padx=1)
            self.botones_etapa.append(b)

        self.titulo = ttk.Label(der, text="", font=("", 12, "bold"))
        self.titulo.pack(anchor="w", pady=(8, 4))

        self.pestanas = ttk.Notebook(der)
        self.pestanas.pack(fill=tk.BOTH, expand=True)

        # Pestaña Gráficas
        pg = ttk.Frame(self.pestanas)
        self.pestanas.add(pg, text="Gráficas")
        fila = ttk.Frame(pg)
        fila.pack(fill=tk.X, pady=4)
        ttk.Label(fila, text="Gráfica:").pack(side=tk.LEFT)
        self.grafica = tk.StringVar()
        self.selector_grafica = ttk.Combobox(fila, textvariable=self.grafica, state="readonly", width=50)
        self.selector_grafica.pack(side=tk.LEFT, padx=4)
        self.selector_grafica.bind("<<ComboboxSelected>>", lambda _: self._dibujar())
        self.fig = Figure(figsize=(9, 6), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=pg)
        barra_mpl = NavigationToolbar2Tk(self.canvas, pg, pack_toolbar=False)
        barra_mpl.update()
        barra_mpl.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña Resumen
        pr = ttk.Frame(self.pestanas)
        self.pestanas.add(pr, text="Resumen")
        self.texto = tk.Text(pr, wrap=tk.NONE, font=("Courier", 10))
        sy = ttk.Scrollbar(pr, orient=tk.VERTICAL, command=self.texto.yview)
        sx = ttk.Scrollbar(pr, orient=tk.HORIZONTAL, command=self.texto.xview)
        self.texto.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        self.texto.pack(fill=tk.BOTH, expand=True)

        # Pestaña Tablas
        pt = ttk.Frame(self.pestanas)
        self.pestanas.add(pt, text="Tablas")
        fila = ttk.Frame(pt)
        fila.pack(fill=tk.X, pady=4)
        ttk.Label(fila, text="Tabla:").pack(side=tk.LEFT)
        self.tabla = tk.StringVar()
        self.selector_tabla = ttk.Combobox(fila, textvariable=self.tabla, state="readonly", width=40)
        self.selector_tabla.pack(side=tk.LEFT, padx=4)
        self.selector_tabla.bind("<<ComboboxSelected>>", lambda _: self._mostrar_tabla())
        self.info_tabla = ttk.Label(fila, text="")
        self.info_tabla.pack(side=tk.LEFT, padx=8)
        self.arbol = ttk.Treeview(pt, show="headings")
        ty = ttk.Scrollbar(pt, orient=tk.VERTICAL, command=self.arbol.yview)
        tx = ttk.Scrollbar(pt, orient=tk.HORIZONTAL, command=self.arbol.xview)
        self.arbol.configure(yscrollcommand=ty.set, xscrollcommand=tx.set)
        ty.pack(side=tk.RIGHT, fill=tk.Y)
        tx.pack(side=tk.BOTTOM, fill=tk.X)
        self.arbol.pack(fill=tk.BOTH, expand=True)

        # Pestaña Registro
        pl = ttk.Frame(self.pestanas)
        self.pestanas.add(pl, text="Registro de la ejecución")
        self.registro = tk.Text(pl, wrap=tk.NONE, font=("Courier", 9))
        ly = ttk.Scrollbar(pl, orient=tk.VERTICAL, command=self.registro.yview)
        self.registro.configure(yscrollcommand=ly.set)
        ly.pack(side=tk.RIGHT, fill=tk.Y)
        self.registro.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.root, textvariable=self.estado, relief=tk.SUNKEN, anchor="w",
                  padding=(6, 2)).pack(side=tk.BOTTOM, fill=tk.X, before=izq)

    # ========================================================
    # Formulario de parámetros (generado desde config.py)
    # ========================================================

    def _clave(self):
        return FUNCIONES[self.funcion.get()]

    def _cfg_por_defecto(self):
        clave = self._clave()
        return C.CONFIGS["prediccion" if clave == "estrategias" else clave]()

    def _formulario(self):
        for w in self.form.winfo_children():
            w.destroy()
        self.campos = {}
        fila = [0]

        def agregar(nombre, valor, prefijo=""):
            clave = prefijo + nombre
            ttk.Label(self.form, text=clave.replace("_", " ")).grid(row=fila[0], column=0, sticky="w", pady=1)
            if isinstance(valor, bool):
                var = tk.BooleanVar(value=valor)
                ttk.Checkbutton(self.form, variable=var).grid(row=fila[0], column=1, sticky="w")
            elif nombre in OPCIONES:
                var = tk.StringVar(value=valor)
                ttk.Combobox(self.form, textvariable=var, values=OPCIONES[nombre], state="readonly",
                             width=14).grid(row=fila[0], column=1, sticky="w")
            else:
                texto = ", ".join(f"{x:g}" for x in valor) if isinstance(valor, list) else str(valor)
                var = tk.StringVar(value=texto)
                ttk.Entry(self.form, textvariable=var, width=16).grid(row=fila[0], column=1, sticky="w")
            self.campos[clave] = (var, valor)
            fila[0] += 1

        cfg = self._cfg_por_defecto()
        for f in dataclasses.fields(cfg):
            valor = getattr(cfg, f.name)
            if dataclasses.is_dataclass(valor):
                for g in dataclasses.fields(valor):
                    agregar(g.name, getattr(valor, g.name), f"{f.name}.")
            else:
                agregar(f.name, valor)

    def _leer_formulario(self):
        cfg = self._cfg_por_defecto()
        for clave, (var, original) in self.campos.items():
            texto = var.get()
            if isinstance(original, bool):
                valor = bool(texto)
            elif isinstance(original, list):
                valor = [float(x) for x in str(texto).replace(";", ",").split(",") if x.strip()]
            elif isinstance(original, int):
                valor = int(texto)
            elif isinstance(original, float):
                valor = float(texto)
            else:
                valor = str(texto)
            destino = cfg
            partes = clave.split(".")
            for p in partes[:-1]:
                destino = getattr(destino, p)
            setattr(destino, partes[-1], valor)
        return cfg

    # ========================================================
    # Ejecución (en un hilo para no congelar la ventana)
    # ========================================================

    def _abrir_csv(self):
        ruta = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if ruta:
            self.ruta_csv.set(ruta)

    def _ejecutar(self):
        try:
            cfg = self._leer_formulario()
        except ValueError as e:
            messagebox.showerror("Parámetro inválido", str(e))
            return
        if not os.path.exists(self.ruta_csv.get()):
            messagebox.showerror("Archivo", "Elija un archivo CSV existente.")
            return

        self.boton.configure(state=tk.DISABLED)
        self.progreso.start(12)
        self.registro.delete("1.0", tk.END)
        clave = self._clave()

        def tarea():
            import pipeline
            try:
                ejec = pipeline.FUNCIONALIDADES[clave](
                    ruta_csv=self.ruta_csv.get(), cfg=cfg, guardar=self.guardar.get(),
                    aviso=lambda s: self.cola.put(("aviso", s)))
                self.cola.put(("fin", ejec))
            except Exception as e:
                self.cola.put(("error", f"{e}\n\n{traceback.format_exc()}"))

        threading.Thread(target=tarea, daemon=True).start()
        self.estado.set(f"Ejecutando '{clave}'…")

    def _revisar_cola(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "aviso":
                    self.estado.set(dato)
                    self.registro.insert(tk.END, dato + "\n")
                    self.registro.see(tk.END)
                elif tipo == "fin":
                    self._terminar(dato)
                elif tipo == "error":
                    self._terminar(None)
                    messagebox.showerror("Error", dato[:3000])
        except queue.Empty:
            pass
        self.root.after(100, self._revisar_cola)

    def _terminar(self, ejec):
        self.progreso.stop()
        self.boton.configure(state=tk.NORMAL)
        if ejec is None:
            self.estado.set("La ejecución falló.")
            return
        self.ejecucion = ejec
        numeros = {e.numero for e in ejec.etapas}
        for n, b in enumerate(self.botones_etapa, 1):
            b.configure(state=tk.NORMAL if n in numeros else tk.DISABLED)
        self.boton_carpeta.configure(state=tk.NORMAL if ejec.carpeta else tk.DISABLED)
        self.registro.insert(tk.END, "\n" + ejec.resultado.get("resumen", ""))
        self.estado.set(f"Listo. {'Resultados en ' + str(ejec.carpeta) if ejec.carpeta else 'Sin guardar.'}")
        self._mostrar_etapa(8 if 8 in numeros else max(numeros))

    # ========================================================
    # Mostrar una etapa
    # ========================================================

    def _mostrar_etapa(self, n):
        self.etapa = self.ejecucion.etapa(n)
        e = self.etapa
        self.titulo.configure(text=f"Etapa {e.numero} - {e.titulo}")
        self.texto.delete("1.0", tk.END)
        self.texto.insert(tk.END, e.texto)

        nombres = list(e.graficas)
        self.selector_grafica.configure(values=nombres)
        self.grafica.set(nombres[0] if nombres else "")
        self._dibujar()

        tablas = list(e.tablas)
        self.selector_tabla.configure(values=tablas)
        self.tabla.set(tablas[0] if tablas else "")
        self._mostrar_tabla()

        # Si la etapa no tiene gráficas se muestra el resumen; al volver a una
        # etapa con gráficas se regresa a la pestaña de gráficas.
        actual = self.pestanas.index(self.pestanas.select())
        if not nombres:
            if actual == 0:
                self.pestanas.select(1)
                self._auto_resumen = True
        else:
            if self._auto_resumen and actual == 1:
                self.pestanas.select(0)
            self._auto_resumen = False

    def _dibujar(self):
        self.fig.clear()
        if self.etapa and self.grafica.get() in self.etapa.graficas:
            try:
                self.etapa.graficas[self.grafica.get()](self.fig)
            except Exception as ex:
                self.fig.text(0.5, 0.5, f"No se pudo dibujar:\n{ex}", ha="center")
        else:
            self.fig.text(0.5, 0.5, "Esta etapa no tiene gráficas; vea Resumen y Tablas.", ha="center")
        self.canvas.draw_idle()

    def _mostrar_tabla(self):
        self.arbol.delete(*self.arbol.get_children())
        if not self.etapa or self.tabla.get() not in self.etapa.tablas:
            self.arbol.configure(columns=())
            self.info_tabla.configure(text="")
            return
        df = self.etapa.tablas[self.tabla.get()]
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(names="datetime_utc")
        columnas = [str(c) for c in df.columns]
        self.arbol.configure(columns=columnas)
        muestra = df.head(1).astype(str)
        for c, original in zip(columnas, df.columns):
            ancho = max(len(c), len(muestra[original].iloc[0]) if len(df) else 0)
            self.arbol.heading(c, text=c)
            self.arbol.column(c, width=max(80, min(320, 8 * ancho + 16)), stretch=False)
        for fila in df.head(FILAS_TABLA).itertuples(index=False):
            self.arbol.insert("", tk.END, values=[f"{v:.6g}" if isinstance(v, float) else str(v) for v in fila])
        extra = f" (se muestran {FILAS_TABLA})" if len(df) > FILAS_TABLA else ""
        self.info_tabla.configure(text=f"{len(df)} filas x {len(columnas)} columnas{extra}")

    def _abrir_carpeta(self):
        ruta = str(self.ejecucion.carpeta)
        try:
            if sys.platform.startswith("win"):
                os.startfile(ruta)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", ruta])
            else:
                subprocess.Popen(["xdg-open", ruta])
        except Exception:
            messagebox.showinfo("Carpeta de resultados", ruta)


def main():
    root = tk.Tk()
    Aplicacion(root)
    root.mainloop()


if __name__ == "__main__":
    main()
