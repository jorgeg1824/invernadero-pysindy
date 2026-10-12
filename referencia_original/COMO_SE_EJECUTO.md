# Versión de referencia (Etapa 2)

Resultados de los **tres programas originales sin modificar** (`originales/`)
con el CSV de ejemplo. Sirven para comprobar que la versión modular conserva su
funcionamiento (`docs/comparacion_original_vs_modular.md`).

## Cómo se generó

```bash
# Desde la raíz del repositorio, con las dependencias de requirements.txt
python referencia_original/ejecutar_referencia.py
# Linux sin pantalla (la interfaz del profesor necesita una):
xvfb-run -a python referencia_original/ejecutar_referencia.py
```

`ejecutar_referencia.py` **no cambia** ningún original; los importa desde
`originales/`. Los md5 quedan en `resultados/registro_ejecucion.json`.

| Programa | Cómo se ejecutó | Configuración |
|---|---|---|
| `invernadero_profesor.py` | Se crea su interfaz (`Aplicacion`) y se llaman en orden los métodos de los botones: *Abrir CSV* → *Buscar mejor tramo* → *Ejecutar PySINDy*. El diálogo de archivo se reemplaza por la ruta del CSV de ejemplo | La del código: tramo ≥ 6 días, 5 días de entrenamiento, 7 thresholds 1e-5…1e-2 |
| `invernadero_20dias.py` | `modo_consola(csv, carpeta)`, lo mismo que `python invernadero_20dias.py --consola` | Constantes del archivo: 20 días, huecos ≤ 2 h, malla de 5 min, threshold 0,001 |
| `validar_modelo.py` | `runpy.run_path()` desde una carpeta con una copia del CSV (el script usa rutas relativas) | Constantes del archivo: ventanas cada 4 días |

Como la interfaz no guarda resultados, para volcar los números el script repite
las mismas llamadas que hace `ejecutar_sindy()`. La ecuación y las métricas del
`resultados.json` son las mismas que mostró la interfaz (`interfaz_texto.txt`).

## Entornos

| Carpeta | Python | pandas | numpy | scipy | pysindy | Uso |
|---|---|---|---|---|---|---|
| `resultados/` | 3.13 | **3.0.5** | 2.5.3 | 1.18.1 | 2.1.0 | Referencia principal (el entorno de `requirements.txt`) |
| `resultados_pandas2/` | 3.13 | **2.2.3** | 2.5.3 | 1.18.1 | 2.1.0 | Solo profesor y 20dias, para documentar el error E1 |

## Contenido

```
resultados/
├── registro_ejecucion.json      entorno, versiones, md5 de los originales y del CSV
├── profesor/
│   ├── resultados.json          configuración, conteos, tramo, barrido, ecuación, métricas
│   ├── trayectoria_validacion.csv
│   ├── interfaz_texto.txt       texto que mostró la interfaz en cada paso
│   ├── consola.txt
│   └── 1_datos_cargados.png  2_mejor_tramo.png  3_validacion_pysindy.png
├── 20dias/
│   ├── resultados.json
│   ├── trayectoria_prediccion_20d.csv  trayectoria_archivo.csv
│   ├── consola.txt
│   └── graficas/  (las 7 gráficas del modo consola)
└── validar_modelo/
    ├── resultados.json          las 6 ventanas
    ├── consola.txt              la tabla que imprime
    └── validacion/  (las 10 gráficas)
```

## Resultados principales

| Programa | Resultado |
|---|---|
| profesor (pandas 3) | threshold 1e-5; `dT/dt = 35.918599 − 0.77755486·T + 5.1905572e-05·L − 0.23417621·H`; validación RMSE 2,820 °C, R² 0,914 |
| profesor (pandas 2) | threshold 0,01; `dT/dt = 0.034859905 − 0.00076353946·T − 0.00023492606·H`; RMSE 2,645 °C, R² 0,924 |
| 20dias | predicción de 20 días RMSE 1,365 °C, R² 0,966; archivo completo sin entrenamiento R² 0,964 |
| validar_modelo | 6 ventanas, R² entre 0,964 y 0,974; PySINDy solo diverge en todas |

## Errores o limitaciones que ya existían

Los 15 problemas encontrados (E1 a E15) están descritos y medidos en
`docs/analisis_programas.md`, sección 4. Los más importantes para la referencia:

* **E1:** `invernadero_profesor.py` da una ecuación distinta según la versión de
  pandas, por eso hay dos carpetas de referencia.
* **E3:** las métricas de validación del profesor no son independientes: el
  threshold se eligió con esos mismos datos.
* **E10:** la "ecuación PySINDy" que imprime `invernadero_20dias.py` es inestable
  (aT > 0); solo la ecuación ajustada por simulación predice.
* **E11:** el modo consola de 20dias y `validar_modelo.py` necesitan `tkinter`
  aunque no abran ventanas.
