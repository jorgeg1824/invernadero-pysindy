# Análisis de los tres programas originales (Etapa 1)

Los tres archivos están intactos en `originales/`. Todo lo que se afirma aquí se
comprobó ejecutándolos con `datos_ejemplo/Aeronic_export_29_09_2026.csv`
(ver `referencia_original/`). Las cifras corresponden a Python 3.13, pandas 3.0.5,
numpy 2.5.3, scipy 1.18.1 y pysindy 2.1.0.

> **Aclaración sobre la guía.** La guía dice que `validar_modelo.py` hace el
> barrido de `threshold`. Al revisar el código no es así: `validar_modelo.py`
> repite la predicción de 20 días en varias ventanas. El barrido de `threshold`
> está en `invernadero_profesor.py` (`buscar_mejor_modelo()`).

## 1. Qué hace cada programa

### 1.1 `invernadero_profesor.py` (1756 líneas)

**Entrada:** un CSV en formato largo, separado por tabuladores, con las columnas
`device_name`, `metric_name`, `datetime_utc` y `number_value`. Se elige con el
botón *Abrir CSV*.

**Salida:** solo la interfaz. Muestra texto en un panel y una gráfica. No guarda
archivos. En consola imprime una línea por cada threshold.

**Flujo** (3 botones: *Abrir CSV*, *Buscar mejor tramo*, *Ejecutar PySINDy*):

1. `cargar_csv()` lee el archivo y convierte `datetime_utc` a fecha UTC.
   `construir_series()` separa Temperatura, Luminosidad y Humedad
   (descarta `bomba` y `Suelo_pct`).
2. `alinear_variables()` usa la Temperatura como referencia y asigna a cada
   instante la L y la H más cercanas (`merge_asof` con `nearest`, tolerancia de
   2 × dt típico = 600 s). Elimina marcas duplicadas (6 en el CSV).
3. `seleccionar_mejor_bloque()` → `obtener_bloques_continuos()` corta los datos
   donde dt > 4 × dt mediano (más de 20 min sin datos). Sin interpolar.
   Entre los bloques de al menos **6 días** elige el de mayor puntaje
   `0.55·log(1+horas) + 0.45·excitación`.
4. `limpiar_tramo()` quita NaN y duplicados, y descarta muestras con dt > 5 × mediana.
5. `dividir_train_validacion()`: los **primeros 5 días** entrenan y el resto valida.
6. `buscar_mejor_modelo()`: para cada uno de 7 thresholds (1e-5 … 1e-2):
   - `identificar_modelo()` normaliza con el entrenamiento (`Escalador`).
     Calcula dT/dt con `calcular_derivada_temperatura()` (Savitzky-Golay de
     ventana 21 + `np.gradient` sobre el tiempo real). Ajusta PySINDy
     (`PolynomialLibrary(degree=1)`, `STLSQ(alpha=0.001)`).
   - `simular_modelo()` integra con `solve_ivp` (RK45) en variables normalizadas
     sobre la validación.
   - Se queda con el de **menor RMSE en la validación**.
7. `obtener_coeficientes_fisicos()` y `ecuacion_fisica()` des-normalizan los
   coeficientes y escriben `dT/dt = a0 + aT·T + aL·L + aH·H`.

**Parámetros que puede cambiar el usuario:** solo qué variables se dibujan
(casillas). Todo lo demás está fijo en el código (6 días, 5 días, factores 4 y 5,
ventana 21, thresholds).

**Gráficas:** las variables en el tiempo, y la temperatura real frente a la
predicha en la validación. **Métricas:** RMSE, MAE y R².

**Resultado con el CSV de ejemplo:**

| | pandas 3.0.5 | pandas 2.2.3 |
|---|---|---|
| Tramo | 15-jun → 25-jun (10,3 días, 3027 muestras) | igual |
| Entrenamiento / validación | 1494 / 1533 muestras | igual |
| Threshold elegido | 1e-5 (los 7 dan el mismo modelo) | 0,01 |
| Términos activos | 4 | 2 (sin constante ni L) |
| Ecuación | `35.918599 − 0.77755486·T + 5.19e-05·L − 0.23417621·H` | `0.034859905 − 0.00076353946·T − 0.00023492606·H` |
| RMSE / R² validación | 2,820 °C / 0,914 | 2,645 °C / 0,924 |

El mismo programa da **otra ecuación** según la versión de pandas (error C1 en
`docs/informe_final.md`).

### 1.2 `invernadero_20dias.py` (498 líneas)

**Importa** `invernadero_profesor.py` y reutiliza sin cambios estas piezas:
`cargar_csv`, `construir_series`, `alinear_variables`, `Escalador`,
`calcular_metricas`, `ecuacion_fisica` y la interfaz (`Aplicacion`, que extiende).

**Entrada:** el mismo CSV. **Salida:** la interfaz del profesor más un selector de
7 gráficas y la barra de matplotlib. Con `--consola` no abre ventana: imprime el
resumen y guarda 7 PNG en `graficas/`.

**Flujo:**

1. Carga y alineación: **las mismas funciones del profesor**.
2. `bloques_rellenados()` corta donde hay más de **2 h** sin datos. Los huecos
   menores los rellena llevando cada tramo a una **malla uniforme de 5 min** con
   interpolación lineal en el tiempo.
3. `elegir_bloque()`: entre los tramos de al menos 20 + 5 días, el de **más
   muestras** (18-ago → 28-sep, 41,1 días).
4. `dividir()`: los **últimos 20 días** se predicen y todo lo anterior entrena
   (21,1 días).
5. `identificar()`:
   - dT/dt con `savgol_filter(..., 21, 2, deriv=1, delta=5 min en horas)` sobre
     la malla uniforme.
   - PySINDy con threshold fijo 0,001 (mismo modelo que `crear_modelo()`).
   - **Ajuste por simulación:** `least_squares` cambia los coeficientes
     normalizados que PySINDy dejó activos para minimizar el error de la
     simulación del entrenamiento, con aT = −exp(q) < 0.
6. `simular()` integra con `scipy.signal.lsim` en variables físicas (horas).
7. `evaluar()` calcula las métricas de la predicción, una **referencia ingenua**
   (perfil diario promedio del entrenamiento) y el MAE de cada día.
8. `simular_archivo()` simula la ecuación en **todos** los tramos del archivo y
   da métricas fuera del entrenamiento y por mes (hora de Bogotá).

**Parámetros:** constantes al inicio del archivo (`DIAS_PREDICCION`,
`MAX_HUECO_HORAS`, `THRESHOLD`, …). No se pueden cambiar desde la interfaz.

**Resultado:**

| | |
|---|---|
| PySINDy solo | `aT = +0,148 /h` → **inestable**, diverge al simular |
| Ecuación ajustada (/h) | `584.49 − 17.9945·T + 0.018866·L − 3.02674·H` (τ ≈ 3,3 min) |
| Predicción 20 días | RMSE 1,365 °C, MAE 1,049 °C, R² 0,966 |
| Referencia ingenua | RMSE 3,923 °C, R² 0,717 |
| Todo el archivo sin el entrenamiento (80 días) | RMSE 1,628 °C, R² 0,964 |

No depende de la versión de pandas: con pandas 2.2.3 los coeficientes cambian en
~1e-7 relativo, por el optimizador.

### 1.3 `validar_modelo.py` (160 líneas)

Es un **script**: hace todo al importarse, sin funciones ni `if __name__`. Lee
`Aeronic_export_29_09_2026.csv` del directorio actual y escribe en
`validacion/`. Importa los otros dos programas.

**Flujo:** toma los tramos rellenados de al menos 25 días (2 tramos) y en cada uno
recorta el final cada 4 días. Cada recorte es una ventana con la función
`evaluar()` de `invernadero_20dias.py` (6 ventanas). Imprime una tabla y guarda 10
PNG: una gráfica por ventana, un resumen, el error por día, real vs predicho y la
distribución del error.

| Ventana de predicción | Entrenamiento | MAE | R² | Referencia R² | PySINDy solo |
|---|---|---|---|---|---|
| 27-jul a 16-ago | 5,9 d | 1,26 | 0,970 | 0,940 | diverge |
| 23-ago a 12-sep | 5,0 d | 1,14 | 0,974 | 0,929 | diverge |
| 27-ago a 16-sep | 9,0 d | 1,10 | 0,974 | 0,923 | diverge |
| 31-ago a 20-sep | 13,0 d | 1,18 | 0,968 | 0,885 | diverge |
| 04-sep a 24-sep | 17,0 d | 1,18 | 0,964 | 0,807 | diverge |
| 08-sep a 28-sep | 21,1 d | 1,05 | 0,966 | 0,717 | diverge |

Las 5 ventanas del segundo tramo se **solapan** y comparten el inicio del
entrenamiento, así que no son pruebas independientes entre sí. La última ventana
es exactamente la predicción de `invernadero_20dias.py`.

## 2. Tabla comparativa

| Aspecto | `invernadero_profesor.py` | `invernadero_20dias.py` | `validar_modelo.py` |
|---|---|---|---|
| Tipo | Interfaz gráfica | Interfaz + modo consola | Script |
| Archivos que produce | ninguno | 7 PNG (`--consola`) | 10 PNG + tabla en consola |
| Carga, series, alineación | propias | **las del profesor** | **las del profesor** (vía 20dias) |
| Huecos | corta en > 4 × dt (20 min) | corta en > 2 h y rellena los menores | igual que 20dias |
| Malla temporal | irregular (muestras originales) | uniforme 5 min | uniforme 5 min |
| Elección del tramo | mayor puntaje, ≥ 6 días | más muestras, ≥ 25 días | todos los tramos ≥ 25 días |
| Partición | primeros 5 días / resto | resto / últimos 20 días | varias ventanas de 20 días |
| Derivada | SavGol(21) + `np.gradient` (tiempo real) | SavGol(21, `deriv=1`), malla | igual que 20dias |
| Unidad de tiempo | segundos (con pandas 3: kilosegundos) | horas | horas |
| Normalización | `Escalador` | **el mismo** | el mismo |
| Modelo PySINDy | grado 1, STLSQ, alpha 0,001 | **el mismo** | el mismo |
| Threshold | barrido de 7 valores | fijo 0,001 | fijo 0,001 |
| Ajuste por simulación | no | sí (`least_squares`, aT < 0) | sí |
| Integrador | `solve_ivp` RK45, normalizado | `lsim`, físico | `lsim` |
| Selección del modelo | menor RMSE **en la validación** | — | — |
| Métricas | RMSE, MAE, R² | + MAE por día, por mes, archivo completo | + por ventana |
| Referencia ingenua | no | perfil diario promedio | sí |
| Parámetros modificables | ninguno (solo casillas de gráfica) | constantes en el código | constantes en el código |

## 3. Funciones repetidas y funciones que solo se parecen

La guía pide comprobar, antes de unificar, si dos funciones hacen realmente lo
mismo. Resultado:

### Se comparten (son equivalentes, comprobado)

| Funciones | Por qué se pueden unificar | Cómo se comprobó |
|---|---|---|
| `cargar_csv`, `construir_series`, `alinear_variables`, `estimar_dt_median` | 20dias y validar_modelo **importan** las del profesor: es el mismo código, el mismo criterio temporal (`nearest`, 2 × dt) y la misma tolerancia | Mismos conteos en los tres (29 579 filas alineadas) |
| `Escalador` | 20dias usa `prof.Escalador` | — |
| `calcular_metricas` | 20dias usa `prof.calcular_metricas` | `tests/test_metricas.py` |
| `obtener_coeficientes_fisicos()` (profesor) y `a_fisico()` (20dias) | Misma fórmula; una toma los coeficientes por nombre y la otra por posición | `tests/test_modelos.py::test_conversion_de_coeficientes_equivalente` (diferencia < 1e-12) |
| `crear_modelo()` (profesor) y el `ps.SINDy(...)` en línea de 20dias | Misma librería y optimizador con los mismos parámetros; solo cambia el threshold, que ahora es argumento | Comparación de coeficientes con la referencia (diferencia 0) |
| Parte común de `identificar_modelo()` (profesor) e `identificar()` (20dias) | Normalizar, ajustar PySINDy con una derivada dada y des-normalizar | Igual a la referencia en ambos flujos |

### Se conservan separadas (se parecen pero hacen otra cosa)

| Funciones | Diferencia |
|---|---|
| `obtener_bloques_continuos()` vs `bloques_rellenados()` | Umbral relativo (4 × dt) vs absoluto (2 h); la segunda interpola en malla uniforme y la primera no |
| `limpiar_tramo()` vs el relleno interno de `bloques_rellenados()` | Uno quita muestras y deja el tiempo irregular; el otro crea una malla nueva e interpola |
| `seleccionar_mejor_bloque()` vs `elegir_bloque()` | Puntaje de duración + excitación (≥ 6 días) vs más muestras (≥ 25 días) |
| `dividir_train_validacion()` vs `dividir()` | Primeros N días entrenan vs últimos N días se predicen |
| `calcular_derivada_temperatura()` vs el `savgol_filter(deriv=1)` de `identificar()` | Suaviza tratando las muestras como equiespaciadas y luego deriva con el tiempo real, vs deriva en un solo paso en malla uniforme. Dan derivadas distintas y PySINDy obtiene ecuaciones distintas |
| `simular_modelo()` (RK45) vs `simular()` (lsim) | Mismo modelo matemático, pero RK45 con `rtol=1e-3` se desvía hasta ~0,13 °C en los cambios bruscos de luz (`tests/test_simulacion.py`), y lsim exige paso fijo. Cada flujo conserva el suyo para reproducir sus resultados |

## 4. Errores y limitaciones encontrados en los originales

Se registraron **sin corregirlos** en la referencia. Las correcciones aplicadas
en la versión modular están en `docs/informe_final.md`.

| # | Programa | Problema | Efecto medido |
|---|---|---|---|
| E1 | profesor | El tiempo se calcula con `astype("int64") / 1e9`, que solo da segundos si pandas guarda nanosegundos. pandas 3 guarda microsegundos | Con pandas 3 el tiempo sale en **kilosegundos** y el barrido elige otro modelo (tabla 1.1) |
| E2 | profesor | Con pandas 3 los 7 thresholds dan el mismo modelo de 4 términos | El barrido no selecciona nada |
| E3 | profesor | El threshold se elige con la validación y luego se reportan las métricas de esa misma validación | Esas métricas no son una evaluación independiente |
| E4 | profesor | Savitzky-Golay supone muestras equiespaciadas y se aplica a datos irregulares | Picos en dT/dt después de cada salto temporal (gráfica de la etapa 6) |
| E5 | profesor | El puntaje del tramo divide por la mediana de cada variable. La mediana de L es casi 0 (de noche L = 0) | El término de luminosidad aporta ~99 % del puntaje (279 de 282 en el tramo elegido); la duración aporta ~1,7. Gana el tramo con menos luz mediana |
| E6 | profesor | `cargar_csv()` prueba primero tabulador; un CSV con comas no falla, queda en una columna y luego dice "no se encontró datetime_utc" | No lee CSV separados por comas |
| E7 | profesor | `cargar_csv()` acepta formato ancho, pero `construir_series()` lo rechaza. Tiene un bucle sin efecto (`pass`) e importa `SpanSelector` y `ZoneInfo` sin usarlos | Código muerto |
| E8 | profesor | El filtro de `limpiar_tramo()` (dt > 5 × mediana) nunca actúa: el bloque ya se cortó en 4 × mediana | 3027 → 3027 muestras |
| E9 | profesor | La interfaz se congela mientras calcula (sin hilos) | ~15 s sin respuesta |
| E10 | 20dias | Con ventana 21, PySINDy solo da aT > 0 en **todas** las ventanas | La "ecuación PySINDy" que imprime es inestable; el ajuste arranca desde aT = −1 |
| E11 | 20dias | El modo consola necesita `tkinter` porque importa el programa del profesor | No corre en un entorno sin interfaz gráfica |
| E12 | 20dias | La referencia ingenua agrupa por hora UTC y la gráfica de error por hora usa hora de Bogotá | Inconsistente, sin efecto en las métricas |
| E13 | validar_modelo | Se ejecuta al importarse, con nombre de CSV y carpetas fijos, y silencia todas las advertencias | No es reutilizable |
| E14 | validar_modelo | Ventanas solapadas que comparten entrenamiento | No son 6 pruebas independientes |
| E15 | profesor/20dias | `merge_asof(nearest)` puede asignar la misma medición de L o H a dos instantes de T | En este CSV los desfases son 0 s (T, L y H llegan juntas): sin efecto |
