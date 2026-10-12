# Informe final

Refactorización y verificación de un sistema de identificación de modelos de
invernadero (PySINDy). Este informe resume las decisiones de diseño, las
dificultades, los errores detectados en los originales, las correcciones y
mejoras aplicadas y las mejoras que se proponen.

Documentos relacionados:

| Documento | Contenido |
|---|---|
| `docs/analisis_programas.md` | Etapa 1: qué hace cada programa, tabla comparativa, funciones repetidas, errores E1–E15 |
| `referencia_original/COMO_SE_EJECUTO.md` | Etapa 2: cómo se generó la referencia |
| `docs/mapa_migracion.md` | Etapa 4: dónde quedó cada función y las omisiones |
| `docs/comparacion_original_vs_modular.md` | Etapa 7: 164 aspectos comparados, todos dentro de tolerancia |
| `docs/resultados_pruebas.txt` | Etapa 7: 60 pruebas pytest, todas aprobadas |
| `README.md` | Etapa 8: instalación, uso, pruebas y cómo ampliar el proyecto |

## 1. Resultado en una tabla

| Programa original | Comando nuevo | ¿Reproduce el original? |
|---|---|---|
| `invernadero_profesor.py` | `python cli.py barrido` | Sí: idéntico al original con pandas 2. Con `--unidad-tiempo original`, idéntico al original con pandas 3 (ver C1) |
| `invernadero_20dias.py` | `python cli.py prediccion` | Sí: idéntico (diferencia máxima 7e-15 °C) |
| `validar_modelo.py` | `python cli.py ventanas` | Sí: las 6 ventanas idénticas |
| — | `python cli.py estrategias` | Nuevo: A, B y C con los mismos datos |
| Interfaces de profesor y 20dias | `python cli.py gui` | Reemplazadas por una interfaz que muestra las 8 etapas |

## 2. Decisiones de diseño

**Arquitectura.** Se siguió la estructura propuesta por la guía con tres cambios
justificados:

1. **`flujos/`** (no estaba en la propuesta). La primera versión de `pipeline.py`
   llegó a 628 líneas: no hacía cálculos, pero sí armaba el texto, las tablas y las
   gráficas de las 8 etapas de 4 funcionalidades. La guía pide que `pipeline.py`
   no sea "otro archivo extenso", así que cada funcionalidad quedó en
   `flujos/<nombre>.py` (85 a 189 líneas) y `pipeline.py` (59 líneas) solo las
   registra y las expone. `flujos/comun.py` tiene las etapas 1 a 3, que son
   iguales en los tres programas.
2. **`modelos/estrategias/identificacion_base.py`** (adicional a `base.py`,
   `ajuste_simulacion.py` y `barrido_threshold.py`). La guía pide que la
   identificación base con PySINDy sea una alternativa diferenciada, así que
   tiene su propio archivo como las otras dos estrategias.
3. **`comparacion/`, `referencia_original/`, `originales/` y `datos_ejemplo/`**:
   guardan los entregables de las etapas 2 y 7, los originales intactos y el CSV,
   separados del código del proyecto.

**Una función compartida solo si se comprobó que es equivalente.** Así se
unificaron la carga, la alineación, la normalización, las métricas, la creación
del modelo PySINDy y la conversión a coeficientes físicos
(`docs/analisis_programas.md`, sección 3). Las funciones que solo se parecían se
mantuvieron separadas, con nombres que dicen cuál es cuál: dos segmentaciones,
dos limpiezas, dos criterios de tramo, tres particiones, dos derivadas y dos
integradores.

**Las tres estrategias son objetos distintos con la misma interfaz**
(`Estrategia.ejecutar()` → `ResultadoModelo`). Cada resultado registra:

* la estrategia (`A_identificacion_base`, `B_ajuste_por_simulacion`, `C_barrido_threshold`),
* sus parámetros,
* con qué datos identificó (`datos_identificacion`) y con qué datos eligió (`datos_seleccion`),
* el resultado del que parte (`origen`).

B parte de A y C repite A, pero sus resultados nunca se mezclan.

**Lo que difiere entre programas se pasa como argumento, no se copia.** Por
ejemplo, la identificación base recibe la derivada como un objeto
(`DerivadaProfesor` o `DerivadaMalla`). El mismo código de PySINDy sirve a los
dos flujos y cada uno conserva su derivada.

**Cálculo y presentación separados.** Los módulos de cálculo no dibujan ni
escriben archivos. Las gráficas son funciones `grafica_x(fig, ...)` que sirven
igual para la interfaz y para guardar PNG. La interfaz no calcula: llama a
`pipeline` en un hilo aparte (la ventana no se congela, a diferencia del
original, E9).

**Trazabilidad.** Cada ejecución crea `resultados/<fecha>_<comando>/` con
`configuracion.json`, `entorno.json` (versiones), `registro.log` (cantidad de
registros antes → después de cada operación) y una carpeta por etapa con
`resumen.txt`, `resumen.json`, tablas CSV y gráficas PNG. Todo se puede revisar
sin abrir el código.

**Parámetros en un solo lugar.** `config.py` define una dataclass por
funcionalidad. Los valores por defecto son los de los originales y se pueden
cambiar por CLI, por un archivo JSON (`--config`) o desde el formulario de la
interfaz.

## 3. Errores detectados

El detalle y las mediciones de E1–E15 están en `docs/analisis_programas.md`,
sección 4. Los que más afectan los resultados:

* **E1 – Unidad de tiempo que depende de la versión de pandas.**
  `astype("int64") / 1e9` supone fechas en nanosegundos; pandas 3 las guarda en
  microsegundos. Resultado: el mismo programa da `35.92 − 0.778·T + …` (pandas 3)
  o `0.0349 − 0.000764·T − …` (pandas 2), con distinto threshold y distintos
  términos.
* **E3 – Métricas de validación no independientes** en el barrido del profesor.
* **E5 – Puntaje del tramo dominado por la luminosidad.** Al dividir por la mediana
  de L (≈ 0 de noche), ese término es ~99 % del puntaje.
* **E10 – PySINDy solo es inestable en el flujo de 20 días.** Con ventana 21,
  aT > 0 en todas las ventanas: todo el trabajo lo hace el ajuste por simulación.

## 4. Correcciones aplicadas

Cada corrección tiene una opción en `config.py` para volver al comportamiento
original y poder compararlo.

| # | Corrección | Opción para el comportamiento original | Efecto con el CSV de ejemplo |
|---|---|---|---|
| C1 | El tiempo se calcula con `total_seconds()`: segundos reales con cualquier versión de pandas (E1) | `--unidad-tiempo original` / `unidad_tiempo="original"` | `barrido` da threshold 0,01, 2 términos, RMSE 2,645 °C (= original con pandas 2). Con pandas 3 el original daba threshold 1e-5, 4 términos, RMSE 2,820 °C |
| C2 | Se pueden leer CSV separados por comas (E6) | — (el CSV tabulado se lee igual que antes) | Ninguno |
| — | Se eliminaron el bucle sin efecto de `cargar_csv` y los imports sin uso (E7) | — | Ninguno |

No se corrigieron, para conservar el comportamiento, E3, E4, E5, E8, E10, E14 y
E15. En su lugar quedan **visibles**: avisos en las etapas 5, 7 y 8, la gráfica de
la derivada en la etapa 6, y opciones para probar alternativas (sección 5).

## 5. Mejoras implementadas

| # | Mejora | Cómo usarla | Resultado |
|---|---|---|---|
| M1 | Flujo de 8 etapas visible y guardado (`resultados/`), con interfaz para recorrerlo | `python cli.py gui` o cualquier comando | — |
| M2 | Elegir el threshold con un conjunto **aparte** (el 20 % final del entrenamiento) para que la validación sea independiente (E3) | `python cli.py barrido --seleccion separado` | Threshold 0,01; validación independiente RMSE 2,629 °C, R² 0,925 |
| M3 | Comparación de las tres estrategias con los mismos datos y la misma validación | `python cli.py estrategias` | Ver tabla abajo |
| M4 | Parámetros editables (CLI, JSON, interfaz) | `python cli.py config prediccion > mi.json` | — |
| M5 | Modo consola sin `tkinter` (E11); `validar_modelo` como función reutilizable (E13) | `python cli.py prediccion` / `ventanas` | — |
| M6 | Versiones fijadas (`requirements.txt` y `requirements-lock.txt`) | `pip install -r requirements.txt` | Resultados reproducibles (E1) |
| M7 | Diagnósticos nuevos: zona horaria, desfases de alineación, huecos rellenados vs interpolados, motivo de selección del tramo | Etapas 2 a 5 | Por ejemplo: T, L y H llegan con la misma marca de tiempo (desfase 0 s en el 100 %); el tramo de 20 días tiene 56 puntos rellenados dentro de huecos de 29 256 |

**Lo que muestra M3** (tramo 18-ago → 28-sep, predicción de los últimos 20 días):

| Estrategia | Ventana SavGol 21 (original) | Ventana SavGol 5 (`--ventana-savgol 5`) |
|---|---|---|
| A – PySINDy base | inestable, diverge | estable, RMSE 5,06 °C, R² 0,53 |
| B – A + ajuste por simulación | RMSE 1,365 °C, R² 0,966 | RMSE 1,365 °C, R² 0,966 |
| C – barrido de threshold (selección independiente) | inestable, diverge | estable, RMSE 5,65 °C, R² 0,41 |

Conclusiones:

* La mejora de B sobre A **se debe al ajuste de coeficientes**, no a la forma de
  la ecuación: los cuatro términos son los mismos.
* B llega al mismo modelo partiendo de una A estable o inestable.
* El barrido de threshold no ayuda en este problema: todos los términos son
  necesarios y quitar términos solo empeora la predicción.

## 6. Dificultades encontradas

* **La misma ecuación del profesor no se pudo reproducir al principio.** Al cambiar
  de pandas 2 a 3 cambiaba el resultado sin ningún error. Hubo que montar dos
  entornos y ejecutar el original en ambos para encontrar la causa (E1).
* **La descripción de la guía no coincide con el código en un punto:**
  `validar_modelo.py` no hace barrido de threshold. Se documentó lo que el código
  hace realmente.
* **Funciones con el mismo propósito y resultados distintos.** Las dos derivadas y
  los dos integradores parecen intercambiables. Se midió la diferencia (los
  integradores difieren hasta 0,13 °C por la tolerancia de RK45) antes de
  decidir no unificarlos.
* **La interfaz del profesor no guarda nada.** Para obtener la referencia hubo que
  manejar sus botones desde un script (con una pantalla virtual en Linux) y
  repetir sus cálculos para volcarlos a disco.
* **El modo consola de 20dias y `validar_modelo.py` necesitan `tkinter`** aunque
  no abran ventanas, porque importan el programa del profesor.

## 7. Mejoras propuestas (no implementadas)

1. **Agregar la temperatura exterior como entrada.** De noche L ≈ 0 y, en el 82 %
   de esas muestras, la humedad está saturada en 100 %. Con las entradas
   constantes, el modelo no puede distinguir una noche fría de una templada: la
   predicción nocturna queda plana (~15,7 °C).
2. **Validación cruzada por bloques sin solapamiento** en lugar de las ventanas
   de `validar_modelo` (E14), para obtener estimaciones independientes del error.
3. **Revisar el puntaje del tramo (E5):** escalar cada variable por su desviación
   o su rango en lugar de por la mediana.
4. **Derivada adecuada a datos irregulares (E4):** remuestrear a malla uniforme
   antes del Savitzky-Golay, como hace el flujo de 20 días.
5. **Interpretar la constante de tiempo:** τ ≈ 3 min es menor que el paso de
   muestreo (5 min), así que el modelo es casi estático:
   T ≈ (a0 + aL·L + aH·H)/(−aT). Como la humedad relativa depende de la propia
   temperatura, parte del buen ajuste viene de esa relación y no de una dinámica
   térmica. Convendría probar sin H, o con humedad absoluta.
6. **Integración continua:** ejecutar `pytest` en GitHub Actions en cada cambio.
