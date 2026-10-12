# Mapa de migración

Dónde quedó cada función de los tres programas originales y si se **unificó**
(una sola versión compartida, porque se comprobó que hacen lo mismo) o se
**conservó separada** (se parecen pero hacen otra cosa). La justificación de
cada decisión está en `docs/analisis_programas.md`, sección 3.

Leyenda: **=** copia fiel · **U** unificada · **S** separada a propósito ·
**D** dividida en partes · **N** nueva · **R** reemplazada

## `invernadero_profesor.py`

| Función original | Destino | | Nota |
|---|---|---|---|
| `VARIABLES`, `VARIABLES_ALIASES` | `config.py` | = | |
| `encontrar_columna`, `detectar_columnas` | `datos/carga.py` | = | |
| `cargar_csv` | `datos/carga.py` (`leer_csv_crudo` + `cargar_csv`) y `datos/transformacion.py` (`convertir_fechas`) | D | La lectura cruda quedó separada de la conversión de fechas para inspeccionar las etapas 1 y 2. Corrección C2 (CSV con comas) y se quitó el bucle sin efecto |
| `construir_series` | `datos/transformacion.py` | = U | La usaban los tres programas |
| `estimar_dt_median`, `alinear_variables` | `datos/alineacion.py` | = U | Los tres programas usaban la misma. Ahora la tolerancia (2 × dt) es un parámetro |
| `obtener_bloques_continuos` | `datos/segmentacion.py` | = S | Distinta de `bloques_rellenados` |
| `calcular_excitación` | `analisis/seleccion_tramo.py` (`calcular_excitacion`; se conserva el alias con tilde) | = | |
| `puntuar_bloque` | `analisis/seleccion_tramo.py` | = | |
| `seleccionar_mejor_bloque` | `analisis/seleccion_tramo.py` | = S | Ahora recibe los bloques ya segmentados (antes segmentaba adentro) y devuelve también el motivo de la elección |
| `limpiar_tramo` | `datos/limpieza.py` | = S | Distinta del relleno en malla |
| `Escalador` | `analisis/normalizacion.py` | = U | La usaban profesor y 20dias. Se añadió `tabla()` para mostrar los parámetros |
| `calcular_derivada_temperatura` | `analisis/derivadas.py` | = S | Corrección C1: el tiempo se calcula con `tiempo_absoluto(unidad)`; con `unidad="original"` es la expresión original |
| `dividir_train_validacion` | `analisis/particion.py` | = S | |
| `crear_modelo` | `modelos/sindy.py` | U | Igual al modelo que 20dias creaba en línea |
| `identificar_modelo` | `modelos/sindy.py` (`ajustar_sindy`) + `modelos/estrategias/identificacion_base.py` (`IdentificacionBase` con `DerivadaProfesor`) | D U | La parte común (normalizar y ajustar) se comparte con 20dias; la derivada propia se elige con `DerivadaProfesor` |
| `obtener_coeficientes_fisicos` | `modelos/coeficientes.py` | U | Llama a `a_fisico` (misma fórmula) |
| `ecuacion_fisica` | `modelos/coeficientes.py` | = | Acepta además la unidad de tiempo para mostrarla |
| `simular_modelo` (+ `rhs`, `interpolar`) | `simulacion/integrador.py` | = S | RK45; distinto de `simular_lsim` |
| `calcular_metricas` | `evaluacion/metricas.py` | = U | La usaban los tres programas |
| `buscar_mejor_modelo` | `modelos/estrategias/barrido_threshold.py` (`BarridoThreshold`) + `modelos/seleccion.py` | D | Estrategia C. Guarda con qué datos eligió y si la selección es independiente. Mejora M2 (selección separada) opcional |
| Texto de resultados de `ejecutar_sindy` | `evaluacion/reportes.py` (`texto_barrido`) y `pipeline.ejecutar_barrido` | D | |
| Gráfica de validación de `ejecutar_sindy` | `visualizacion/modelos.py` (`grafica_validacion_profesor`) | = | |
| `actualizar_grafica` (variables en el tiempo) | `visualizacion/datos.py` (`grafica_variables`) | D | Un panel por variable, con unidades |
| Clase `Aplicacion` | `interfaz/aplicacion.py` | R | Nueva interfaz que muestra las 8 etapas (ver "Omisiones") |
| Orden de los botones (abrir → tramo → identificar) | `pipeline.ejecutar_barrido` | D | |

## `invernadero_20dias.py`

| Función original | Destino | | Nota |
|---|---|---|---|
| Constantes (`DIAS_PREDICCION`, `MAX_HUECO_HORAS`, `PASO`, `THRESHOLD`, `ZONA_HORARIA`, …) | `config.py` (`ConfigPrediccion`) | = | Ahora se cambian desde la CLI, un JSON o la interfaz |
| `bloques_rellenados` | `datos/segmentacion.py` (corte) + `datos/limpieza.py` (`rellenar_en_malla`) | D S | Mismo cálculo. Distinta de `obtener_bloques_continuos` |
| `elegir_bloque` | `analisis/seleccion_tramo.py` | = S | Devuelve también el motivo |
| `dividir` | `analisis/particion.py` (`dividir_ultimos_dias`) | = S | |
| `a_fisico` | `modelos/coeficientes.py` | U | |
| `simular` | `simulacion/integrador.py` (`simular_lsim`) | = S | El paso `dt` es un argumento (antes la constante `DT_H`) |
| `identificar` (1.ª parte: derivada + PySINDy) | `modelos/estrategias/identificacion_base.py` (`IdentificacionBase` con `DerivadaMalla`) + `modelos/sindy.py` + `analisis/derivadas.py` (`derivada_savgol_directa`) | D U | Estrategia A |
| `identificar` (2.ª parte: `least_squares`) | `modelos/estrategias/ajuste_simulacion.py` (`AjusteSimulacion`) | D | Estrategia B, mismas restricciones |
| `mae_por_dia` | `evaluacion/metricas.py` | = | Recibe las muestras por día como argumento |
| `por_segundo` | `modelos/coeficientes.py` | = | |
| `evaluar` | `pipeline.evaluar_prediccion` | = | Coordina A, B, la simulación y las métricas; también la usa `ventanas` |
| Referencia ingenua (dentro de `evaluar`) | `simulacion/prediccion.py` (`perfil_diario`) | D | |
| `simular_archivo` | `simulacion/prediccion.py` | = | |
| `metricas_archivo` | `evaluacion/metricas.py` | = | |
| `resumen`, `resumen_archivo` | `evaluacion/reportes.py` (`resumen_prediccion`, `resumen_archivo`) | = | Indica además qué estrategia produjo la ecuación final |
| `_por_tramos`, `_sombrear_entrenamiento`, las 7 `grafica_*`, `GRAFICAS` | `visualizacion/modelos.py` (`GRAFICAS_PREDICCION`) | = | Las constantes vienen en `r` en lugar de variables globales |
| `Aplicacion20` | `interfaz/aplicacion.py` | R | El selector de gráficas y la barra de matplotlib se conservan |
| `modo_consola` | `cli.py prediccion` + `pipeline.ejecutar_prediccion` | R | Ya no necesita `tkinter` |

## `validar_modelo.py`

| Parte original | Destino | | Nota |
|---|---|---|---|
| Filtro de tramos ≥ 25 días y cortes cada 4 días | `analisis/particion.py` (`cortes_ventanas`, `recortar`) + `pipeline.ejecutar_ventanas` | D | |
| Bucle de evaluación | `pipeline.ejecutar_ventanas` (usa `evaluar_prediccion`) | = | |
| Tabla en consola | `evaluacion/reportes.py` (`tabla_ventanas`) | = | |
| Gráfica por ventana y gráficas 1 a 4 | `visualizacion/modelos.py` (`grafica_ventana`, `grafica_resumen_ventanas`, `grafica_error_por_dia_ventanas`, `grafica_real_vs_predicho_ventanas`, `grafica_errores_ventanas`) | = | |
| Ejecución al importar, rutas fijas | `cli.py ventanas` | R | |

## Código nuevo (no existía en los originales)

| Archivo / función | Para qué |
|---|---|
| `config.py` (dataclasses, JSON) | Centralizar parámetros, rutas y opciones |
| `datos/carga.validar_estructura` | Validación básica con mensajes claros |
| `datos/transformacion.analizar_zona_horaria`, `resumen_construccion` | Etapa 2: documentar la zona horaria y los registros descartados |
| `datos/alineacion.diferencias_temporales`, `resumen_alineacion` | Etapa 3: desfases entre variables |
| `datos/limpieza.resumen_limpieza_tramo`, `resumen_relleno`, `puntos_en_huecos` | Etapa 4: qué se conservó, interpoló o descartó |
| `datos/segmentacion.listar_huecos`, `tabla_bloques` | Etapa 4: ubicación de los huecos |
| `analisis/particion.separar_seleccion` | Mejora M2 |
| `analisis/derivadas.tiempo_absoluto`, `unidad_efectiva` | Corrección C1 |
| `modelos/estrategias/base.py` | Interfaz común y `ResultadoModelo`, que registra estrategia, parámetros y datos usados |
| `evaluacion/reportes.py` (`Etapa`, `Ejecucion`, `RegistroEjecucion`) | Guardar cada etapa en `resultados/<fecha>_<comando>/` |
| `visualizacion/datos.py`, `visualizacion/transformaciones.py` | Gráficas de las etapas 1 a 6 |
| `pipeline.comparar_estrategias`, `cli.py estrategias` | Comparar A, B y C sobre los mismos datos |
| `referencia_original/ejecutar_referencia.py` | Etapa 2 |
| `comparacion/comparar_con_original.py` | Etapa 7 |

## Omisiones declaradas

* **Casillas de variables y botón "Mostrar todo"** de la interfaz del profesor:
  no se reprodujeron. Las tres variables se ven siempre, una por panel, en las
  etapas 1, 3 y 5, y la barra de matplotlib permite acercar cualquier zona.
* **Botón separado "Buscar mejor tramo"**: en la nueva interfaz un solo botón
  *Ejecutar* corre todo el flujo. El tramo elegido y su motivo se ven en la etapa 5.
* **Títulos de dos gráficas** de 20dias y validar_modelo afirmaban
  "no crece con el horizonte". Ahora describen lo que muestran
  ("Error de cada día de la predicción"), porque es una conclusión, no un
  título, y no se cumple en todas las ventanas: en V1 el MAE medio de los últimos
  5 días (1,62 °C) es mayor que el de los primeros 5 (1,02 °C) y su peor día es el 17.
* Imports sin uso del profesor (`SpanSelector`, `ZoneInfo`): eliminados.
