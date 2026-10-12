# Comparación: programas originales vs versión modular

Generado por `python comparacion/comparar_con_original.py` (2026-10-12 16:40, 44 s).

Se ejecutó la versión modular con el mismo CSV y la configuración por defecto, y se
comparó con lo que guardaron los programas originales en `referencia_original/`
(ver `referencia_original/COMO_SE_EJECUTO.md`).

Entorno de la referencia: Python 3.13.14, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, pysindy 2.1.0.

## Tolerancias

| Tipo | Tolerancia | Por qué |
|---|---|---|
| Conteos, fechas, parámetros, threshold | igualdad exacta | son enteros o texto |
| Coeficientes de PySINDy y conversiones | 1e-09 relativo | mismo cálculo en el mismo orden: debe dar el mismo número |
| Coeficientes ajustados (least_squares) y sus métricas | 1e-06 relativo | optimizador iterativo: el mismo programa original da diferencias ~1e-7 al cambiar solo la versión de pandas |
| Trayectorias simuladas | 1e-06 °C | muy por debajo de la resolución del sensor (0,01 °C) |

## Resumen

| Comparación | Aspectos | Coinciden |
|---|---|---|
| invernadero_profesor.py vs barrido (unidad_tiempo='original') | 35 | 35 ✅ |
| invernadero_profesor.py vs barrido (unidad_tiempo='s' vs original con pandas 2) | 35 | 35 ✅ |
| invernadero_20dias.py vs prediccion | 33 | 33 ✅ |
| validar_modelo.py vs ventanas | 61 | 61 ✅ |

## invernadero_profesor.py vs barrido (unidad_tiempo='original')

| Aspecto | Original | Modular | Diferencia | Tolerancia | OK |
|---|---|---|---|---|---|
| filas CSV | 147927 | 147927 | - | igual | ✅ |
| muestras por variable | {'Temperatura': 29585, 'Luminosidad': 29586, 'Humedad': 29585} | {'Temperatura': 29585, 'Luminosidad': 29586, 'Humedad': 29585} | - | igual | ✅ |
| filas alineadas | 29579 | 29579 | - | igual | ✅ |
| duplicados eliminados | 6 | 6 | - | igual | ✅ |
| muestras del tramo | 3027 | 3027 | - | igual | ✅ |
| muestras tramo limpio | 3027 | 3027 | - | igual | ✅ |
| muestras entrenamiento | 1494 | 1494 | - | igual | ✅ |
| muestras validación | 1533 | 1533 | - | igual | ✅ |
| inicio del tramo | 2026-06-15T15:36:17+00:00 | 2026-06-15T15:36:17+00:00 | - | igual | ✅ |
| fin del tramo | 2026-06-25T22:03:00+00:00 | 2026-06-25T22:03:00+00:00 | - | igual | ✅ |
| puntaje del tramo | 282.3202528 | 282.3202528 | 0.00e+00 rel | 1e-09 rel | ✅ |
| thresholds probados | [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01] | [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01] | - | igual | ✅ |
| threshold elegido | 1e-05 | 1e-05 | - | igual | ✅ |
| coeficiente a0 | 35.91859859 | 35.91859859 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coeficiente aT | -0.7775548613 | -0.7775548613 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coeficiente aL | 5.190557155e-05 | 5.190557155e-05 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coeficiente aH | -0.2341762053 | -0.2341762053 | 0.00e+00 rel | 1e-09 rel | ✅ |
| RMSE barrido th=1e-05 | 2.820347456 | 2.820347456 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=1e-05 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=3e-05 | 2.820347456 | 2.820347456 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=3e-05 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=0.0001 | 2.820347456 | 2.820347456 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.0001 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=0.0003 | 2.820347456 | 2.820347456 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.0003 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=0.001 | 2.820347456 | 2.820347456 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.001 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=0.003 | 2.820347456 | 2.820347456 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.003 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=0.01 | 2.820347456 | 2.820347456 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.01 | 4 | 4 | - | igual | ✅ |
| trayectoria simulada (validación) | 1533 valores | 1533 valores | 7.11e-15 °C máx | 1e-06 °C | ✅ |
| métrica RMSE | 2.820347456 | 2.820347456 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica MAE | 2.051577883 | 2.051577883 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica R2 | 0.9136682017 | 0.9136682017 | 0.00e+00 rel | 1e-06 rel | ✅ |

## invernadero_profesor.py vs barrido (unidad_tiempo='s' vs original con pandas 2)

| Aspecto | Original | Modular | Diferencia | Tolerancia | OK |
|---|---|---|---|---|---|
| filas CSV | 147927 | 147927 | - | igual | ✅ |
| muestras por variable | {'Temperatura': 29585, 'Luminosidad': 29586, 'Humedad': 29585} | {'Temperatura': 29585, 'Luminosidad': 29586, 'Humedad': 29585} | - | igual | ✅ |
| filas alineadas | 29579 | 29579 | - | igual | ✅ |
| duplicados eliminados | 6 | 6 | - | igual | ✅ |
| muestras del tramo | 3027 | 3027 | - | igual | ✅ |
| muestras tramo limpio | 3027 | 3027 | - | igual | ✅ |
| muestras entrenamiento | 1494 | 1494 | - | igual | ✅ |
| muestras validación | 1533 | 1533 | - | igual | ✅ |
| inicio del tramo | 2026-06-15T15:36:17+00:00 | 2026-06-15T15:36:17+00:00 | - | igual | ✅ |
| fin del tramo | 2026-06-25T22:03:00+00:00 | 2026-06-25T22:03:00+00:00 | - | igual | ✅ |
| puntaje del tramo | 282.3202528 | 282.3202528 | 0.00e+00 rel | 1e-09 rel | ✅ |
| thresholds probados | [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01] | [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01] | - | igual | ✅ |
| threshold elegido | 0.01 | 0.01 | - | igual | ✅ |
| coeficiente a0 | 0.0348599055 | 0.0348599055 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coeficiente aT | -0.0007635394556 | -0.0007635394556 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coeficiente aL | 0 | 0 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coeficiente aH | -0.0002349260559 | -0.0002349260559 | 0.00e+00 rel | 1e-09 rel | ✅ |
| RMSE barrido th=1e-05 | 2.820392301 | 2.820392301 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=1e-05 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=3e-05 | 2.820392301 | 2.820392301 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=3e-05 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=0.0001 | 2.820392301 | 2.820392301 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.0001 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=0.0003 | 2.820392301 | 2.820392301 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.0003 | 4 | 4 | - | igual | ✅ |
| RMSE barrido th=0.001 | 2.846528632 | 2.846528632 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.001 | 3 | 3 | - | igual | ✅ |
| RMSE barrido th=0.003 | 2.846528632 | 2.846528632 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.003 | 3 | 3 | - | igual | ✅ |
| RMSE barrido th=0.01 | 2.644835054 | 2.644835054 | 0.00e+00 rel | 1e-06 rel | ✅ |
| términos th=0.01 | 2 | 2 | - | igual | ✅ |
| trayectoria simulada (validación) | 1533 valores | 1533 valores | 7.11e-15 °C máx | 1e-06 °C | ✅ |
| métrica RMSE | 2.644835054 | 2.644835054 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica MAE | 2.011173645 | 2.011173645 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica R2 | 0.9240788563 | 0.9240788563 | 0.00e+00 rel | 1e-06 rel | ✅ |

## invernadero_20dias.py vs prediccion

| Aspecto | Original | Modular | Diferencia | Tolerancia | OK |
|---|---|---|---|---|---|
| filas alineadas | 29579 | 29579 | - | igual | ✅ |
| tramos rellenados | 5 | 5 | - | igual | ✅ |
| muestras del tramo | 11837 | 11837 | - | igual | ✅ |
| muestras entrenamiento | 6077 | 6077 | - | igual | ✅ |
| muestras predicción | 5760 | 5760 | - | igual | ✅ |
| inicio del tramo | 2026-08-18T13:05:00+00:00 | 2026-08-18T13:05:00+00:00 | - | igual | ✅ |
| fin del tramo | 2026-09-28T15:25:00+00:00 | 2026-09-28T15:25:00+00:00 | - | igual | ✅ |
| parámetros | {'THRESHOLD': 0.001, 'MAX_HUECO_HORAS': 2, 'DIAS_PREDICCION': 20} | {'THRESHOLD': 0.001, 'MAX_HUECO_HORAS': 2.0, 'DIAS_PREDICCION': 20.0} | - | igual | ✅ |
| coef. PySINDy (/h) a0 | -8.792055147 | -8.792055147 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coef. PySINDy (/h) aT | 0.1484967133 | 0.1484967133 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coef. PySINDy (/h) aL | 0.0001714638272 | 0.0001714638272 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coef. PySINDy (/h) aH | 0.06723541877 | 0.06723541877 | 0.00e+00 rel | 1e-09 rel | ✅ |
| coef. ajustado (/h) a0 | 584.4918723 | 584.4918723 | 0.00e+00 rel | 1e-06 rel | ✅ |
| coef. ajustado (/h) aT | -17.99452548 | -17.99452548 | 0.00e+00 rel | 1e-06 rel | ✅ |
| coef. ajustado (/h) aL | 0.01886589654 | 0.01886589654 | 0.00e+00 rel | 1e-06 rel | ✅ |
| coef. ajustado (/h) aH | -3.026742138 | -3.026742138 | 0.00e+00 rel | 1e-06 rel | ✅ |
| trayectoria predicción 20 días | 5760 valores | 5760 valores | 7.11e-15 °C máx | 1e-06 °C | ✅ |
| referencia perfil diario | 5760 valores | 5760 valores | 7.11e-15 °C máx | 1e-06 °C | ✅ |
| trayectoria todo el archivo | 29256 valores | 29256 valores | 7.11e-15 °C máx | 1e-06 °C | ✅ |
| métrica predicción RMSE | 1.364947093 | 1.364947093 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica predicción MAE | 1.049217826 | 1.049217826 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica predicción R2 | 0.9657044688 | 0.9657044688 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica referencia RMSE | 3.922968271 | 3.922968271 | 0.00e+00 rel | 1e-09 rel | ✅ |
| métrica referencia MAE | 2.363285377 | 2.363285377 | 0.00e+00 rel | 1e-09 rel | ✅ |
| métrica referencia R2 | 0.7167072316 | 0.7167072316 | 0.00e+00 rel | 1e-09 rel | ✅ |
| métrica archivo RMSE | 1.628275998 | 1.628275998 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica archivo MAE | 1.213962115 | 1.213962115 | 0.00e+00 rel | 1e-06 rel | ✅ |
| métrica archivo R2 | 0.9635687161 | 0.9635687161 | 0.00e+00 rel | 1e-06 rel | ✅ |
| MAE 2026-06 | 1.397446032 | 1.397446032 | 0.00e+00 rel | 1e-06 rel | ✅ |
| MAE 2026-07 | 1.152692411 | 1.152692411 | 0.00e+00 rel | 1e-06 rel | ✅ |
| MAE 2026-08 | 1.368380194 | 1.368380194 | 0.00e+00 rel | 1e-06 rel | ✅ |
| MAE 2026-09 | 1.049742582 | 1.049742582 | 0.00e+00 rel | 1e-06 rel | ✅ |
| MAE por día | 20 valores | 20 valores | 0.00e+00 °C máx | 1e-06 °C | ✅ |

## validar_modelo.py vs ventanas

| Aspecto | Original | Modular | Diferencia | Tolerancia | OK |
|---|---|---|---|---|---|
| cantidad de ventanas | 6 | 6 | - | igual | ✅ |
| V1 nombre | 27-Jul a 16-Aug | 27-Jul a 16-Aug | - | igual | ✅ |
| V1 inicio predicción | 2026-07-27T10:25:00+00:00 | 2026-07-27T10:25:00+00:00 | - | igual | ✅ |
| V1 coef. ajustado a0 | 520.0914903 | 520.0914903 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V1 coef. ajustado aT | -16.41266303 | -16.41266303 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V1 coef. ajustado aL | 0.01591142067 | 0.01591142067 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V1 coef. ajustado aH | -2.622501451 | -2.622501451 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V1 métrica RMSE | 1.619735559 | 1.619735559 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V1 métrica MAE | 1.262260449 | 1.262260449 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V1 métrica R2 | 0.9695660809 | 0.9695660809 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V1 peor día MAE | 2.279244506 | 2.279244506 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V2 nombre | 23-Aug a 12-Sep | 23-Aug a 12-Sep | - | igual | ✅ |
| V2 inicio predicción | 2026-08-23T13:10:00+00:00 | 2026-08-23T13:10:00+00:00 | - | igual | ✅ |
| V2 coef. ajustado a0 | 493.9143058 | 493.9143058 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V2 coef. ajustado aT | -15.13133067 | -15.13133067 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V2 coef. ajustado aL | 0.01592729039 | 0.01592729039 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V2 coef. ajustado aH | -2.627333454 | -2.627333454 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V2 métrica RMSE | 1.437736746 | 1.437736746 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V2 métrica MAE | 1.139815143 | 1.139815143 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V2 métrica R2 | 0.9735528531 | 0.9735528531 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V2 peor día MAE | 1.512501739 | 1.512501739 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V3 nombre | 27-Aug a 16-Sep | 27-Aug a 16-Sep | - | igual | ✅ |
| V3 inicio predicción | 2026-08-27T13:10:00+00:00 | 2026-08-27T13:10:00+00:00 | - | igual | ✅ |
| V3 coef. ajustado a0 | 571.9403646 | 571.9403646 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V3 coef. ajustado aT | -17.5797225 | -17.5797225 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V3 coef. ajustado aL | 0.01823023194 | 0.01823023194 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V3 coef. ajustado aH | -2.968786184 | -2.968786184 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V3 métrica RMSE | 1.413509709 | 1.413509709 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V3 métrica MAE | 1.097813431 | 1.097813431 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V3 métrica R2 | 0.9741541315 | 0.9741541315 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V3 peor día MAE | 1.698650383 | 1.698650383 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V4 nombre | 31-Aug a 20-Sep | 31-Aug a 20-Sep | - | igual | ✅ |
| V4 inicio predicción | 2026-08-31T13:10:00+00:00 | 2026-08-31T13:10:00+00:00 | - | igual | ✅ |
| V4 coef. ajustado a0 | 565.8460023 | 565.8460023 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V4 coef. ajustado aT | -17.39622175 | -17.39622175 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V4 coef. ajustado aL | 0.01837072637 | 0.01837072637 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V4 coef. ajustado aH | -2.895528142 | -2.895528142 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V4 métrica RMSE | 1.513213353 | 1.513213353 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V4 métrica MAE | 1.180826003 | 1.180826003 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V4 métrica R2 | 0.9684422772 | 0.9684422772 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V4 peor día MAE | 1.902693686 | 1.902693686 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V5 nombre | 04-Sep a 24-Sep | 04-Sep a 24-Sep | - | igual | ✅ |
| V5 inicio predicción | 2026-09-04T13:10:00+00:00 | 2026-09-04T13:10:00+00:00 | - | igual | ✅ |
| V5 coef. ajustado a0 | 633.5678965 | 633.5678965 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V5 coef. ajustado aT | -19.18551845 | -19.18551845 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V5 coef. ajustado aL | 0.01937151297 | 0.01937151297 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V5 coef. ajustado aH | -3.283283708 | -3.283283708 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V5 métrica RMSE | 1.506190502 | 1.506190502 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V5 métrica MAE | 1.181666612 | 1.181666612 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V5 métrica R2 | 0.9635346562 | 0.9635346562 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V5 peor día MAE | 1.956677777 | 1.956677777 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V6 nombre | 08-Sep a 28-Sep | 08-Sep a 28-Sep | - | igual | ✅ |
| V6 inicio predicción | 2026-09-08T15:25:00+00:00 | 2026-09-08T15:25:00+00:00 | - | igual | ✅ |
| V6 coef. ajustado a0 | 584.0261886 | 584.0261886 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V6 coef. ajustado aT | -17.98236793 | -17.98236793 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V6 coef. ajustado aL | 0.01885779903 | 0.01885779903 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V6 coef. ajustado aH | -3.02397105 | -3.02397105 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V6 métrica RMSE | 1.364943851 | 1.364943851 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V6 métrica MAE | 1.049209291 | 1.049209291 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V6 métrica R2 | 0.9657118024 | 0.9657118024 | 0.00e+00 rel | 1e-06 rel | ✅ |
| V6 peor día MAE | 1.774797637 | 1.774797637 | 0.00e+00 rel | 1e-06 rel | ✅ |

## Funcionalidades y gráficas

| Original | En la versión modular | Estado |
|---|---|---|
| profesor: cargar CSV, alinear, elegir tramo, barrido de threshold, ecuación y métricas | `python cli.py barrido` / interfaz | Conservado (tabla de arriba) |
| profesor: gráfica de variables | Etapas 1, 3 y 5 (un panel por variable) | Conservado; sin las casillas para ocultar variables |
| profesor: gráfica de validación | Etapa 8, `validacion.png` | Conservado |
| 20dias: predicción de 20 días, referencia ingenua, simulación de todo el archivo, métricas por mes y por día | `python cli.py prediccion` / interfaz | Conservado (tabla de arriba) |
| 20dias: 7 gráficas y selector de gráficas | Etapa 8 de `prediccion`; selector "Gráfica" en la interfaz | Conservado; títulos de 1 gráfica ajustados (ver `docs/mapa_migracion.md`) |
| 20dias: modo consola | `python cli.py prediccion` | Conservado; ya no necesita tkinter |
| validar_modelo: 6 ventanas, tabla y 10 gráficas | `python cli.py ventanas`, etapa 8 | Conservado; título de 1 gráfica ajustado |

## Diferencias explicadas

1. **Unidad de tiempo del barrido (corrección C1).** Por defecto la versión
   modular calcula el tiempo en segundos reales (`unidad_tiempo="s"`). Con
   pandas 3, el original calculaba kilosegundos. Por eso, con la configuración
   por defecto:
   * `python cli.py barrido` da el mismo resultado que el **original con
     pandas 2** (threshold 0,01, 2 términos, RMSE 2,645 °C). Comparación
     "unidad_tiempo='s'" de arriba.
   * `python cli.py barrido --unidad-tiempo original` da el mismo resultado que
     el **original con pandas 3** (threshold 1e-5, 4 términos, RMSE 2,820 °C).
     Comparación "unidad_tiempo='original'" de arriba.
2. **Etiqueta de unidades en la ecuación.** La ecuación se imprime igual que en el
   original, más la unidad de tiempo al final (`[°C/s]`, `[°C/ks]` o `[°C/h]`).
   Solo es presentación.
3. **Archivos de salida.** Los originales no guardaban (profesor) o guardaban
   solo PNG (20dias, validar_modelo). La versión modular guarda además, por
   etapa, tablas CSV, resúmenes JSON y un registro. No cambia ningún resultado.
4. **Ninguna otra diferencia numérica:** la mayor diferencia medida en
   trayectorias es 7e-15 °C y en coeficientes 0 (ver las tablas).

