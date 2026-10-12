"""
Piezas compartidas por todos los flujos:

  * Contexto: agrupa la Ejecucion en memoria, el registro en disco
    (resultados/<fecha>_<comando>/) y el aviso de progreso.
  * etapas_datos(): etapas 1 a 3 (datos originales, transformación y
    alineación), idénticas en los tres programas originales.
"""

from __future__ import annotations

from functools import partial

import pandas as pd

import config as C
from datos import alineacion, carga, transformacion
from evaluacion import reportes as R
from visualizacion import datos as VD




class Contexto:
    """Agrupa la Ejecucion en memoria, el registro en disco y el aviso de progreso."""

    def __init__(self, comando, cfg, guardar, carpeta, aviso, graficas=True):
        self.cfg = cfg
        self.ejec = R.Ejecucion(comando=comando, config=C.a_dict(cfg))
        self.reg = R.RegistroEjecucion(carpeta or C.CARPETA_RESULTADOS, comando,
                                       self.ejec.config, guardar, graficas)
        self.ejec.carpeta = self.reg.carpeta
        self.aviso = aviso or (lambda s: None)

    def etapa(self, e: R.Etapa):
        self.ejec.etapas.append(e)
        self.reg.info(e.numero, e.titulo)
        self.reg.guardar_etapa(e)
        self.aviso(f"Etapa {e.numero}: {e.titulo}")

    def cerrar(self, texto: str):
        self.ejec.resultado["resumen"] = texto
        self.reg.cerrar(texto, self.ejec.resultado)
        if self.reg.carpeta:
            self.aviso(f"Resultados guardados en {self.reg.carpeta}")
        return self.ejec


def paso_malla(cfg) -> pd.Timedelta:
    return pd.Timedelta(minutes=cfg.paso_minutos)


# ============================================================
# Etapas 1-3 (comunes a todos los flujos)
# ============================================================

def etapas_datos(ctx: Contexto, ruta_csv, tolerancia: float):
    """Lee, transforma y alinea. Devuelve (datos_alineados, dt_referencia)."""
    reg = ctx.reg

    # ---------- Etapa 1: datos originales
    crudo = carga.leer_csv_crudo(ruta_csv)
    info = carga.validar_estructura(crudo)
    reg.info(1, f"Leído {ruta_csv}: {info['filas']} filas, columnas {info['columnas']}")
    tipos = pd.DataFrame({"columna": crudo.columns, "tipo": crudo.dtypes.astype(str).values,
                          "nulos": crudo.isna().sum().values,
                          "ejemplo": [crudo[c].iloc[0] for c in crudo.columns]})

    def series_crudas():
        s = {}
        for v in C.VARIABLES:
            d = crudo[crudo["metric_name"] == v]
            s[v] = pd.DataFrame({"datetime_utc": pd.to_datetime(d["datetime_utc"], utc=True, errors="coerce"),
                                 v: pd.to_numeric(d["number_value"], errors="coerce")})
        return s

    ctx.etapa(R.Etapa(
        1, "Datos originales",
        texto=f"Archivo: {ruta_csv}\nFormato: {info['formato']}\nFilas: {info['filas']}\n\n"
              f"Tipos de columna:\n{tipos.to_string(index=False)}\n\n"
              f"Filas por métrica:\n{R.texto_dict(info.get('metricas', {}))}\n\n"
              f"Primeras filas:\n{crudo.head(8).to_string()}",
        datos=info,
        tablas={"primeras_filas": crudo.head(50), "tipos_de_columna": tipos},
        graficas={"filas_por_metrica": partial(VD.grafica_registros_por_metrica, df_crudo=crudo),
                  "series_originales": lambda fig: VD.grafica_series_originales(fig, series_crudas())},
    ))

    # ---------- Etapa 2: transformación
    conv, info_f = transformacion.convertir_fechas(crudo)
    series = transformacion.construir_series(conv)
    info_s = transformacion.resumen_construccion(conv, series)
    reg.info(2, info_f["zona_horaria"]["conclusion"])
    reg.cantidades(2, "Filas del CSV -> filas con fecha y valor válidos",
                   info_s["filas_entrada"], info_s["filas_entrada"] - info_s["filas_sin_fecha_o_valor"])
    for v, n in info_s["muestras_por_variable"].items():
        reg.info(2, f"Serie {v}: {n} muestras")
    antes_despues = pd.DataFrame({"texto_original": crudo["datetime_utc"].head(20).values,
                                  "fecha_convertida_utc": conv["datetime_utc"].head(20).values,
                                  "metrica": crudo["metric_name"].head(20).values})
    ctx.etapa(R.Etapa(
        2, "Transformación",
        texto=f"Conversión de fechas:\n{R.texto_dict(info_f)}\n\nConstrucción de series:\n"
              f"{R.texto_dict(info_s)}\n\nAntes / después (primeras filas):\n{antes_despues.head(8).to_string(index=False)}",
        datos={"conversion_fechas": info_f, "construccion_series": info_s},
        tablas={"fechas_antes_despues": antes_despues,
                "registros": pd.DataFrame([{"paso": "filas CSV", "registros": info_s["filas_entrada"]},
                                           {"paso": "sin fecha o valor", "registros": info_s["filas_sin_fecha_o_valor"]}]
                                          + [{"paso": f"métrica ignorada: {k}", "registros": v}
                                             for k, v in info_s["metricas_ignoradas"].items()]
                                          + [{"paso": f"serie {k}", "registros": v}
                                             for k, v in info_s["muestras_por_variable"].items()])},
        graficas={"conversion_fechas": partial(VD.grafica_conversion_fechas, df_crudo=crudo, df_conv=conv),
                  "construccion_series": partial(VD.grafica_construccion_series, resumen=info_s)},
    ))

    # ---------- Etapa 3: alineación
    datos, dt_ref = alineacion.alinear_variables(series, tolerancia)
    difs = alineacion.diferencias_temporales(series)
    info_a = alineacion.resumen_alineacion(series, datos, dt_ref)
    reg.cantidades(3, "Muestras de T -> filas alineadas", info_a["filas_temperatura"], info_a["filas_alineadas"])
    desc = difs.drop(columns="datetime_utc").describe().T
    ctx.etapa(R.Etapa(
        3, "Alineación",
        texto=f"Criterio: a cada instante de Temperatura se le asigna la L y la H más cercanas "
              f"(merge_asof 'nearest'), si están a menos de {tolerancia:g} x dt típico.\n\n"
              f"{R.texto_dict(info_a)}\n\nDesfases [s] respecto a T:\n{desc.to_string()}",
        datos=info_a,
        tablas={"datos_alineados": datos, "desfases_resumen": desc.reset_index(names="variable")},
        graficas={"desfases": partial(VD.grafica_desfases, difs=difs, tolerancia_s=info_a["tolerancia_s"]),
                  "datos_alineados": partial(VD.grafica_variables, datos=datos,
                                             titulo="Etapa 3 - Variables alineadas a las marcas de Temperatura")},
    ))
    return datos, dt_ref
