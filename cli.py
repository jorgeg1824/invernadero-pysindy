"""
Punto de entrada por terminal.

    python cli.py --help
    python cli.py barrido        # invernadero_profesor.py : barrido de threshold
    python cli.py prediccion     # invernadero_20dias.py   : PySINDy + ajuste por simulación
    python cli.py ventanas       # validar_modelo.py       : predicción en varias ventanas
    python cli.py estrategias    # nuevo: compara A, B y C con los mismos datos
    python cli.py gui            # interfaz gráfica para recorrer las 8 etapas
    python cli.py config barrido # imprime la configuración por defecto (JSON)

Todas las funcionalidades aceptan --csv, --config, --salida y --sin-guardar.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import config as C

DESCRIPCION = """Identificación de un modelo de temperatura de invernadero con PySINDy.

Funcionalidades (programa original entre paréntesis):
  barrido      Estrategia C: identificación base (A) con varios threshold y
               elección del de menor RMSE de simulación.       (invernadero_profesor.py)
  prediccion   Estrategias A + B: PySINDy y ajuste de coeficientes por simulación;
               predice los últimos N días y simula todo el archivo. (invernadero_20dias.py)
  ventanas     Repite 'prediccion' en varias ventanas de 20 días.  (validar_modelo.py)
  estrategias  Compara A, B y C sobre la misma validación.         (nuevo)
  gui          Interfaz gráfica.
  config       Muestra la configuración por defecto de un flujo.

Los resultados de cada ejecución quedan en resultados/<fecha>_<comando>/ con una
carpeta por etapa (01_datos_originales ... 08_simulacion_evaluacion).
"""


def _comunes(p):
    p.add_argument("--csv", help=f"archivo de datos (por defecto {C.CSV_EJEMPLO.name})")
    p.add_argument("--config", help="JSON con parámetros (ver: python cli.py config <flujo>)")
    p.add_argument("--salida", help=f"carpeta de resultados (por defecto {C.CARPETA_RESULTADOS.name}/)")
    p.add_argument("--sin-guardar", action="store_true", help="no escribir resultados en disco")
    p.add_argument("--sin-graficas", action="store_true", help="guardar tablas y textos pero no PNG")
    p.add_argument("-q", "--silencioso", action="store_true", help="no mostrar el progreso por etapa")


def _prediccion(p):
    p.add_argument("--dias-prediccion", type=float)
    p.add_argument("--threshold", type=float)
    p.add_argument("--ventana-savgol", type=int)
    p.add_argument("--max-hueco-horas", type=float)
    p.add_argument("--sin-ajuste", action="store_true",
                   help="solo la identificación base de PySINDy (sin estrategia B)")


def construir_parser():
    p = argparse.ArgumentParser(prog="python cli.py", description=DESCRIPCION,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", metavar="comando")

    b = sub.add_parser("barrido", help="barrido de threshold (invernadero_profesor.py)")
    _comunes(b)
    b.add_argument("--thresholds", type=float, nargs="+")
    b.add_argument("--dias-entrenamiento", type=float)
    b.add_argument("--unidad-tiempo", choices=["s", "original"],
                   help="'s' (corregido, por defecto) u 'original' (expresión int64/1e9 del original)")
    b.add_argument("--seleccion", choices=["validacion", "separado"],
                   help="con qué datos se elige el threshold (por defecto 'validacion', como el original)")

    pr = sub.add_parser("prediccion", help="PySINDy + ajuste por simulación (invernadero_20dias.py)")
    _comunes(pr)
    _prediccion(pr)

    v = sub.add_parser("ventanas", help="predicción en varias ventanas (validar_modelo.py)")
    _comunes(v)
    v.add_argument("--paso-ventanas", type=int)

    e = sub.add_parser("estrategias", help="compara A, B y C sobre los mismos datos")
    _comunes(e)
    _prediccion(e)
    e.add_argument("--thresholds", type=float, nargs="+")

    sub.add_parser("gui", help="interfaz gráfica")

    c = sub.add_parser("config", help="imprime la configuración por defecto")
    c.add_argument("flujo", choices=list(C.CONFIGS))
    return p


def _aplicar(cfg, args, mapa: dict):
    for arg, campo in mapa.items():
        valor = getattr(args, arg, None)
        if valor is not None:
            setattr(cfg, campo, valor)
    return cfg


def main(argv=None):
    parser = construir_parser()
    args = parser.parse_args(argv)

    if args.comando is None:
        parser.print_help()
        return 0
    if args.comando == "config":
        print(json.dumps(C.a_dict(C.CONFIGS[args.flujo]()), indent=2, ensure_ascii=False))
        return 0
    if args.comando == "gui":
        from interfaz.aplicacion import main as gui
        gui()
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import pipeline

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    flujo = "prediccion" if args.comando == "estrategias" else args.comando
    cfg = C.cargar_config(flujo, args.config)

    extra = {}
    if args.comando == "barrido":
        _aplicar(cfg, args, {"thresholds": "thresholds", "dias_entrenamiento": "dias_entrenamiento",
                             "unidad_tiempo": "unidad_tiempo", "seleccion": "seleccion_threshold"})
    elif args.comando in ("prediccion", "estrategias"):
        _aplicar(cfg, args, {"dias_prediccion": "dias_prediccion", "threshold": "threshold",
                             "ventana_savgol": "ventana_savgol", "max_hueco_horas": "max_hueco_horas"})
        if args.sin_ajuste:
            cfg.ajustar_por_simulacion = False
        if args.comando == "estrategias" and args.thresholds:
            extra["thresholds"] = args.thresholds
    elif args.comando == "ventanas":
        _aplicar(cfg, args, {"paso_ventanas": "paso_ventanas_dias"})

    aviso = (lambda s: None) if args.silencioso else (lambda s: print("  ·", s, flush=True))
    ejec = pipeline.FUNCIONALIDADES[args.comando](
        ruta_csv=args.csv, cfg=cfg, guardar=not args.sin_guardar, carpeta=args.salida,
        aviso=aviso, graficas=not args.sin_graficas, **extra)

    print()
    print(ejec.resultado["resumen"])
    if ejec.carpeta:
        print(f"\nResultados por etapa en: {ejec.carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
