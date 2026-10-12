"""Punto de entrada, configuración y registro de etapas."""

import json

import pytest

import cli
import config as C


def test_help_lista_las_funcionalidades(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    salida = capsys.readouterr().out
    for comando in ("barrido", "prediccion", "ventanas", "estrategias", "gui"):
        assert comando in salida


def test_config_por_defecto_en_json(capsys):
    cli.main(["config", "barrido"])
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["thresholds"] == C.ConfigBarrido().thresholds
    assert cfg["unidad_tiempo"] == "s"


def test_config_desde_json(tmp_path):
    ruta = tmp_path / "c.json"
    ruta.write_text(json.dumps({"dias_prediccion": 10, "threshold": 0.01}))
    cfg = C.cargar_config("prediccion", ruta)
    assert cfg.dias_prediccion == 10 and cfg.threshold == 0.01 and cfg.max_hueco_horas == 2.0
    ruta.write_text(json.dumps({"no_existe": 1}))
    with pytest.raises(ValueError, match="no_existe"):
        C.cargar_config("prediccion", ruta)


@pytest.mark.lenta
def test_prediccion_registra_las_8_etapas_en_disco(tmp_path):
    import pipeline
    ej = pipeline.ejecutar_prediccion(cfg=C.ConfigPrediccion(), carpeta=tmp_path, aviso=None, graficas=False)
    assert [e.numero for e in ej.etapas] == list(range(1, 9))
    for e in ej.etapas:
        assert e.texto, e.titulo
        assert (ej.carpeta / e.clave / "resumen.json").exists()
    assert (ej.carpeta / "configuracion.json").exists()
    registro = (ej.carpeta / "registro.log").read_text(encoding="utf-8")
    assert "->" in registro                   # cantidades antes -> después
    assert ej.resultado["metricas"]["R2"] > 0.9


@pytest.mark.lenta
def test_barrido_con_seleccion_separada_es_independiente(tmp_path):
    import pipeline
    cfg = C.ConfigBarrido(seleccion_threshold="separado")
    ej = pipeline.ejecutar_barrido(cfg=cfg, guardar=False, aviso=None)
    assert ej.resultado["seleccion_independiente"] is True
    assert ej.etapa(7).datos["datos_seleccion"]["conjunto"].startswith("seleccion")
