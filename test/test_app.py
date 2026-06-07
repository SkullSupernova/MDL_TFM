"""Tests de la lógica de selección de modelo en dos pasos de la web (src/app.py)."""

import pytest

from src.app import (
    _CLASS_CONFIG_ORDER,
    _agrupar_modelos_por_arquitectura,
    _ordenar_class_configs,
)


def _modelos_ejemplo():
    """Modelos descubiertos sintéticos: {label: {path, backbone, class_config}}."""
    return {
        "densenet121 · full13": {"path": "a.pth", "backbone": "densenet121", "class_config": "full13"},
        "densenet121 · nofracture12": {"path": "b.pth", "backbone": "densenet121", "class_config": "nofracture12"},
        "densenet121 · min5pct9": {"path": "c.pth", "backbone": "densenet121", "class_config": "min5pct9"},
        "resnet50 · nofracture12": {"path": "d.pth", "backbone": "resnet50", "class_config": "nofracture12"},
        "resnet50 · min5pct9": {"path": "e.pth", "backbone": "resnet50", "class_config": "min5pct9"},
    }


# --------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------
def test_agrupar_modelos_devuelve_estructura_por_backbone():
    por_arq = _agrupar_modelos_por_arquitectura(_modelos_ejemplo())
    assert set(por_arq) == {"densenet121", "resnet50"}
    assert set(por_arq["densenet121"]) == {"full13", "nofracture12", "min5pct9"}
    assert set(por_arq["resnet50"]) == {"nofracture12", "min5pct9"}


def test_agrupar_modelos_conserva_info_del_checkpoint():
    por_arq = _agrupar_modelos_por_arquitectura(_modelos_ejemplo())
    assert por_arq["resnet50"]["min5pct9"]["path"] == "e.pth"


def test_ordenar_configs_respeta_orden_canonico():
    # Entrada desordenada -> salida en orden full13, nofracture12, min5pct9.
    assert _ordenar_class_configs(["min5pct9", "full13", "nofracture12"]) == _CLASS_CONFIG_ORDER


# --------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------
def test_ordenar_configs_subconjunto_mantiene_orden():
    assert _ordenar_class_configs(["min5pct9", "nofracture12"]) == ["nofracture12", "min5pct9"]


def test_ordenar_configs_desconocida_va_al_final():
    resultado = _ordenar_class_configs(["otra", "full13"])
    assert resultado == ["full13", "otra"]


def test_ordenar_configs_none_va_al_final():
    # Checkpoint en formato antiguo (class_config None) no debe romper el orden.
    resultado = _ordenar_class_configs(["nofracture12", None])
    assert resultado[0] == "nofracture12"
    assert resultado[-1] is None


def test_agrupar_modelos_vacio_devuelve_dict_vacio():
    assert _agrupar_modelos_por_arquitectura({}) == {}


def test_agrupar_modelos_formato_antiguo_class_config_none():
    modelos = {"densenet121": {"path": "x.pth", "backbone": "densenet121", "class_config": None}}
    por_arq = _agrupar_modelos_por_arquitectura(modelos)
    assert por_arq["densenet121"][None]["path"] == "x.pth"