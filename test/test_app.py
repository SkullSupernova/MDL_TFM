"""Tests de la lógica de selección de modelo en dos pasos de la web (src/app.py)."""

import numpy as np
import pytest

from src.app import (
    _CLASS_CONFIG_ORDER,
    _agrupar_modelos_por_arquitectura,
    _chart_probabilidades,
    _ordenar_class_configs,
    _tabla_comparacion,
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


# --------------------------------------------------------------------------------------
# Comparación de dos modelos (F8)
# --------------------------------------------------------------------------------------
def test_comparacion_modelos_solo_patologias_comunes():
    labels_a = ["Cardiomegaly", "Edema", "Fracture"]
    probs_a = np.array([0.9, 0.1, 0.4])
    labels_b = ["Edema", "Cardiomegaly", "Pneumonia"]
    probs_b = np.array([0.2, 0.8, 0.7])
    df = _tabla_comparacion(labels_a, probs_a, labels_b, probs_b, 0.5)
    # Comunes en orden de labels_a: Cardiomegaly, Edema (Fracture/Pneumonia se descartan).
    assert list(df["Patología"]) == ["Cardiomegaly", "Edema"]
    card = df[df["Patología"] == "Cardiomegaly"].iloc[0]
    assert card["Modelo A"] == pytest.approx(0.9)
    assert card["Modelo B"] == pytest.approx(0.8)
    assert bool(card["Coinciden"])  # ambos >= 0.5


def test_comparacion_modelos_desacuerdo_en_umbral():
    df = _tabla_comparacion(["Edema"], np.array([0.9]), ["Edema"], np.array([0.1]), 0.5)
    fila = df.iloc[0]
    assert not bool(fila["Coinciden"])  # 0.9 detecta, 0.1 no
    assert fila["delta"] == pytest.approx(0.8)


def test_comparacion_modelos_sin_clases_comunes_devuelve_vacio():
    df = _tabla_comparacion(["A"], np.array([0.5]), ["B"], np.array([0.5]), 0.5)
    assert df.empty
    assert list(df.columns) == ["Patología", "Modelo A", "Modelo B", "delta", "Coinciden"]


def test_chart_probabilidades_devuelve_grafico_serializable():
    chart = _chart_probabilidades(["Edema", "Cardiomegaly"], np.array([0.9, 0.2]), 0.5)
    # Gráfico Altair componible: debe serializarse a dict (Vega-Lite) sin lanzar.
    assert chart.to_dict() is not None