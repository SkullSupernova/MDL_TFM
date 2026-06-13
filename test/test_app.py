"""Tests de la lógica de selección de modelo en dos pasos de la web (src/app.py)."""

import numpy as np
import pandas as pd
import pytest

from src.app import (
    _CLASS_CONFIG_ORDER,
    _agrupar_modelos_por_arquitectura,
    _chart_comparacion,
    _chart_probabilidades,
    _descargar_y_verificar,
    _discover_models,
    _estilo_tabla_comparacion,
    _estilo_tabla_probabilidades,
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
def test_comparacion_modelos_incluye_todas_las_clases():
    labels_a = ["Cardiomegaly", "Edema", "Fracture"]
    probs_a = np.array([0.9, 0.1, 0.4])
    labels_b = ["Edema", "Cardiomegaly", "Pneumonia"]
    probs_b = np.array([0.2, 0.8, 0.7])
    df = _tabla_comparacion(labels_a, probs_a, labels_b, probs_b, 0.5)
    # Unión: primero las de A en su orden, luego las exclusivas de B (Pneumonia).
    assert list(df["Patología"]) == ["Cardiomegaly", "Edema", "Fracture", "Pneumonia"]
    card = df[df["Patología"] == "Cardiomegaly"].iloc[0]
    assert card["Modelo A"] == pytest.approx(0.9)
    assert card["Modelo B"] == pytest.approx(0.8)
    assert bool(card["Coinciden"])  # ambos >= 0.5
    # Clases presentes en un solo modelo: el otro queda NaN y 'Coinciden' es False.
    frac = df[df["Patología"] == "Fracture"].iloc[0]
    assert frac["Modelo A"] == pytest.approx(0.4)
    assert np.isnan(frac["Modelo B"])
    assert not bool(frac["Coinciden"])
    pneu = df[df["Patología"] == "Pneumonia"].iloc[0]
    assert np.isnan(pneu["Modelo A"])
    assert pneu["Modelo B"] == pytest.approx(0.7)


def test_comparacion_modelos_desacuerdo_en_umbral():
    df = _tabla_comparacion(["Edema"], np.array([0.9]), ["Edema"], np.array([0.1]), 0.5)
    fila = df.iloc[0]
    assert not bool(fila["Coinciden"])  # 0.9 detecta, 0.1 no
    assert fila["delta"] == pytest.approx(0.8)


def test_comparacion_modelos_sin_clases_comunes_incluye_ambas():
    df = _tabla_comparacion(["A"], np.array([0.5]), ["B"], np.array([0.5]), 0.5)
    assert list(df["Patología"]) == ["A", "B"]
    assert np.isnan(df.iloc[0]["Modelo B"])  # 'A' no existe en el modelo B
    assert np.isnan(df.iloc[1]["Modelo A"])  # 'B' no existe en el modelo A
    assert not df["Coinciden"].any()


def test_chart_probabilidades_devuelve_grafico_serializable():
    chart = _chart_probabilidades(["Edema", "Cardiomegaly"], np.array([0.9, 0.2]), 0.5)
    # Gráfico Altair componible: debe serializarse a dict (Vega-Lite) sin lanzar.
    assert chart.to_dict() is not None


def test_chart_comparacion_devuelve_grafico_serializable():
    df_cmp = _tabla_comparacion(
        ["Edema", "Cardiomegaly"], np.array([0.9, 0.2]),
        ["Edema", "Cardiomegaly"], np.array([0.4, 0.7]), 0.5,
    )
    chart = _chart_comparacion(df_cmp, "DenseNet-121 · full13", "ResNet-50 · min5pct9")
    assert chart.to_dict() is not None


def test_estilo_tabla_probabilidades_resalta_detectadas_en_verde():
    df = pd.DataFrame({
        "Patología": ["Edema", "Cardiomegaly"],
        "Probabilidad": [0.9, 0.2],
        "Detectada": [True, False],
    })
    html = _estilo_tabla_probabilidades(df).to_html()
    # La fila detectada lleva el verde de "detectada"; el render no debe lanzar.
    assert "#d4edda" in html
    assert "✓ Detectada" in html


def test_estilo_tabla_comparacion_marca_coincidencias():
    df_cmp = _tabla_comparacion(["Edema"], np.array([0.9]), ["Edema"], np.array([0.8]), 0.5)
    html = _estilo_tabla_comparacion(df_cmp, 0.5).to_html()
    assert "✓ Sí" in html  # ambos por encima del umbral


# --------------------------------------------------------------------------------------
# Modelos remotos (release assets)
# --------------------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]


def test_descargar_y_verificar_sha_correcto_escribe_fichero(tmp_path, monkeypatch):
    data = b"contenido del checkpoint"
    sha = __import__("hashlib").sha256(data).hexdigest()
    monkeypatch.setattr("src.app.requests.get", lambda *a, **k: _FakeResp(data))
    destino = tmp_path / "modelo.pth"
    _descargar_y_verificar(destino, "http://x/modelo.pth", sha)
    assert destino.read_bytes() == data
    assert not (tmp_path / "modelo.pth.part").exists()


def test_descargar_y_verificar_sha_incorrecto_lanza_y_no_deja_fichero(tmp_path, monkeypatch):
    monkeypatch.setattr("src.app.requests.get", lambda *a, **k: _FakeResp(b"datos"))
    destino = tmp_path / "modelo.pth"
    with pytest.raises(ValueError):
        _descargar_y_verificar(destino, "http://x/modelo.pth", "0" * 64)
    assert not destino.exists()
    assert not (tmp_path / "modelo.pth.part").exists()


def test_discover_models_incluye_los_modelos_remotos():
    labels = _discover_models()
    backbones = {v["backbone"] for v in labels.values()}
    assert {"convnext_tiny", "swin_t", "vgg16_bn"} <= backbones