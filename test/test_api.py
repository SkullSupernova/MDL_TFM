"""
Tests de la API de inferencia (src/api.py).

Se prueban las funciones de ruta directamente (sin TestClient ni httpx, que no están en el
entorno del CI), fijando el estado global que en producción establece el `lifespan`:
un modelo simulado, la configuración y las etiquetas. El modelo falso devuelve logits 0
→ sigmoid 0.5 en todas las clases, lo que hace deterministas las comprobaciones de umbral.
"""

import asyncio
import io
import json

import pytest
import torch
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import Headers, UploadFile

from src import api


class _FakeModel(torch.nn.Module):
    """Modelo de prueba: devuelve logits 0 (sigmoid 0.5) para 13 clases."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 13))


@pytest.fixture(autouse=True)
def _estado_api(monkeypatch):
    """Simula el estado tras el arranque (lifespan): modelo, etiquetas y config listos."""
    monkeypatch.setattr(api, "_model", _FakeModel())
    monkeypatch.setattr(api, "_labels", api.get_pathology_labels(13))
    monkeypatch.setattr(api, "_cfg", {"training": {"threshold": 0.5}, "model": {"name": "densenet121"}})
    monkeypatch.setattr(api, "_device", torch.device("cpu"))


def _upload(data: bytes, content_type: str = "image/jpeg", filename: str = "rx.jpg") -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=filename,
                      headers=Headers({"content-type": content_type}))


def _img_bytes(fmt: str = "JPEG", color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buf, format=fmt)
    return buf.getvalue()


def _predict(upload: UploadFile, threshold=None) -> dict:
    resp = asyncio.run(api.predict(file=upload, threshold=threshold))
    return json.loads(resp.body)


# --------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------
def test_health_devuelve_estado_ok():
    assert api.health() == {"status": "ok"}


def test_predict_imagen_valida_devuelve_probabilidades_de_13_clases():
    body = _predict(_upload(_img_bytes()))
    assert set(body) == {"threshold", "probabilities", "detected_pathologies"}
    assert len(body["probabilities"]) == 13
    assert body["threshold"] == 0.5  # umbral por defecto de config


def test_predict_png_tambien_valido():
    body = _predict(_upload(_img_bytes("PNG"), content_type="image/png", filename="rx.png"))
    assert len(body["probabilities"]) == 13


# --------------------------------------------------------------------------------------
# Edge cases (umbral)
# --------------------------------------------------------------------------------------
def test_predict_umbral_alto_no_detecta_ninguna():
    # sigmoid(0)=0.5: con umbral 0.6 ninguna patología lo supera.
    body = _predict(_upload(_img_bytes()), threshold=0.6)
    assert body["detected_pathologies"] == []
    assert body["threshold"] == 0.6


def test_predict_umbral_bajo_detecta_todas():
    body = _predict(_upload(_img_bytes()), threshold=0.4)
    assert len(body["detected_pathologies"]) == 13


# --------------------------------------------------------------------------------------
# Gestión de errores
# --------------------------------------------------------------------------------------
def test_predict_tipo_no_soportado_lanza_422():
    with pytest.raises(HTTPException) as exc:
        _predict(_upload(b"texto plano", content_type="text/plain", filename="x.txt"))
    assert exc.value.status_code == 422


def test_predict_imagen_corrupta_lanza_422():
    with pytest.raises(HTTPException) as exc:
        _predict(_upload(b"esto no es una imagen", content_type="image/jpeg"))
    assert exc.value.status_code == 422


def test_predict_sin_modelo_cargado_lanza_503(monkeypatch):
    monkeypatch.setattr(api, "_model", None)
    with pytest.raises(HTTPException) as exc:
        _predict(_upload(_img_bytes()))
    assert exc.value.status_code == 503
