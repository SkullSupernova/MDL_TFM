import zipfile
from io import BytesIO

import numpy as np
from PIL import Image

from src.image_utils import validar_imagen_radiografia, empaquetar_imagenes_zip


# =========================================================
# validar_imagen_radiografia — happy path
# =========================================================

def test_validar_radiografia_gris_valida_es_ok():
    img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8))  # modo L
    res = validar_imagen_radiografia(img, n_bytes=500_000)
    assert res["ok"] is True
    assert res["errores"] == []
    assert res["avisos"] == []


def test_validar_devuelve_estructura_esperada():
    img = Image.new("L", (128, 128))
    res = validar_imagen_radiografia(img)
    assert set(res.keys()) == {"ok", "errores", "avisos"}


# =========================================================
# validar_imagen_radiografia — errores y avisos
# =========================================================

def test_validar_resolucion_baja_es_error():
    img = Image.new("L", (32, 32))
    res = validar_imagen_radiografia(img, lado_min=64)
    assert res["ok"] is False
    assert any("baja" in e.lower() or "mínimo" in e.lower() for e in res["errores"])


def test_validar_tamano_excedido_es_error():
    img = Image.new("L", (224, 224))
    res = validar_imagen_radiografia(img, n_bytes=20 * 1024 * 1024, max_mb=10)
    assert res["ok"] is False
    assert any("MB" in e for e in res["errores"])


def test_validar_imagen_en_color_genera_aviso_no_bloqueante():
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[..., 0] = 220   # canal rojo dominante → claramente en color
    res = validar_imagen_radiografia(Image.fromarray(arr, mode="RGB"))
    assert res["ok"] is True          # es aviso, no error
    assert any("color" in a.lower() for a in res["avisos"])


# =========================================================
# empaquetar_imagenes_zip
# =========================================================

def _panels(n):
    return [
        {"label": f"Patología {i}", "heatmap": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)}
        for i in range(n)
    ]


def test_empaquetar_zip_contiene_original_y_paneles():
    original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    data = empaquetar_imagenes_zip(original, _panels(5))
    with zipfile.ZipFile(BytesIO(data)) as zf:
        nombres = zf.namelist()
    assert "original.png" in nombres
    assert len(nombres) == 6          # 1 original + 5 heatmaps
    assert sum(n.startswith("heatmap_") for n in nombres) == 5


def test_empaquetar_zip_sin_paneles_solo_original():
    original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    with zipfile.ZipFile(BytesIO(empaquetar_imagenes_zip(original, []))) as zf:
        assert zf.namelist() == ["original.png"]


def test_empaquetar_zip_nombre_de_fichero_saneado():
    original = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    panels = [{"label": "Pleural Effusion/Other", "heatmap": original.copy()}]
    with zipfile.ZipFile(BytesIO(empaquetar_imagenes_zip(original, panels))) as zf:
        heatmaps = [n for n in zf.namelist() if n.startswith("heatmap_")]
    assert heatmaps == ["heatmap_01_Pleural_Effusion_Other.png"]


def test_empaquetar_zip_comparacion_incluye_ambos_modelos():
    original = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    panels = [{
        "label": "Cardiomegaly",
        "heatmap": original.copy(),
        "heatmap_b": original.copy(),
    }]
    with zipfile.ZipFile(BytesIO(empaquetar_imagenes_zip(original, panels))) as zf:
        heatmaps = sorted(n for n in zf.namelist() if n.startswith("heatmap_"))
    assert heatmaps == ["heatmap_01_Cardiomegaly_A.png", "heatmap_01_Cardiomegaly_B.png"]


def test_empaquetar_zip_comparacion_panel_sin_modelo_b_exporta_solo_a():
    original = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    panels = [
        {"label": "Edema", "heatmap": original.copy(), "heatmap_b": original.copy()},
        {"label": "Fracture", "heatmap": original.copy(), "heatmap_b": None},
    ]
    with zipfile.ZipFile(BytesIO(empaquetar_imagenes_zip(original, panels))) as zf:
        heatmaps = sorted(n for n in zf.namelist() if n.startswith("heatmap_"))
    assert heatmaps == [
        "heatmap_01_Edema_A.png",
        "heatmap_01_Edema_B.png",
        "heatmap_02_Fracture.png",
    ]
