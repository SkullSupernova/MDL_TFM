from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocess_resize import resize_tree


def _crear_arbol(root: Path, n: int = 3, tam: int = 320) -> None:
    """Crea n imágenes .jpg en una estructura anidada patientX/studyY/view.jpg."""
    for i in range(n):
        p = root / f"patient{i:05d}" / "study1" / "view1_frontal.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = np.random.randint(0, 255, (tam, tam), dtype=np.uint8)
        Image.fromarray(arr).save(str(p))


# =========================================================
# resize_tree — happy path
# =========================================================

def test_resize_tree_redimensiona_y_preserva_estructura(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _crear_arbol(src, n=3, tam=320)

    rep = resize_tree(str(src), str(dst), size=224)

    assert rep["procesadas"] == 3
    salidas = list(dst.rglob("*.jpg"))
    assert len(salidas) == 3
    with Image.open(salidas[0]) as img:
        assert img.size == (224, 224)
    # La ruta relativa se conserva.
    assert (dst / "patient00000" / "study1" / "view1_frontal.jpg").exists()


def test_resize_tree_es_idempotente(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _crear_arbol(src, n=2)

    primera = resize_tree(str(src), str(dst), size=224)
    segunda = resize_tree(str(src), str(dst), size=224)

    assert primera["procesadas"] == 2
    assert segunda["procesadas"] == 0
    assert segunda["omitidas"] == 2


# =========================================================
# resize_tree — edge cases
# =========================================================

def test_resize_tree_respeta_limite(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _crear_arbol(src, n=5)

    rep = resize_tree(str(src), str(dst), size=224, limit=2)

    assert rep["procesadas"] == 2
    assert len(list(dst.rglob("*.jpg"))) == 2


def test_resize_tree_arbol_vacio_no_genera_salidas(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()

    rep = resize_tree(str(src), str(dst), size=224)

    assert rep["total"] == 0


# =========================================================
# resize_tree — errores
# =========================================================

def test_resize_tree_origen_inexistente_lanza_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        resize_tree(str(tmp_path / "no_existe"), str(tmp_path / "dst"), size=224)
