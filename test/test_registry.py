from src.model_registry import (
    es_mejor,
    cargar_registro,
    guardar_registro,
    registrar_experimento,
)


# =========================================================
# es_mejor — happy path
# =========================================================

def test_es_mejor_sin_campeon_siempre_promueve():
    assert es_mejor({"auroc_chexpert5": 0.5, "f1_macro": 0.3}, None) is True


def test_es_mejor_auroc_superior_promueve():
    nuevas = {"auroc_chexpert5": 0.82, "f1_macro": 0.5}
    campeon = {"auroc_chexpert5": 0.80, "f1_macro": 0.6}
    assert es_mejor(nuevas, campeon, min_delta=0.005) is True


# =========================================================
# es_mejor — no mejora / edge cases
# =========================================================

def test_es_mejor_auroc_inferior_no_promueve():
    nuevas = {"auroc_chexpert5": 0.78, "f1_macro": 0.9}
    campeon = {"auroc_chexpert5": 0.80, "f1_macro": 0.5}
    assert es_mejor(nuevas, campeon, min_delta=0.005) is False


def test_es_mejor_empate_tecnico_desempata_por_f1():
    # Diferencia de AUROC (0.001) dentro de min_delta (0.005): decide el F1-macro.
    nuevas = {"auroc_chexpert5": 0.801, "f1_macro": 0.70}
    campeon = {"auroc_chexpert5": 0.800, "f1_macro": 0.60}
    assert es_mejor(nuevas, campeon, min_delta=0.005) is True


def test_es_mejor_empate_tecnico_f1_inferior_no_promueve():
    nuevas = {"auroc_chexpert5": 0.801, "f1_macro": 0.50}
    campeon = {"auroc_chexpert5": 0.800, "f1_macro": 0.60}
    assert es_mejor(nuevas, campeon, min_delta=0.005) is False


def test_es_mejor_auroc_none_usa_f1():
    nuevas = {"auroc_chexpert5": None, "f1_macro": 0.70}
    campeon = {"auroc_chexpert5": 0.80, "f1_macro": 0.60}
    assert es_mejor(nuevas, campeon) is True


# =========================================================
# Registro e historial — persistencia
# =========================================================

def test_guardar_y_cargar_registro(tmp_path):
    ruta = tmp_path / "reg.json"
    guardar_registro("densenet121", {"auroc_chexpert5": 0.8}, ruta=ruta)
    assert cargar_registro("densenet121", ruta=ruta)["auroc_chexpert5"] == 0.8
    # Un backbone no registrado devuelve None.
    assert cargar_registro("resnet50", ruta=ruta) is None


def test_cargar_registro_inexistente_devuelve_none(tmp_path):
    assert cargar_registro("densenet121", ruta=tmp_path / "noexiste.json") is None


def test_registrar_experimento_es_append(tmp_path):
    ruta = tmp_path / "exp.jsonl"
    registrar_experimento({"a": 1}, ruta=ruta)
    registrar_experimento({"a": 2}, ruta=ruta)
    lineas = ruta.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 2
