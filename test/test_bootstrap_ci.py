"""Tests del módulo de IC bootstrap del leaderboard (src/bootstrap_ci.py)."""

import math

import numpy as np
import pytest
import yaml

from src.bootstrap_ci import (
    _indices_competicion,
    _mean_auroc,
    _mean_prauc,
    agregar_leaderboard,
    bootstrap_ci,
    cargar_run,
)
from src.models import CHEXPERT_COMPETITION_5, get_active_pathology_cols


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture
def datos_separables():
    """y_prob = y_true: separación perfecta (AUROC 1.0) en 2 columnas con ambos valores."""
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    y_prob = y_true.copy()
    return y_true, y_prob


def _crear_run(base, nombre, class_config, n=40, seed=0):
    """Crea un run sintético con predicciones npz y config.yaml; devuelve su ruta."""
    run_dir = base / nombre
    (run_dir / "predictions").mkdir(parents=True)
    labels = get_active_pathology_cols(class_config)
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, len(labels))).astype(float)
    # Garantizar ambos valores en cada columna para que todas sean evaluables.
    y_true[0, :] = 1.0
    y_true[1, :] = 0.0
    y_prob = rng.random(size=(n, len(labels)))
    np.savez_compressed(
        run_dir / "predictions" / "test_predictions.npz",
        y_true=y_true, y_prob=y_prob, rutas=np.array([], dtype=object),
    )
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"data": {"class_config": class_config}}, f)
    return run_dir


# --------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------
def test_separacion_perfecta_mean_auroc_devuelve_uno(datos_separables):
    y_true, y_prob = datos_separables
    assert _mean_auroc(y_true, y_prob, [0, 1]) == pytest.approx(1.0)


def test_separacion_perfecta_bootstrap_devuelve_punto_e_intervalo(datos_separables):
    y_true, y_prob = datos_separables
    rng = np.random.default_rng(42)
    punto, lo, hi = bootstrap_ci(y_true, y_prob, [0, 1], _mean_auroc, 500, rng)
    assert punto == pytest.approx(1.0)
    assert lo <= punto <= hi
    assert hi == pytest.approx(1.0)


def test_dos_clases_indices_competicion_estan_presentes():
    labels = get_active_pathology_cols("nofracture12")
    idx = _indices_competicion(labels)
    assert len(idx) == len(CHEXPERT_COMPETITION_5)
    assert all(labels[i] in CHEXPERT_COMPETITION_5 for i in idx)


# --------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------
def test_columna_sin_ambos_valores_se_omite_de_auroc():
    # Columna 0 evaluable; columna 1 todo unos (no evaluable) -> media = solo la 0.
    y_true = np.array([[1, 1], [0, 1], [1, 1], [0, 1]], dtype=float)
    y_prob = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5], [0.2, 0.5]])
    assert _mean_auroc(y_true, y_prob, [0, 1]) == pytest.approx(1.0)


def test_ninguna_columna_evaluable_devuelve_nan():
    y_true = np.ones((4, 2))  # sin negativos en ninguna columna
    y_prob = np.random.default_rng(0).random((4, 2))
    assert math.isnan(_mean_auroc(y_true, y_prob, [0, 1]))


def test_sin_positivos_pr_auc_devuelve_nan():
    y_true = np.zeros((4, 1))  # sin positivos
    y_prob = np.random.default_rng(0).random((4, 1))
    assert math.isnan(_mean_prauc(y_true, y_prob, [0]))


def test_misma_semilla_produce_intervalo_identico(datos_separables):
    y_true, y_prob = datos_separables
    a = bootstrap_ci(y_true, y_prob, [0, 1], _mean_auroc, 300, np.random.default_rng(7))
    b = bootstrap_ci(y_true, y_prob, [0, 1], _mean_auroc, 300, np.random.default_rng(7))
    assert a == b


# --------------------------------------------------------------------------------------
# Gestión de errores / robustez
# --------------------------------------------------------------------------------------
def test_todas_las_replicas_no_evaluables_devuelve_nan_en_intervalo():
    # Una sola fila: cualquier remuestreo tiene una única clase -> no evaluable.
    y_true = np.array([[1, 0]], dtype=float)
    y_prob = np.array([[0.7, 0.3]])
    punto, lo, hi = bootstrap_ci(y_true, y_prob, [0, 1], _mean_auroc, 100, np.random.default_rng(0))
    assert math.isnan(lo) and math.isnan(hi)


def test_npz_inexistente_lanza_error(tmp_path):
    (tmp_path / "predictions").mkdir(parents=True)
    with open(tmp_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"data": {"class_config": "nofracture12"}}, f)
    with pytest.raises(FileNotFoundError):
        cargar_run(tmp_path)


# --------------------------------------------------------------------------------------
# Integración con datos sintéticos en disco
# --------------------------------------------------------------------------------------
def test_cargar_run_devuelve_labels_de_la_config(tmp_path):
    run_dir = _crear_run(tmp_path, "20260101-000000_densenet121_nofracture12", "nofracture12")
    y_true, y_prob, labels, class_config = cargar_run(run_dir)
    assert class_config == "nofracture12"
    assert labels == get_active_pathology_cols("nofracture12")
    assert y_prob.shape[1] == len(labels)


def test_agregar_leaderboard_escribe_csv_y_excluye_calibracion(tmp_path):
    _crear_run(tmp_path, "20260101-000000_densenet121_nofracture12", "nofracture12")
    _crear_run(tmp_path, "20260101-010000_resnet50_min5pct9", "min5pct9")
    _crear_run(tmp_path, "20260101-020000_densenet121_nofracture12_calibracion", "nofracture12")

    filas = agregar_leaderboard(tmp_path, n_boot=100, seed=42, alpha=0.05)

    assert len(filas) == 2  # el run 'calibracion' se excluye
    assert (tmp_path / "leaderboard_ci.csv").exists()
    for fila in filas:
        assert fila["auroc_chexpert5_lo"] <= fila["auroc_chexpert5"] <= fila["auroc_chexpert5_hi"]
        assert fila["n_boot"] == 100


def test_sin_runs_devuelve_lista_vacia(tmp_path):
    assert agregar_leaderboard(tmp_path, n_boot=10, seed=1, alpha=0.05) == []