import json
import os

import numpy as np

from src.experiment_tracker import ExperimentTracker
from src.evaluate import calcular_metricas_completas


def _cfg(tmp_path):
    return {
        "model": {"name": "densenet121", "num_classes": 3, "dropout": 0.5,
                  "hidden_units": 1024, "pretrained": False},
        "training": {"learning_rate": 1e-4, "weight_decay": 0.01, "batch_size": 4,
                     "seed": 42, "epochs": 1},
        "data": {"csv_path": "x", "images_root": "x", "batches": [],
                 "test_csv_path": "x", "train_split": 0.9},
        "experiments": {"root": str(tmp_path / "experiments"), "n_worst_cases": 5},
    }


def _arrays():
    labels = ["A", "B", "C"]   # C no tiene positivos (clase ausente / no evaluable)
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=float)
    prob = np.array([[0.9, 0.1, 0.2], [0.2, 0.8, 0.1], [0.7, 0.6, 0.3], [0.1, 0.2, 0.1]])
    pred = (prob > 0.5).astype(float)
    return labels, y, pred, prob


# =========================================================
# Creación de run y snapshot de configuración
# =========================================================

def test_tracker_crea_run_dir_y_snapshot_config(tmp_path):
    t = ExperimentTracker(_cfg(tmp_path), tag="prueba")
    assert t.dir.exists()
    assert (t.dir / "config.yaml").exists()
    assert (t.dir / "plots").exists()


# =========================================================
# dataset.json: distribución y clases ausentes
# =========================================================

def test_tracker_dataset_json_detecta_clase_ausente(tmp_path):
    t = ExperimentTracker(_cfg(tmp_path))
    labels, y, _, _ = _arrays()
    t.registrar_datasets(y, y, y, labels, {"total_final": 4}, {"A": 1.0}, {"seed": 42})
    d = json.load(open(t.dir / "dataset.json", encoding="utf-8"))
    assert d["distribucion"]["test"]["por_clase"]["C"]["positivos"] == 0
    assert "C" in d["distribucion"]["test"]["clases_ausentes"]
    # C no tiene ni positivos ni negativos variables -> no evaluable para AUROC
    assert "C" in d["clases_no_evaluables_auroc"]["test"]


# =========================================================
# Flujo completo: artefactos e informe
# =========================================================

def test_tracker_flujo_completo_genera_artefactos(tmp_path):
    cfg = _cfg(tmp_path)
    t = ExperimentTracker(cfg)
    labels, y, pred, prob = _arrays()

    t.registrar_datasets(y, y, y, labels, {}, {}, {})
    t.registrar_historial({
        "train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9],
        "val_acc": [0.1, 0.2], "val_f1": [0.3, 0.4], "val_auroc": [0.5, 0.6],
    })
    metrics = calcular_metricas_completas(y, pred, prob, labels)
    t.registrar_evaluacion("test", metrics, y, pred, prob, labels, ["r0", "r1", "r2", "r3"])

    t.finalizar(
        {"backbone": "densenet121", "num_classes": 3,
         "params": {"total": 1, "entrenables": 1}, "checkpoint_path": None},
        {"epochs_max": 1, "epochs_ejecutadas": 2, "mejor_epoca": 2,
         "mejor_epoca_val_auroc": 0.6, "learning_rate": 1e-4, "batch_size": 4, "seed": 42},
        {"promovido": True},
    )

    for fichero in ["manifest.json", "report.md", "metrics_test.json",
                    "metrics_test_per_class.csv", "history.csv"]:
        assert (t.dir / fichero).exists(), fichero
    assert (t.dir / "predictions" / "test_predictions.npz").exists()
    assert (t.dir / "error_analysis" / "test_worst_cases.csv").exists()
    assert os.path.exists(os.path.join(cfg["experiments"]["root"], "leaderboard.csv"))


def test_tracker_report_menciona_clase_no_evaluable(tmp_path):
    cfg = _cfg(tmp_path)
    t = ExperimentTracker(cfg)
    labels, y, pred, prob = _arrays()
    t.registrar_datasets(y, y, y, labels, {}, {}, {})
    metrics = calcular_metricas_completas(y, pred, prob, labels)
    t.registrar_evaluacion("test", metrics, y, pred, prob, labels, ["r0", "r1", "r2", "r3"])
    t.finalizar(
        {"backbone": "densenet121", "num_classes": 3, "params": {"total": 1, "entrenables": 1},
         "checkpoint_path": None},
        {"epochs_max": 1, "epochs_ejecutadas": 1, "mejor_epoca": 1, "mejor_epoca_val_auroc": 0.6,
         "learning_rate": 1e-4, "batch_size": 4, "seed": 42},
        {"promovido": False},
    )
    report = (t.dir / "report.md").read_text(encoding="utf-8")
    # La clase C (sin positivos) debe aparecer marcada como no evaluable en el informe.
    assert "no evaluables" in report.lower()
    assert "C" in report
