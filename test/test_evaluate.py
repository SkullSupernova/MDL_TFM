from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate_model
from src.utils import construir_df_test_valid, calculate_metrics, auroc_macro, auc_por_clase
from src.models import CHEXPERT_PATHOLOGY_COLS


# =========================================================
# Helpers
# =========================================================

class _ScaleModel(nn.Module):
    """Modelo determinista: mapea entradas {0,1} a logits {-50,+50}.

    Permite forzar una predicción perfecta cuando la entrada coincide con las
    etiquetas, sin necesidad de entrenar.
    """
    def forward(self, x):
        return (x - 0.5) * 100.0


def _crear_valid_sintetico(tmp_path):
    root = tmp_path / "valid_root"
    specs = [
        ("patient00001", "Frontal", "AP", True),    # válido: se conserva
        ("patient00002", "Frontal", "PA", True),    # descartado: no es AP
        ("patient00003", "Lateral", "AP", True),    # descartado: no es Frontal
        ("patient00004", "Frontal", "AP", False),   # descartado: imagen inexistente
    ]
    rows = []
    for pid, vista, pos, crear in specs:
        rel = f"valid/{pid}/study1/view1_frontal.jpg"
        if crear:
            img_path = root / Path(rel)
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(b"fake-jpg")
        row = {"Path": f"CheXpert-v1.0/{rel}", "Frontal/Lateral": vista, "AP/PA": pos}
        for c in CHEXPERT_PATHOLOGY_COLS:
            row[c] = np.nan   # para verificar la imputación a 0.0
        rows.append(row)
    csv_path = tmp_path / "valid.csv"
    pd.DataFrame(rows).to_csv(str(csv_path), index=False)
    return str(csv_path), str(root)


# =========================================================
# evaluate_model — happy path
# =========================================================

def test_evaluate_model_devuelve_shapes_correctas():
    model = nn.Sequential(nn.Linear(4, 3))
    X = torch.randn(6, 4)
    Y = torch.randint(0, 2, (6, 3)).float()
    loader = DataLoader(TensorDataset(X, Y), batch_size=2)

    y_true, y_pred, y_prob = evaluate_model(model, loader, torch.device("cpu"))
    assert y_true.shape == y_pred.shape == y_prob.shape == (6, 3)


def test_evaluate_model_predicciones_binarias_y_probabilidades_en_rango():
    model = nn.Sequential(nn.Linear(4, 3))
    X = torch.randn(8, 4)
    Y = torch.randint(0, 2, (8, 3)).float()
    loader = DataLoader(TensorDataset(X, Y), batch_size=4)

    _, y_pred, y_prob = evaluate_model(model, loader, torch.device("cpu"))
    assert set(np.unique(y_pred)).issubset({0.0, 1.0})
    assert y_prob.min() >= 0.0 and y_prob.max() <= 1.0


def test_evaluate_model_prediccion_perfecta_devuelve_f1_uno():
    labels = torch.tensor(
        [[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=torch.float32
    )
    loader = DataLoader(TensorDataset(labels.clone(), labels), batch_size=2)

    y_true, y_pred, _ = evaluate_model(_ScaleModel(), loader, torch.device("cpu"))
    np.testing.assert_array_equal(y_true, y_pred)
    assert calculate_metrics(y_true, y_pred)["f1_macro"] == 1.0


# =========================================================
# construir_df_test_valid — filtrado y edge cases
# =========================================================

def test_construir_df_test_valid_solo_conserva_frontal_ap(tmp_path):
    csv_path, root = _crear_valid_sintetico(tmp_path)
    df = construir_df_test_valid(csv_path, root, CHEXPERT_PATHOLOGY_COLS)
    assert len(df) == 1
    assert "patient00001" in df.loc[0, "Ruta_Absoluta"]


def test_construir_df_test_valid_imputa_nan_a_cero(tmp_path):
    csv_path, root = _crear_valid_sintetico(tmp_path)
    df = construir_df_test_valid(csv_path, root, CHEXPERT_PATHOLOGY_COLS)
    assert (df[CHEXPERT_PATHOLOGY_COLS].iloc[0] == 0.0).all()


def test_construir_df_test_valid_descarta_imagen_inexistente(tmp_path):
    csv_path, root = _crear_valid_sintetico(tmp_path)
    df = construir_df_test_valid(csv_path, root, CHEXPERT_PATHOLOGY_COLS)
    assert not df["Ruta_Absoluta"].str.contains("patient00004").any()


def test_construir_df_test_valid_columna_ruta_absoluta_presente(tmp_path):
    csv_path, root = _crear_valid_sintetico(tmp_path)
    df = construir_df_test_valid(csv_path, root, CHEXPERT_PATHOLOGY_COLS)
    assert "Ruta_Absoluta" in df.columns


# =========================================================
# auroc_macro / auc_por_clase
# =========================================================

def test_auroc_macro_separacion_perfecta_devuelve_uno():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    media, n = auroc_macro(y_true, y_prob)
    assert media == 1.0
    assert n == 2


def test_auroc_macro_omite_clase_sin_positivos():
    # La columna 1 no tiene ningún positivo: no es evaluable y se omite del promedio.
    y_true = np.array([[1, 0], [0, 0], [1, 0]], dtype=float)
    y_prob = np.array([[0.9, 0.3], [0.2, 0.5], [0.7, 0.1]])
    _, n = auroc_macro(y_true, y_prob)
    assert n == 1


def test_auc_por_clase_none_para_clase_degenerada():
    y_true = np.array([[1, 0], [0, 0], [1, 0]], dtype=float)
    y_prob = np.array([[0.9, 0.3], [0.2, 0.5], [0.7, 0.1]])
    aucs = auc_por_clase(y_true, y_prob, ["A", "B"])
    assert aucs["B"] is None
    assert aucs["A"] is not None
