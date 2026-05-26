import numpy as np


def test_imports():
    from src.utils import setup_environment, set_seed, calculate_metrics
    from src.models import CheXpertDataset, CHEXPERT_PATHOLOGY_COLS


def test_calculate_metrics_perfect():
    from src.utils import calculate_metrics
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1]])
    m = calculate_metrics(y_true, y_pred)
    assert m['accuracy'] == 1.0
    assert m['f1_macro'] == 1.0


def test_pathology_count():
    from src.models import CHEXPERT_PATHOLOGY_COLS
    assert len(CHEXPERT_PATHOLOGY_COLS) == 13, (
        f"Se esperaban 13 patologías activas, se encontraron {len(CHEXPERT_PATHOLOGY_COLS)}. "
        "Pendiente corrección en Phase 2 (eliminar 'Pleural Other' de CHEXPERT_PATHOLOGY_COLS)."
    )
