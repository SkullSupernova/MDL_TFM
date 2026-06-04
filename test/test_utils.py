import numpy as np
import pandas as pd
import pytest
import torch.nn as nn

from src.utils import (
    calculate_metrics,
    EarlyStopping,
    ModelCheckpoint,
    aplicar_seleccion_clases,
)
from src.models import CHEXPERT_PATHOLOGY_COLS, get_active_pathology_cols




# =========================================================
# calculate_metrics — happy path
# =========================================================

def test_metricas_prediccion_perfecta_devuelve_uno():
    y = np.array([[1, 0, 1], [0, 1, 0]])
    m = calculate_metrics(y, y)
    assert m['accuracy'] == 1.0
    assert m['f1_macro'] == 1.0


def test_metricas_prediccion_parcial_devuelve_valor_intermedio():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_pred = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    m = calculate_metrics(y_true, y_pred)
    assert 0.0 < m['f1_macro'] < 1.0


# =========================================================
# calculate_metrics — edge cases
# =========================================================

def test_metricas_prediccion_todo_negativo_correcto():
    y_true = np.array([[0, 0], [0, 0]])
    y_pred = np.array([[0, 0], [0, 0]])
    m = calculate_metrics(y_true, y_pred)
    assert m['accuracy'] == 1.0


def test_metricas_una_clase_ausente_no_lanza():
    # Una clase sin ningún positivo → zero_division=0 debe manejarlo
    y_true = np.array([[1, 0], [1, 0]])
    y_pred = np.array([[1, 0], [0, 0]])
    m = calculate_metrics(y_true, y_pred)
    assert 'accuracy' in m and 'f1_macro' in m


def test_metricas_devuelve_claves_esperadas():
    y = np.array([[1, 0], [0, 1]])
    m = calculate_metrics(y, y)
    assert set(m.keys()) == {'accuracy', 'f1_macro'}


# =========================================================
# EarlyStopping — happy path
# =========================================================

def test_early_stopping_activa_tras_patience_epocas():
    es = EarlyStopping(patience=2)
    es(1.0)   # best_loss=1.0
    es(1.1)   # peor → counter=1
    es(1.1)   # peor → counter=2 ≥ patience → early_stop=True
    assert es.early_stop is True


def test_early_stopping_no_activa_con_mejora_continua():
    es = EarlyStopping(patience=3)
    es(1.0)
    es(0.9)
    es(0.8)
    assert es.early_stop is False


# =========================================================
# EarlyStopping — edge cases
# =========================================================

def test_early_stopping_reinicia_contador_tras_mejora():
    es = EarlyStopping(patience=2)
    es(1.0)   # best=1.0, counter=0
    es(1.0)   # counter=1
    es(0.5)   # mejora → counter=0
    es(0.5)   # counter=1
    assert es.early_stop is False


def test_early_stopping_patience_uno_dispara_tras_dos_llamadas():
    es = EarlyStopping(patience=1)
    es(1.0)   # best_loss=1.0
    es(1.1)   # peor → counter=1 ≥ patience → early_stop=True
    assert es.early_stop is True


def test_early_stopping_primera_llamada_no_dispara():
    es = EarlyStopping(patience=1)
    es(1.0)
    assert es.early_stop is False


# =========================================================
# ModelCheckpoint — happy path
# =========================================================

def test_checkpoint_guarda_primer_modelo():
    mc = ModelCheckpoint()
    model = nn.Linear(2, 2)
    saved = mc(model, 0.5)
    assert saved is True
    assert mc.best_model_state is not None


def test_checkpoint_actualiza_con_f1_superior():
    mc = ModelCheckpoint()
    model = nn.Linear(2, 2)
    mc(model, 0.5)
    saved = mc(model, 0.7)
    assert saved is True
    assert mc.best_f1 == pytest.approx(0.7)


# =========================================================
# ModelCheckpoint — edge cases
# =========================================================

def test_checkpoint_no_guarda_f1_inferior():
    mc = ModelCheckpoint()
    model = nn.Linear(2, 2)
    mc(model, 0.5)
    saved = mc(model, 0.4)
    assert saved is False
    assert mc.best_f1 == pytest.approx(0.5)


def test_checkpoint_no_guarda_f1_igual():
    mc = ModelCheckpoint()
    model = nn.Linear(2, 2)
    mc(model, 0.5)
    saved = mc(model, 0.5)
    assert saved is False


def test_checkpoint_estado_inicial_sin_modelo():
    mc = ModelCheckpoint()
    assert mc.best_model_state is None
    assert mc.best_f1 == -float('inf')




# =========================================================
# aplicar_seleccion_clases — happy path
# =========================================================

def _df_etiquetas(filas):
    """Construye un DataFrame con las 13 columnas; las no indicadas valen 0.0."""
    rows = []
    for f in filas:
        row = {c: 0.0 for c in CHEXPERT_PATHOLOGY_COLS}
        row.update(f)
        rows.append(row)
    return pd.DataFrame(rows)


def test_seleccion_full13_modo_ninguno_no_elimina():
    df = _df_etiquetas([{"Fracture": 1.0}, {"Cardiomegaly": 1.0}, {}])
    df_out, rep = aplicar_seleccion_clases(
        df, CHEXPERT_PATHOLOGY_COLS, CHEXPERT_PATHOLOGY_COLS, "ninguno"
    )
    assert len(df_out) == 3
    assert rep["estudios_eliminados"] == 0
    assert rep["clases_descartadas"] == []


def test_seleccion_nofracture12_orfanos_elimina_solo_fracture_only():
    activas = get_active_pathology_cols("nofracture12")   # 12, sin Fracture
    df = _df_etiquetas([
        {"Fracture": 1.0},                       # Fracture-only -> se elimina
        {"Fracture": 1.0, "Cardiomegaly": 1.0},  # tiene positivo activo -> se conserva
        {"Cardiomegaly": 1.0},                   # se conserva
        {},                                      # ya negativo en origen -> orfanos no lo elimina
    ])
    df_out, rep = aplicar_seleccion_clases(df, activas, CHEXPERT_PATHOLOGY_COLS, "orfanos")
    assert rep["estudios_eliminados"] == 1
    assert len(df_out) == 3
    assert rep["clases_descartadas"] == ["Fracture"]


# =========================================================
# aplicar_seleccion_clases — edge cases
# =========================================================

def test_seleccion_min5pct9_sin_positivos_elimina_tambien_negativos_de_origen():
    activas = get_active_pathology_cols("min5pct9")   # 9
    df = _df_etiquetas([
        {"Pneumonia": 1.0},      # solo positivo en clase descartada -> se elimina
        {},                      # negativo de origen -> sin_positivos también lo elimina
        {"Cardiomegaly": 1.0},   # positivo activo -> se conserva
    ])
    df_out, rep = aplicar_seleccion_clases(df, activas, CHEXPERT_PATHOLOGY_COLS, "sin_positivos")
    assert rep["estudios_eliminados"] == 2
    assert len(df_out) == 1


def test_seleccion_df_vacio_no_lanza():
    df = _df_etiquetas([]).reindex(columns=CHEXPERT_PATHOLOGY_COLS)
    activas = get_active_pathology_cols("nofracture12")
    df_out, rep = aplicar_seleccion_clases(df, activas, CHEXPERT_PATHOLOGY_COLS, "orfanos")
    assert len(df_out) == 0
    assert rep["estudios_eliminados"] == 0


def test_seleccion_reporte_contiene_claves_esperadas():
    df = _df_etiquetas([{"Cardiomegaly": 1.0}])
    _, rep = aplicar_seleccion_clases(
        df, get_active_pathology_cols("nofracture12"), CHEXPERT_PATHOLOGY_COLS, "orfanos"
    )
    esperadas = {"clases_activas", "clases_descartadas", "modo_anti_ruido",
                 "estudios_antes", "estudios_eliminados", "estudios_despues"}
    assert esperadas.issubset(rep.keys())


# =========================================================
# aplicar_seleccion_clases — errores
# =========================================================

def test_seleccion_modo_invalido_lanza_valueerror():
    df = _df_etiquetas([{"Cardiomegaly": 1.0}])
    with pytest.raises(ValueError, match="modo_anti_ruido"):
        aplicar_seleccion_clases(
            df, get_active_pathology_cols("nofracture12"), CHEXPERT_PATHOLOGY_COLS, "desconocido"
        )
