import numpy as np
import pytest

from src.report import build_report_pdf


def _contexto(n_panels=5, con_metricas=True, detectadas=None, n_clases=6):
    labels = [f"Patologia {i}" for i in range(n_clases)]
    probs = np.linspace(0.9, 0.1, n_clases)
    filas = [
        {"patologia": lab, "probabilidad": float(p), "detectada": bool(p >= 0.5)}
        for lab, p in zip(labels, probs)
    ]
    original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    panels = [
        {
            "label": labels[i],
            "prob": float(probs[i]),
            "heatmap": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        }
        for i in range(n_panels)
    ]
    ctx = {
        "filas": filas,
        "original": original,
        "panels": panels,
        "modelo": "densenet121",
        "class_config": "nofracture12",
        "imagen_nombre": "radiografia.jpg",
        "umbral": 0.5,
        "detectadas": detectadas if detectadas is not None
        else [f["patologia"] for f in filas if f["detectada"]],
    }
    if con_metricas:
        ctx["metricas_modelo"] = {
            "auroc_chexpert5": 0.83, "f1_macro": 0.51, "accuracy": 0.72,
        }
    return ctx


# =========================================================
# build_report_pdf — happy path
# =========================================================

def test_build_report_pdf_devuelve_pdf_valido():
    pdf = build_report_pdf(_contexto())
    assert isinstance(pdf, bytes)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1000


def test_build_report_pdf_sin_metricas_de_modelo():
    pdf = build_report_pdf(_contexto(con_metricas=False))
    assert pdf[:4] == b"%PDF"


# =========================================================
# build_report_pdf — edge cases
# =========================================================

@pytest.mark.parametrize("n_panels", [5, 8])
def test_build_report_pdf_distinto_numero_de_paneles(n_panels):
    pdf = build_report_pdf(_contexto(n_panels=n_panels, n_clases=max(n_panels, 6)))
    assert pdf[:4] == b"%PDF"


def test_build_report_pdf_sin_detectadas():
    ctx = _contexto(detectadas=[])
    for f in ctx["filas"]:
        f["detectada"] = False
    pdf = build_report_pdf(ctx)
    assert pdf[:4] == b"%PDF"


def test_build_report_pdf_sin_paneles_no_lanza():
    pdf = build_report_pdf(_contexto(n_panels=0))
    assert pdf[:4] == b"%PDF"


# =========================================================
# build_report_pdf — errores
# =========================================================

@pytest.mark.parametrize("clave", ["filas", "original", "panels"])
def test_build_report_pdf_falta_clave_obligatoria_lanza_valueerror(clave):
    ctx = _contexto()
    del ctx[clave]
    with pytest.raises(ValueError, match=clave):
        build_report_pdf(ctx)
