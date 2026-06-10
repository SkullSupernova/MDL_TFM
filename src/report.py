"""
Generación del informe PDF de un análisis de radiografía.

Módulo de presentación independiente de Streamlit: recibe los datos ya calculados
(predicciones, imagen original, paneles Grad-CAM, métricas del modelo) y devuelve los
bytes de un PDF profesional. Al no depender de la UI, es reutilizable y testeable.

El informe incluye: cabecera del estudio, resumen de hallazgos, tabla de probabilidades
por patología, gráfica de barras, paneles original+Grad-CAM de las clases más probables,
métricas de validación del modelo (si se aportan) y un aviso legal.

Dependencia: reportlab (PDF en Python puro, sin dependencias de sistema).
"""

from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # backend headless: render a buffer, sin ventana
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Etiquetas legibles de las métricas de validación del modelo (claves de test_metrics).
_METRICAS_DISPLAY = [
    ("auroc_chexpert5", "AUROC CheXpert-5"),
    ("auroc_macro_evaluable", "AUROC-macro (evaluable)"),
    ("pr_auc_macro_evaluable", "PR-AUC-macro"),
    ("f1_macro", "F1-macro"),
    ("f1_micro", "F1-micro"),
    ("accuracy", "Accuracy"),
]

_AVISO_LEGAL = (
    "Este informe ha sido generado automáticamente por un modelo de inteligencia "
    "artificial con fines de apoyo y demostración. No constituye un diagnóstico médico "
    "ni sustituye la valoración de un profesional sanitario cualificado. Las "
    "probabilidades reflejan la confianza del modelo, no certezas clínicas."
)

_AZUL = colors.HexColor("#1f4e79")
_GRIS_CABECERA = colors.HexColor("#dfe6ef")
_VERDE = colors.HexColor("#d4edda")


def _np_to_rlimage(arr: np.ndarray, lado_mm: float) -> RLImage:
    """Convierte un array uint8 (H, W, 3) en una imagen de reportlab cuadrada."""
    buf = BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=lado_mm * mm, height=lado_mm * mm)


def _bar_chart_png(filas: List[Dict]) -> bytes:
    """Renderiza la gráfica de barras horizontal de probabilidades por patología."""
    labels = [f["patologia"] for f in filas]
    probs = [f["probabilidad"] for f in filas]
    detect = [f["detectada"] for f in filas]
    colores = ["#28a745" if d else "#6c757d" for d in detect]

    fig, ax = plt.subplots(figsize=(7.2, max(2.5, 0.34 * len(labels))))
    y = np.arange(len(labels))
    # zorder alto + set_axisbelow para que las barras queden por encima de la cuadrícula.
    bars = ax.barh(y, probs, color=colores, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # la patología más probable arriba
    # Margen extra (1.08) para que la etiqueta de valor de las barras largas no se recorte.
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Probabilidad", fontsize=9)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, color="#b0b0b0", alpha=0.7)
    ax.set_axisbelow(True)
    # Valor exacto de cada barra al final de la misma.
    ax.bar_label(bars, labels=[f"{p:.2f}" for p in probs], padding=3, fontsize=7, color="#333333")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _bar_chart_comparacion_png(labels: List[str], probs_a: List[float], probs_b: List[float],
                               etiqueta_a: str, etiqueta_b: str) -> bytes:
    """Barras agrupadas horizontales (modelo A vs B) sobre las patologías comunes, con valores."""
    n = len(labels)
    y = np.arange(n)
    alto = 0.38
    fig, ax = plt.subplots(figsize=(7.2, max(2.8, 0.62 * n)))
    ba = ax.barh(y - alto / 2, probs_a, height=alto, color="#1f77b4", zorder=3, label=etiqueta_a)
    bb = ax.barh(y + alto / 2, probs_b, height=alto, color="#ff7f0e", zorder=3, label=etiqueta_b)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Probabilidad", fontsize=9)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, color="#b0b0b0", alpha=0.7)
    ax.set_axisbelow(True)
    ax.bar_label(ba, labels=[f"{p:.2f}" for p in probs_a], padding=2, fontsize=6, color="#333333")
    ax.bar_label(bb, labels=[f"{p:.2f}" for p in probs_b], padding=2, fontsize=6, color="#333333")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _tabla_probabilidades(filas: List[Dict]) -> Table:
    """Tabla reportlab de probabilidades por patología, resaltando las detectadas."""
    data = [["Patología", "Probabilidad", "Detectada"]]
    for f in filas:
        data.append([f["patologia"], f"{f['probabilidad']:.4f}", "✓" if f["detectada"] else ""])
    tabla = Table(data, colWidths=[95 * mm, 40 * mm, 30 * mm], repeatRows=1)
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), _AZUL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f9")]),
    ]
    for i, f in enumerate(filas, start=1):
        if f["detectada"]:
            estilo.append(("BACKGROUND", (0, i), (-1, i), _VERDE))
    tabla.setStyle(TableStyle(estilo))
    return tabla


def _imagen_grafica(filas: List[Dict]) -> RLImage:
    """RLImage de la gráfica de barras de probabilidades de un modelo (ancho fijo de página)."""
    fig_alto_in = max(2.5, 0.34 * len(filas))
    ancho_mm = 170
    chart_buf = BytesIO(_bar_chart_png(filas))
    return RLImage(chart_buf, width=ancho_mm * mm, height=ancho_mm * (fig_alto_in / 7.2) * mm)


def _tabla_metricas(metricas: Optional[Dict]) -> Optional[Table]:
    """Tabla reportlab de las métricas de validación, o None si no hay métricas."""
    mfilas = [["Métrica", "Valor"]]
    for clave, etiqueta in _METRICAS_DISPLAY:
        if metricas and metricas.get(clave) is not None:
            mfilas.append([etiqueta, f"{float(metricas[clave]):.4f}"])
    if len(mfilas) == 1:
        return None
    tmet = Table(mfilas, colWidths=[100 * mm, 65 * mm])
    tmet.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _AZUL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
    ]))
    return tmet


def build_report_pdf(contexto: Dict) -> bytes:
    """
    Construye el informe PDF y devuelve sus bytes.

    Parámetros
    ----------
    contexto : dict con las claves:
        - "filas": list[dict] con "patologia", "probabilidad" (float), "detectada" (bool),
          ordenadas de mayor a menor probabilidad. (obligatoria)
        - "original": np.ndarray uint8 (H, W, 3), radiografía a 224×224. (obligatoria)
        - "panels": list[dict] con "label", "prob" (float) y "heatmap" (np.ndarray uint8),
          una entrada por clase explicada. (obligatoria)
        - "modelo", "class_config", "checkpoint", "imagen_nombre", "umbral" (float),
          "detectadas" (list[str]), "metricas_modelo" (dict|None), "titulo" (opcionales).
        - Modo comparación de dos modelos (opcional): "comparacion" (bool), "modelo_b",
          "class_config_b", "filas_b" (mismo formato que "filas"), "metricas_modelo_b"
          (dict|None) y, en cada panel, "heatmap_b" (np.ndarray uint8 o None) con el Grad-CAM
          del segundo modelo.

    Devuelve
    --------
    bytes del PDF.

    Raises
    ------
    ValueError: si falta alguna clave obligatoria.
    """
    for clave in ("filas", "original", "panels"):
        if clave not in contexto:
            raise ValueError(f"Falta la clave obligatoria '{clave}' en el contexto del informe.")

    filas = contexto["filas"]
    umbral = float(contexto.get("umbral", 0.5))
    # Se derivan de filas (ya ordenadas por probabilidad) para que el resumen siga el
    # mismo orden que la tabla y la gráfica.
    detectadas = [f["patologia"] for f in filas if f["detectada"]]
    metricas = contexto.get("metricas_modelo")
    fecha = contexto.get("fecha") or datetime.now().strftime("%Y-%m-%d %H:%M")

    # Modo comparación de dos modelos (todas las claves *_b son opcionales).
    comparacion = bool(contexto.get("comparacion"))
    filas_b = contexto.get("filas_b") or []
    modelo_a = str(contexto.get("modelo", "—"))
    modelo_b = str(contexto.get("modelo_b", "—"))
    cfg_a = contexto.get("class_config")
    cfg_b = contexto.get("class_config_b")
    etiqueta_a = f"A · {modelo_a}" + (f" ({cfg_a})" if cfg_a else "")
    etiqueta_b = f"B · {modelo_b}" + (f" ({cfg_b})" if cfg_b else "")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=_AZUL, fontSize=18)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=_AZUL, fontSize=12, spaceBefore=10)
    normal = styles["BodyText"]
    pequeno = ParagraphStyle("small", parent=normal, fontSize=8, textColor=colors.grey)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=contexto.get("titulo", "Informe de análisis de radiografía torácica"),
    )
    story: list = []

    # --- Cabecera -----------------------------------------------------------
    story.append(Paragraph(contexto.get("titulo", "Informe de análisis de radiografía torácica"), h1))
    meta = [["Fecha", fecha, "Imagen", contexto.get("imagen_nombre", "—")]]
    if comparacion:
        meta.append(["Modelo A", f"{modelo_a} · {cfg_a or '—'}", "Umbral", f"{umbral:.2f}"])
        meta.append(["Modelo B", f"{modelo_b} · {cfg_b or '—'}", "", ""])
    else:
        cfg_txt = f"{modelo_a} · {cfg_a}" if cfg_a else modelo_a
        meta.append(["Modelo", cfg_txt, "Umbral", f"{umbral:.2f}"])
    tabla_meta = Table(meta, colWidths=[22 * mm, 62 * mm, 22 * mm, 62 * mm])
    tabla_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _GRIS_CABECERA),
        ("BACKGROUND", (2, 0), (2, -1), _GRIS_CABECERA),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tabla_meta)
    story.append(Spacer(1, 6))

    # --- Resumen de hallazgos (modelo A) -----------------------------------
    story.append(Paragraph("Resumen de hallazgos", h2))
    suf_a = f" (Modelo A: {modelo_a})" if comparacion else ""
    if detectadas:
        resumen = "Patologías detectadas{} (umbral {:.2f}): <b>{}</b>.".format(
            suf_a, umbral, ", ".join(detectadas))
    else:
        resumen = f"Ninguna patología supera el umbral de {umbral:.2f}{suf_a}."
    story.append(Paragraph(resumen, normal))

    # --- Probabilidades -----------------------------------------------------
    if comparacion:
        story.append(Paragraph(f"Probabilidades — Modelo A ({modelo_a})", h2))
        story.append(_tabla_probabilidades(filas))
        story.append(Spacer(1, 4))
        story.append(_imagen_grafica(filas))
        if filas_b:
            story.append(Paragraph(f"Probabilidades — Modelo B ({modelo_b})", h2))
            story.append(_tabla_probabilidades(filas_b))
            story.append(Spacer(1, 4))
            story.append(_imagen_grafica(filas_b))
            # Gráfica comparativa sobre las patologías comunes a ambos modelos.
            comunes_b = {g["patologia"] for g in filas_b}
            comunes = [f["patologia"] for f in filas if f["patologia"] in comunes_b]
            if comunes:
                prob_a = {f["patologia"]: f["probabilidad"] for f in filas}
                prob_b = {g["patologia"]: g["probabilidad"] for g in filas_b}
                story.append(Paragraph("Comparación (patologías comunes)", h2))
                comp_png = _bar_chart_comparacion_png(
                    comunes, [prob_a[c] for c in comunes], [prob_b[c] for c in comunes],
                    etiqueta_a, etiqueta_b,
                )
                comp_alto_in = max(2.8, 0.62 * len(comunes))
                story.append(RLImage(BytesIO(comp_png), width=170 * mm,
                                     height=170 * (comp_alto_in / 7.2) * mm))
    else:
        story.append(Paragraph("Probabilidades por patología", h2))
        story.append(_tabla_probabilidades(filas))
        story.append(Spacer(1, 4))
        story.append(_imagen_grafica(filas))

    # --- Paneles original + Grad-CAM ---------------------------------------
    panels = contexto["panels"]
    if panels:
        story.append(Paragraph("Explicabilidad visual (Grad-CAM)", h2))
        if comparacion:
            descripcion = ("Para cada patología: la radiografía original y el mapa de calor de cada modelo "
                           "(A y B). Las zonas cálidas indican las regiones que más influyeron en la predicción.")
        else:
            descripcion = ("Para cada patología: a la izquierda la radiografía original; a la derecha el mapa de "
                           "calor, donde las zonas cálidas indican las regiones que más influyeron en la predicción.")
        story.append(Paragraph(descripcion, pequeno))
        story.append(Spacer(1, 4))
        original = contexto["original"]
        cap = ParagraphStyle("cap", parent=pequeno, alignment=1)  # texto centrado
        for p in panels:
            if comparacion:
                col_b = (_np_to_rlimage(p["heatmap_b"], 54) if p.get("heatmap_b") is not None
                         else Paragraph("No presente en el modelo B", pequeno))
                par = Table(
                    [[_np_to_rlimage(original, 54), _np_to_rlimage(p["heatmap"], 54), col_b],
                     [Paragraph("Original", cap), Paragraph("Grad-CAM · A", cap), Paragraph("Grad-CAM · B", cap)]],
                    colWidths=[56 * mm, 56 * mm, 56 * mm],
                )
            else:
                par = Table(
                    [[_np_to_rlimage(original, 72), _np_to_rlimage(p["heatmap"], 72)],
                     [Paragraph("Original", cap), Paragraph("Grad-CAM", cap)]],
                    colWidths=[82 * mm, 82 * mm],
                )
            par.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            # KeepTogether evita que el título quede en una página y sus imágenes en la
            # siguiente (título huérfano) al cruzar un salto de página.
            story.append(KeepTogether([
                Paragraph(f"<b>{p['label']}</b> — {p['prob'] * 100:.1f}%", normal),
                par,
                Spacer(1, 6),
            ]))

    # --- Métricas de validación (si existen) -------------------------------
    tmet_a = _tabla_metricas(metricas)
    if comparacion:
        if tmet_a:
            story.append(Paragraph(f"Fiabilidad validada — Modelo A ({modelo_a})", h2))
            story.append(tmet_a)
            story.append(Spacer(1, 6))
        tmet_b = _tabla_metricas(contexto.get("metricas_modelo_b"))
        if tmet_b:
            story.append(Paragraph(f"Fiabilidad validada — Modelo B ({modelo_b})", h2))
            story.append(tmet_b)
            story.append(Spacer(1, 6))
    elif tmet_a:
        story.append(Paragraph("Fiabilidad validada del modelo (test silver-standard)", h2))
        story.append(tmet_a)
        story.append(Spacer(1, 6))

    # --- Aviso legal --------------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph(_AVISO_LEGAL, pequeno))

    doc.build(story)
    return buffer.getvalue()
