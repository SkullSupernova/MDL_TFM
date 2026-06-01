"""
Evaluación final sobre el conjunto de test "silver standard" (valid oficial de Stanford).

Este módulo es independiente del bucle de entrenamiento: puede evaluar cualquier
checkpoint sobre el test sin reentrenar, lo que lo hace adecuado para experimentos
comparativos entre arquitecturas. También lo invoca src/main.py al terminar el
entrenamiento para reportar una métrica final no sesgada por la selección de modelo.

Uso:
    python -m src.evaluate
    python -m src.evaluate --checkpoint models/mejor_modelo_resnet50.pth --model resnet50
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, f1_score

from src.logging_config import get_logger
from src.models import (
    CheXpertDataset,
    get_pathology_labels,
    load_checkpoint,
    CHEXPERT_COMPETITION_5,
)
from src.utils import (
    setup_environment,
    calculate_metrics,
    construir_df_test_valid,
    auc_por_clase,
    auroc_macro,
    pr_auc_por_clase,
    pr_auc_macro,
)

logger = get_logger(__name__)

_LOG_DIR = Path("logs")

# Umbral de decisión para las métricas de test. Se fija en 0.5 para que sean
# directamente comparables con las métricas de validación del entrenamiento
# (ver src/train.py), que también umbralizan a 0.5.
_THRESHOLD = 0.5

# Transformación de evaluación: idéntica a validación e inferencia (sin data
# augmentation). Normalización ImageNet, imprescindible para coincidir con la
# distribución de entrada vista por el modelo durante el entrenamiento.
_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ejecuta inferencia sobre un loader y devuelve (y_true, y_pred, y_prob).

    y_pred se obtiene umbralizando las probabilidades a 0.5, el mismo umbral que
    usa la fase de validación del entrenamiento, para que las métricas sean comparables.

    Devuelve tres matrices (n_muestras, n_clases): etiquetas reales, predicciones
    binarias y probabilidades.
    """
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            all_prob.append(probs)
            all_true.append(labels.numpy())
    y_true = np.vstack(all_true)
    y_prob = np.vstack(all_prob)
    y_pred = (y_prob > _THRESHOLD).astype(np.float32)
    return y_true, y_pred, y_prob


def construir_test_loader(cfg: dict, etiquetas_cols: List[str], df_test=None):
    """
    Construye (DataLoader, df_test) del test silver-standard a partir de config.yml.

    Si se pasa df_test (ya construido), se reutiliza para evitar releer el valid.csv y
    re-resolver rutas; esto permite compartir el mismo conjunto entre la documentación
    de la distribución y la evaluación.
    """
    if df_test is None:
        df_test = construir_df_test_valid(
            cfg["data"]["test_csv_path"],
            cfg["data"]["test_images_root"],
            etiquetas_cols,
        )
    dataset = CheXpertDataset(df_test, transform=_EVAL_TRANSFORM, etiquetas_cols=etiquetas_cols)
    # num_workers por defecto (0): el test es pequeño (~169 imágenes) y así se evita
    # la limitación de multiprocessing del DataLoader en Windows.
    loader = DataLoader(dataset, batch_size=cfg["training"]["inference_batch_size"], shuffle=False)
    return loader, df_test


def calcular_metricas_completas(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    labels: List[str], threshold: float = _THRESHOLD
) -> Dict:
    """
    Calcula el conjunto completo de métricas de un split (val o test) en formato serializable.

    Incluye, a nivel global: accuracy, F1-macro/micro, AUROC-macro evaluable, AUROC de las 5
    de CheXpert y PR-AUC-macro; y por clase: soporte, AUROC, PR-AUC, precision, recall, F1.
    Las clases sin positivos (o sin ambos valores) tienen AUROC/PR-AUC = None y se listan en
    'clases_no_evaluables' para documentar explícitamente que no admiten métrica fiable.
    """
    base = calculate_metrics(y_true, y_pred)
    aucs = auc_por_clase(y_true, y_prob, labels)
    praucs = pr_auc_por_clase(y_true, y_prob, labels)
    auroc_ev, n_auroc = auroc_macro(y_true, y_prob)
    prauc_ev, _ = pr_auc_macro(y_true, y_prob)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    chexpert5 = [aucs[c] for c in CHEXPERT_COMPETITION_5 if aucs.get(c) is not None]
    auroc5 = float(np.mean(chexpert5)) if chexpert5 else None

    per_class, no_evaluables = {}, []
    for i, lab in enumerate(labels):
        sop = int(support[i])
        per_class[lab] = {
            "soporte": sop,
            "auroc": aucs[lab],
            "pr_auc": praucs[lab],
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
        }
        if aucs[lab] is None or sop == 0:
            no_evaluables.append(lab)

    return {
        "n_muestras": int(len(y_true)),
        "threshold": threshold,
        "accuracy": base["accuracy"],
        "f1_macro": base["f1_macro"],
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "auroc_macro_evaluable": auroc_ev,
        "auroc_chexpert5": auroc5,
        "pr_auc_macro_evaluable": prauc_ev,
        "n_clases_auroc_evaluables": n_auroc,
        "n_clases_chexpert5_evaluables": len(chexpert5),
        "per_class": per_class,
        "clases_no_evaluables": no_evaluables,
    }


def evaluar_loader(
    model: torch.nn.Module, loader: DataLoader, labels: List[str],
    device: torch.device, threshold: float = _THRESHOLD
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa un loader y devuelve (métricas_completas, y_true, y_pred, y_prob)."""
    y_true, y_pred, y_prob = evaluate_model(model, loader, device)
    metrics = calcular_metricas_completas(y_true, y_pred, y_prob, labels, threshold)
    return metrics, y_true, y_pred, y_prob


def _reportar_por_clase(metrics: Dict, labels: List[str], model_name: str) -> None:
    """Registra por log la tabla de métricas por clase y la guarda en CSV (logs/)."""
    logger.info("Métricas por clase (AUROC/PR-AUC independientes del umbral; P/R/F1 a umbral 0.5):")
    logger.info(
        f"  {'Patología':30s}{'AUROC':>8}{'PR-AUC':>8}{'Precision':>10}{'Recall':>9}{'F1':>9}{'Soporte':>9}"
    )
    filas = []
    for lab in labels:
        m = metrics["per_class"][lab]
        auc, prauc, sop = m["auroc"], m["pr_auc"], m["soporte"]
        auc_str = f"{auc:8.4f}" if auc is not None else f"{'n/a':>8}"
        pr_str = f"{prauc:8.4f}" if prauc is not None else f"{'n/a':>8}"
        aviso = "   <-- no evaluable (soporte insuficiente)" if lab in metrics["clases_no_evaluables"] else ""
        logger.info(
            f"  {lab:30s}{auc_str}{pr_str}{m['precision']:10.4f}{m['recall']:9.4f}{m['f1']:9.4f}{sop:9d}{aviso}"
        )
        filas.append({
            "patologia": lab,
            "auroc": round(auc, 4) if auc is not None else "",
            "pr_auc": round(prauc, 4) if prauc is not None else "",
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "soporte": sop,
        })

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = _LOG_DIR / f"test_metrics_{model_name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["patologia", "auroc", "pr_auc", "precision", "recall", "f1", "soporte"]
        )
        writer.writeheader()
        writer.writerows(filas)
    logger.info(f"Métricas por clase guardadas en: {out}")


def evaluar_test(
    cfg: dict, model: torch.nn.Module, device: torch.device, num_classes: int, df_test=None
):
    """
    Evalúa un modelo sobre el test silver-standard.

    Métrica principal de promoción: AUROC media de las 5 patologías oficiales de CheXpert
    (CHEXPERT_COMPETITION_5), bien representadas en este test. Devuelve la tupla
    (metrics, y_true, y_pred, y_prob, df_test): el dict de `calcular_metricas_completas`
    (global + por clase + clases no evaluables) y las predicciones crudas para trazabilidad.

    Raises:
        FileNotFoundError: si el valid.csv configurado no existe.
    """
    labels = get_pathology_labels(num_classes)
    loader, df_test = construir_test_loader(cfg, labels, df_test)
    metrics, y_true, y_pred, y_prob = evaluar_loader(model, loader, labels, device, _THRESHOLD)

    logger.info("=== Evaluación en test (silver standard) ===")
    a5 = f"{metrics['auroc_chexpert5']:.4f}" if metrics["auroc_chexpert5"] is not None else "n/a"
    logger.info(
        f"Muestras: {metrics['n_muestras']} | AUROC CheXpert-5: {a5} "
        f"({metrics['n_clases_chexpert5_evaluables']}/5 clases) | "
        f"AUROC-macro evaluable: {metrics['auroc_macro_evaluable']:.4f} "
        f"({metrics['n_clases_auroc_evaluables']} clases) | "
        f"PR-AUC-macro: {metrics['pr_auc_macro_evaluable']:.4f} | "
        f"F1-macro: {metrics['f1_macro']:.4f} | Accuracy: {metrics['accuracy']:.4f}"
    )
    _reportar_por_clase(metrics, labels, cfg["model"]["name"])
    return metrics, y_true, y_pred, y_prob, df_test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación sobre el test silver-standard")
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Ruta al checkpoint. Por defecto usa model.checkpoint_path de config.yml."
    )
    parser.add_argument(
        "--model", default=None,
        help="Backbone del checkpoint. Sobreescribe model.name de config.yml."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["model"]["name"] = args.model

    checkpoint = args.checkpoint or cfg["model"]["checkpoint_path"]
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: '{checkpoint}'. "
            "Entrena el modelo o indica --checkpoint."
        )

    device, _ = setup_environment()
    model, num_classes = load_checkpoint(cfg, checkpoint, device)
    logger.info(f"Checkpoint cargado: {checkpoint} ({num_classes} clases)")
    evaluar_test(cfg, model, device, num_classes)


if __name__ == "__main__":
    main()
