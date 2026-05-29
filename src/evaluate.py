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
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support

from src.logging_config import get_logger
from src.models import CheXpertDataset, get_pathology_labels, load_checkpoint
from src.utils import setup_environment, calculate_metrics, construir_df_test_valid

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


def construir_test_loader(cfg: dict, etiquetas_cols: List[str]) -> DataLoader:
    """Construye el DataLoader del test silver-standard a partir de config.yml."""
    df_test = construir_df_test_valid(
        cfg["data"]["test_csv_path"],
        cfg["data"]["test_images_root"],
        etiquetas_cols,
    )
    dataset = CheXpertDataset(df_test, transform=_EVAL_TRANSFORM, etiquetas_cols=etiquetas_cols)
    # num_workers por defecto (0): el test es pequeño (~169 imágenes) y así se evita
    # la limitación de multiprocessing del DataLoader en Windows.
    return DataLoader(dataset, batch_size=cfg["training"]["inference_batch_size"], shuffle=False)


def _reportar_por_clase(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], model_name: str
) -> List[dict]:
    """Calcula métricas por clase, las registra y las guarda en CSV. Devuelve las filas."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    logger.info("Métricas por clase (umbral 0.5):")
    logger.info(f"  {'Patología':30s}{'Precision':>10}{'Recall':>9}{'F1':>9}{'Soporte':>9}")
    filas = []
    for i, lab in enumerate(labels):
        sop = int(support[i])
        # Una clase sin positivos reales en el test no permite estimar recall/F1:
        # se marca para que el sesgo por falta de soporte sea explícito en el reporte.
        aviso = "   <-- soporte 0: no fiable" if sop == 0 else ""
        logger.info(f"  {lab:30s}{precision[i]:10.4f}{recall[i]:9.4f}{f1[i]:9.4f}{sop:9d}{aviso}")
        filas.append({
            "patologia": lab,
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "soporte": sop,
        })

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = _LOG_DIR / f"test_metrics_{model_name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patologia", "precision", "recall", "f1", "soporte"])
        writer.writeheader()
        writer.writerows(filas)
    logger.info(f"Métricas por clase guardadas en: {out}")
    return filas


def evaluar_test(
    cfg: dict, model: torch.nn.Module, device: torch.device, num_classes: int
) -> Dict[str, float]:
    """
    Evalúa un modelo ya cargado sobre el test silver-standard y devuelve las métricas globales.

    Registra accuracy y F1-macro globales y una tabla de métricas por clase con su
    soporte (nº de positivos reales), marcando las clases sin soporte como no fiables.

    Raises:
        FileNotFoundError: si el valid.csv configurado no existe.
    """
    labels = get_pathology_labels(num_classes)
    loader = construir_test_loader(cfg, labels)
    y_true, y_pred, _ = evaluate_model(model, loader, device)

    metrics = calculate_metrics(y_true, y_pred)
    logger.info("=== Evaluación en test (silver standard) ===")
    logger.info(
        f"Muestras: {len(y_true)} | Accuracy: {metrics['accuracy']:.4f} | "
        f"F1-macro: {metrics['f1_macro']:.4f}"
    )
    _reportar_por_clase(y_true, y_pred, labels, cfg["model"]["name"])
    return metrics


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
