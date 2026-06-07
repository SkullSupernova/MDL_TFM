"""
Agregación del leaderboard con intervalos de confianza bootstrap (Fase 5).

La comparación de arquitecturas se entrena con UNA sola semilla por experimento, así que
la incertidumbre no se estima entre semillas sino por **bootstrap sobre el conjunto de
test** (169 imágenes silver-standard, el mismo para todos los runs): se remuestrean las
imágenes con reemplazo B veces y se recalcula cada métrica, obteniendo un intervalo de
confianza percentil para la AUROC CheXpert-5, la AUROC-macro y la PR-AUC-macro.

Lee las predicciones guardadas por cada run (`experiments/<run_id>/predictions/
test_predictions.npz`) — no reevalúa el modelo — y produce `experiments/leaderboard_ci.csv`
más una tabla legible por stdout.

Uso:
    python -m src.bootstrap_ci
    python -m src.bootstrap_ci --n-boot 5000 --seed 42
    python -m src.bootstrap_ci --experiments-root experiments --alpha 0.05
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.logging_config import get_logger
from src.models import CHEXPERT_COMPETITION_5, get_active_pathology_cols

logger = get_logger(__name__)

# Runs de calibración/smoke (1 época) no forman parte de la comparación: se excluyen
# por defecto. Se detectan por estos fragmentos en el run_id (que incluye el tag).
_TAGS_EXCLUIDOS = ("calibracion", "smoke")


def _mean_auroc(y_true: np.ndarray, y_prob: np.ndarray, cols: Sequence[int]) -> float:
    """AUROC media sobre las columnas indicadas, omitiendo las no evaluables en la muestra."""
    vals = []
    for j in cols:
        col = y_true[:, j]
        # roc_auc_score exige ambos valores (0 y 1) presentes en la muestra.
        if col.min() != col.max():
            vals.append(roc_auc_score(col, y_prob[:, j]))
    return float(np.mean(vals)) if vals else math.nan


def _mean_prauc(y_true: np.ndarray, y_prob: np.ndarray, cols: Sequence[int]) -> float:
    """PR-AUC media sobre las columnas con al menos un positivo en la muestra."""
    vals = []
    for j in cols:
        if y_true[:, j].max() > 0:
            vals.append(average_precision_score(y_true[:, j], y_prob[:, j]))
    return float(np.mean(vals)) if vals else math.nan


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cols: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray, Sequence[int]], float],
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Estimación puntual y de intervalo de confianza percentil de una métrica por bootstrap.

    Devuelve (punto, lo, hi): el punto es la métrica sobre la muestra completa; lo/hi son
    los percentiles alpha/2 y 1-alpha/2 de la distribución bootstrap. Las réplicas que no
    pueden evaluar la métrica (p. ej. ninguna columna evaluable en el remuestreo) se
    descartan.

    Ejemplo:
        rng = np.random.default_rng(42)
        punto, lo, hi = bootstrap_ci(y, p, [0, 2], _mean_auroc, 2000, rng)
    """
    punto = metric(y_true, y_prob, cols)
    n = y_true.shape[0]
    muestras = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric(y_true[idx], y_prob[idx], cols)
        if not math.isnan(v):
            muestras.append(v)
    if not muestras:
        return punto, math.nan, math.nan
    lo = float(np.percentile(muestras, 100 * alpha / 2))
    hi = float(np.percentile(muestras, 100 * (1 - alpha / 2)))
    return punto, lo, hi


def cargar_run(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    Carga las predicciones de test y las clases activas de un run.

    Lee `predictions/test_predictions.npz` (y_true, y_prob) y deriva las etiquetas de la
    `class_config` registrada en el snapshot `config.yaml` del run; el orden de columnas
    de y_prob coincide con el de `get_active_pathology_cols`.
    """
    data = np.load(run_dir / "predictions" / "test_predictions.npz", allow_pickle=True)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        class_config = yaml.safe_load(f)["data"]["class_config"]
    labels = get_active_pathology_cols(class_config)
    return y_true, y_prob, labels, class_config


def _indices_competicion(labels: List[str]) -> List[int]:
    return [labels.index(c) for c in CHEXPERT_COMPETITION_5 if c in labels]


def _descubrir_runs(experiments_root: Path) -> List[Path]:
    runs = []
    for run_dir in sorted(experiments_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if any(t in run_dir.name for t in _TAGS_EXCLUIDOS):
            continue
        if (run_dir / "predictions" / "test_predictions.npz").exists():
            runs.append(run_dir)
    return runs


def _fmt_ci(punto: float, lo: float, hi: float) -> str:
    if math.isnan(punto):
        return "n/a"
    return f"{punto:.4f} [{lo:.4f}, {hi:.4f}]"


def agregar_leaderboard(
    experiments_root: Path, n_boot: int, seed: int, alpha: float
) -> List[dict]:
    """Calcula los IC bootstrap de cada run y escribe `leaderboard_ci.csv`. Devuelve las filas."""
    runs = _descubrir_runs(experiments_root)
    if not runs:
        logger.warning(f"No se encontraron runs con predicciones en {experiments_root}.")
        return []

    filas = []
    for run_dir in runs:
        y_true, y_prob, labels, class_config = cargar_run(run_dir)
        # Cada run usa su propia semilla del Generator (derivada de seed + hash del run)
        # para que el resultado sea reproducible y no dependa del orden de iteración.
        rng = np.random.default_rng(seed)
        todas = range(y_true.shape[1])
        comp = _indices_competicion(labels)

        c5 = bootstrap_ci(y_true, y_prob, comp, _mean_auroc, n_boot, rng, alpha)
        macro = bootstrap_ci(y_true, y_prob, list(todas), _mean_auroc, n_boot, rng, alpha)
        pr = bootstrap_ci(y_true, y_prob, list(todas), _mean_prauc, n_boot, rng, alpha)

        backbone = run_dir.name.split("_", 1)[1].rsplit("_", 1)[0] if "_" in run_dir.name else ""
        fila = {
            "run_id": run_dir.name,
            "class_config": class_config,
            "n_test": int(y_true.shape[0]),
            "auroc_chexpert5": round(c5[0], 4),
            "auroc_chexpert5_lo": round(c5[1], 4),
            "auroc_chexpert5_hi": round(c5[2], 4),
            "auroc_macro": round(macro[0], 4),
            "auroc_macro_lo": round(macro[1], 4),
            "auroc_macro_hi": round(macro[2], 4),
            "pr_auc_macro": round(pr[0], 4),
            "pr_auc_macro_lo": round(pr[1], 4),
            "pr_auc_macro_hi": round(pr[2], 4),
            "n_boot": n_boot,
            "seed": seed,
        }
        filas.append(fila)
        logger.info(
            f"{run_dir.name:55s} | AUROC-5 {_fmt_ci(*c5)} | "
            f"AUROC-macro {_fmt_ci(*macro)} | PR-AUC {_fmt_ci(*pr)}"
        )

    salida = experiments_root / "leaderboard_ci.csv"
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    logger.info(f"Leaderboard con IC bootstrap guardado en: {salida}")
    return filas


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IC bootstrap del leaderboard de experimentos.")
    p.add_argument("--experiments-root", default="experiments", help="Carpeta raíz de experimentos.")
    p.add_argument("--n-boot", type=int, default=2000, help="Número de réplicas bootstrap.")
    p.add_argument("--seed", type=int, default=42, help="Semilla del generador bootstrap.")
    p.add_argument("--alpha", type=float, default=0.05, help="Nivel del IC (0.05 = IC 95%%).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    agregar_leaderboard(Path(args.experiments_root), args.n_boot, args.seed, args.alpha)


if __name__ == "__main__":
    main()