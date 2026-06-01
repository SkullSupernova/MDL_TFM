"""
Sistema de seguimiento y trazabilidad de experimentos (file-based).

Cada entrenamiento real genera una carpeta `experiments/<run_id>/` autocontenida con la
configuración, los metadatos de entorno/git, la composición de los conjuntos (incluidas las
clases ausentes/no evaluables), las curvas de aprendizaje, las métricas globales y por clase
de validación y test, las matrices de confusión, las curvas ROC/PR, las predicciones crudas,
el análisis de error y un informe Markdown. Un `leaderboard.csv` agrega todos los runs.

El objetivo es que cada experimento quede documentado y sea reproducible, y que las
limitaciones (p. ej. clases sin positivos en el test) se reflejen explícitamente para evitar
interpretaciones erróneas de las métricas.
"""

import csv
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision
import sklearn
import yaml

from src.logging_config import get_logger
from src.utils import distribucion_clases, contar_parametros
from src.visualization import (
    graficar_entrenamiento,
    plot_confusion_matrices,
    matriz_resumen_multietiqueta,
    plot_roc_curves,
    plot_pr_curves,
)

logger = get_logger(__name__)


def _git_info() -> Dict[str, Optional[object]]:
    """Captura commit, rama y estado 'dirty' del repositorio (None si no es un repo git)."""
    def _run(args):
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_run(["git", "status", "--porcelain"])),
        }
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def _env_info(device: torch.device) -> Dict:
    """Captura versiones de librerías y hardware para reproducibilidad."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "cuda_disponible": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    return info


def _sha256(path: str) -> Optional[str]:
    """SHA-256 de un fichero (identidad/integridad del checkpoint). None si no existe."""
    if not path or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(x: Optional[float]) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"


class ExperimentTracker:
    """Acumula y persiste todos los artefactos de un experimento en experiments/<run_id>/."""

    def __init__(self, cfg: dict, tag: Optional[str] = None, device: Optional[torch.device] = None):
        root = Path(cfg.get("experiments", {}).get("root", "experiments"))
        backbone = cfg["model"]["name"]
        tag_slug = "_" + re.sub(r"[^A-Za-z0-9.-]+", "-", tag) if tag else ""
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{backbone}{tag_slug}"
        self.dir = root / self.run_id
        (self.dir / "predictions").mkdir(parents=True, exist_ok=True)
        (self.dir / "error_analysis").mkdir(parents=True, exist_ok=True)
        (self.dir / "plots").mkdir(parents=True, exist_ok=True)

        self.cfg = cfg
        self.tag = tag
        self.root = root
        self.device = device or torch.device("cpu")
        self.inicio = datetime.now()
        self.env = _env_info(self.device)
        self.git = _git_info()
        self.dataset_info: Dict = {}
        self.metrics: Dict[str, Dict] = {}

        # Snapshot de la configuración efectiva (con overrides CLI ya aplicados).
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        logger.info(f"Experimento iniciado: {self.dir}")

    # ------------------------------------------------------------------
    def registrar_datasets(
        self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
        labels: List[str], etl_reporte: Dict, pos_weight: Dict[str, float], provenance: Dict
    ) -> None:
        """Documenta tamaños, distribución por clase, clases ausentes/no evaluables y ETL."""
        def _no_evaluables(y):
            return [labels[j] for j in range(y.shape[1]) if y[:, j].min() == y[:, j].max()]

        self.dataset_info = {
            "distribucion": {
                "train": distribucion_clases(y_train, labels),
                "val": distribucion_clases(y_val, labels),
                "test": distribucion_clases(y_test, labels),
            },
            # Clases sin ambos valores (0 y 1) → AUROC indefinida en ese conjunto.
            "clases_no_evaluables_auroc": {
                "train": _no_evaluables(y_train),
                "val": _no_evaluables(y_val),
                "test": _no_evaluables(y_test),
            },
            "etl_reporte": etl_reporte,
            "pos_weight": pos_weight,
            "provenance": provenance,
        }
        with open(self.dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(self.dataset_info, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    def registrar_historial(self, history: Dict[str, list]) -> None:
        """Guarda las curvas de aprendizaje (history.csv) y la gráfica de entrenamiento."""
        self.history = history
        n = len(history.get("train_loss", []))
        campos = ["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_auroc"]
        with open(self.dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(campos)
            for i in range(n):
                writer.writerow([i + 1] + [history.get(k, [None] * n)[i] for k in campos[1:]])
        try:
            graficar_entrenamiento(history, save_path=str(self.dir / "plots" / "learning_curves.png"))
        except Exception as e:
            logger.warning(f"No se pudo generar learning_curves.png: {e}")

    # ------------------------------------------------------------------
    def registrar_evaluacion(
        self, split: str, metrics: Dict, y_true: np.ndarray, y_pred: np.ndarray,
        y_prob: np.ndarray, labels: List[str], rutas: Optional[List[str]] = None
    ) -> None:
        """Persiste métricas, predicciones, análisis de error y gráficas de un split."""
        self.metrics[split] = metrics

        with open(self.dir / f"metrics_{split}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        self._csv_por_clase(split, metrics, labels)

        np.savez_compressed(
            self.dir / "predictions" / f"{split}_predictions.npz",
            y_true=y_true, y_prob=y_prob,
            rutas=np.array(rutas if rutas is not None else [], dtype=object),
        )

        n_worst = int(self.cfg.get("experiments", {}).get("n_worst_cases", 20))
        self._csv_worst_cases(split, y_true, y_pred, y_prob, labels, rutas, n_worst)

        # Gráficas: matrices de confusión siempre; ROC/PR solo en test (informe principal).
        try:
            plot_confusion_matrices(
                y_true, y_pred, labels,
                save_path=str(self.dir / "plots" / f"confusion_matrices_{split}.png"),
            )
            if split == "test":
                matriz_resumen_multietiqueta(
                    y_true, y_pred, labels,
                    save_path=str(self.dir / "plots" / "clinical_summary_test.png"),
                )
                plot_roc_curves(y_true, y_prob, labels, save_path=str(self.dir / "plots" / "roc_curves_test.png"))
                plot_pr_curves(y_true, y_prob, labels, save_path=str(self.dir / "plots" / "pr_curves_test.png"))
        except Exception as e:
            logger.warning(f"No se pudieron generar las gráficas de {split}: {e}")

    def _csv_por_clase(self, split: str, metrics: Dict, labels: List[str]) -> None:
        out = self.dir / f"metrics_{split}_per_class.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["patologia", "soporte", "auroc", "pr_auc", "precision", "recall", "f1", "evaluable"])
            for lab in labels:
                m = metrics["per_class"][lab]
                evaluable = lab not in metrics["clases_no_evaluables"]
                w.writerow([
                    lab, m["soporte"],
                    "" if m["auroc"] is None else round(m["auroc"], 6),
                    "" if m["pr_auc"] is None else round(m["pr_auc"], 6),
                    round(m["precision"], 6), round(m["recall"], 6), round(m["f1"], 6), evaluable,
                ])

    def _csv_worst_cases(self, split, y_true, y_pred, y_prob, labels, rutas, n) -> None:
        filas = []
        for i, lab in enumerate(labels):
            t, p, pr = y_true[:, i], y_pred[:, i], y_prob[:, i]
            # FP más confiados (prob alta pese a ser negativo real) y FN más graves
            # (prob baja pese a ser positivo real): los casos más útiles para depurar.
            fp = np.where((p == 1) & (t == 0))[0]
            fp = fp[np.argsort(-pr[fp])][:n]
            fn = np.where((p == 0) & (t == 1))[0]
            fn = fn[np.argsort(pr[fn])][:n]
            for tipo, idxs in (("FP", fp), ("FN", fn)):
                for idx in idxs:
                    filas.append({
                        "clase": lab, "tipo": tipo,
                        "prob": round(float(pr[idx]), 4),
                        "etiqueta_real": int(t[idx]),
                        "ruta": rutas[idx] if rutas is not None and idx < len(rutas) else "",
                    })
        out = self.dir / "error_analysis" / f"{split}_worst_cases.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["clase", "tipo", "prob", "etiqueta_real", "ruta"])
            w.writeheader()
            w.writerows(filas)

    # ------------------------------------------------------------------
    def finalizar(self, info_modelo: Dict, info_entrenamiento: Dict, promocion: Dict) -> str:
        """Escribe manifest.json y report.md, y añade la fila al leaderboard. Devuelve run_id."""
        fin = datetime.now()
        info_modelo = dict(info_modelo)
        info_modelo["checkpoint_sha256"] = _sha256(info_modelo.get("checkpoint_path"))

        self.manifest = {
            "run_id": self.run_id,
            "tag": self.tag,
            "inicio": self.inicio.isoformat(timespec="seconds"),
            "fin": fin.isoformat(timespec="seconds"),
            "duracion_segundos": round((fin - self.inicio).total_seconds(), 1),
            "git": self.git,
            "entorno": self.env,
            "modelo": info_modelo,
            "entrenamiento": info_entrenamiento,
            "promocion": promocion,
        }
        with open(self.dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

        self._escribir_report()
        self._append_leaderboard()
        logger.info(f"Experimento finalizado y documentado en: {self.dir}")
        return self.run_id

    def _append_leaderboard(self) -> None:
        test = self.metrics.get("test", {})
        ent = self.manifest["entrenamiento"]
        fila = {
            "run_id": self.run_id,
            "timestamp": self.manifest["inicio"],
            "backbone": self.cfg["model"]["name"],
            "tag": self.tag or "",
            "epochs_ejecutadas": ent.get("epochs_ejecutadas"),
            "lr": ent.get("learning_rate"),
            "batch_size": ent.get("batch_size"),
            "seed": ent.get("seed"),
            "val_auroc_best": round(ent.get("mejor_epoca_val_auroc"), 4) if ent.get("mejor_epoca_val_auroc") is not None else "",
            "test_auroc_chexpert5": "" if test.get("auroc_chexpert5") is None else round(test["auroc_chexpert5"], 4),
            "test_auroc_macro": round(test.get("auroc_macro_evaluable", 0.0), 4),
            "test_pr_auc_macro": round(test.get("pr_auc_macro_evaluable", 0.0), 4),
            "test_f1_macro": round(test.get("f1_macro", 0.0), 4),
            "duracion_min": round(self.manifest["duracion_segundos"] / 60, 2),
            "promovido": self.manifest["promocion"].get("promovido"),
            "git_commit": (self.git.get("commit") or "")[:8],
        }
        lb = self.root / "leaderboard.csv"
        existe = lb.exists()
        with open(lb, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fila.keys()))
            if not existe:
                w.writeheader()
            w.writerow(fila)

    def _escribir_report(self) -> None:
        m = self.manifest
        di = self.dataset_info
        lineas = [f"# Informe de experimento — {self.run_id}", ""]

        # Resumen ejecutivo
        prom = m["promocion"]
        test = self.metrics.get("test", {})
        lineas += [
            "## Resumen ejecutivo", "",
            f"- **Backbone:** {m['modelo'].get('backbone')} ({m['modelo'].get('num_classes')} clases)",
            f"- **AUROC CheXpert-5 (test):** {_fmt(test.get('auroc_chexpert5'))} · "
            f"**PR-AUC-macro:** {_fmt(test.get('pr_auc_macro_evaluable'))} · "
            f"**F1-macro:** {_fmt(test.get('f1_macro'))}",
            f"- **Promovido a producción:** {prom.get('promovido')}",
            f"- **Duración:** {m['duracion_segundos']} s · **Mejor época:** {m['entrenamiento'].get('mejor_epoca')}",
            f"- **Git:** {(self.git.get('commit') or 'n/a')[:8]} (dirty={self.git.get('dirty')}) · "
            f"**Device:** {self.env.get('device')}", "",
        ]

        # Reproducibilidad / config
        ent = m["entrenamiento"]
        lineas += [
            "## Configuración e hiperparámetros", "",
            "| Parámetro | Valor |", "|---|---|",
            f"| optimizer | {ent.get('optimizer')} |",
            f"| learning_rate | {ent.get('learning_rate')} |",
            f"| weight_decay | {ent.get('weight_decay')} |",
            f"| batch_size | {ent.get('batch_size')} |",
            f"| epochs (máx/ejec) | {ent.get('epochs_max')} / {ent.get('epochs_ejecutadas')} |",
            f"| scheduler | {ent.get('scheduler')} |",
            f"| seed | {ent.get('seed')} |",
            f"| AMP | {ent.get('amp')} |",
            f"| params (total/entrenables) | {m['modelo'].get('params', {}).get('total')} / {m['modelo'].get('params', {}).get('entrenables')} |",
            f"| entorno | torch {self.env.get('torch')}, cuda={self.env.get('cuda_disponible')} |", "",
        ]

        # Datasets + clases ausentes
        if di:
            lineas += ["## Conjuntos de datos", "", "| Conjunto | Nº muestras | Clases ausentes |", "|---|---|---|"]
            for s in ("train", "val", "test"):
                d = di["distribucion"][s]
                aus = ", ".join(d["clases_ausentes"]) or "—"
                lineas.append(f"| {s} | {d['n_muestras']} | {aus} |")
            lineas += ["", "### Distribución por clase (positivos)", "",
                       "| Patología | train | val | test |", "|---|---|---|---|"]
            for lab in di["distribucion"]["train"]["por_clase"]:
                row = [lab]
                for s in ("train", "val", "test"):
                    c = di["distribucion"][s]["por_clase"][lab]
                    row.append(f"{c['positivos']} ({c['porcentaje']}%)")
                lineas.append("| " + " | ".join(row) + " |")
            lineas += ["",
                       "> **Clases no evaluables (sin positivos/negativos) en test:** "
                       + (", ".join(di["clases_no_evaluables_auroc"]["test"]) or "ninguna")
                       + ". Sus AUROC/PR-AUC/F1 no son válidas y se omiten de los promedios.", ""]

        # Métricas por split
        for split in ("val", "test"):
            mt = self.metrics.get(split)
            if not mt:
                continue
            lineas += [
                f"## Métricas — {split}", "",
                f"- n={mt['n_muestras']} · accuracy={_fmt(mt['accuracy'])} · F1-macro={_fmt(mt['f1_macro'])} "
                f"· F1-micro={_fmt(mt['f1_micro'])} · AUROC-macro={_fmt(mt['auroc_macro_evaluable'])} "
                f"· PR-AUC-macro={_fmt(mt['pr_auc_macro_evaluable'])}", "",
                "| Patología | Soporte | AUROC | PR-AUC | Precision | Recall | F1 |",
                "|---|---|---|---|---|---|---|",
            ]
            for lab, c in mt["per_class"].items():
                marca = " ⚠️" if lab in mt["clases_no_evaluables"] else ""
                lineas.append(
                    f"| {lab}{marca} | {c['soporte']} | {_fmt(c['auroc'])} | {_fmt(c['pr_auc'])} | "
                    f"{c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |"
                )
            lineas.append("")

        # Gráficas
        lineas += ["## Gráficas", ""]
        for nombre in ["learning_curves.png", "confusion_matrices_test.png",
                       "roc_curves_test.png", "pr_curves_test.png", "clinical_summary_test.png"]:
            if (self.dir / "plots" / nombre).exists():
                lineas.append(f"![{nombre}](plots/{nombre})")
        lineas += ["", "## Artefactos", "",
                   "- Predicciones crudas: `predictions/`  ·  Análisis de error: `error_analysis/`",
                   "- Métricas: `metrics_val.json`, `metrics_test.json` (+ CSV por clase)",
                   "- Configuración: `config.yaml`  ·  Metadatos: `manifest.json`  ·  Datos: `dataset.json`", ""]

        with open(self.dir / "report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))
