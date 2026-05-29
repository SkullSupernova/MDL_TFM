"""
Pipeline CLI de entrenamiento para clasificación multietiqueta de patologías torácicas.

Punto de entrada principal del proyecto. Orquesta la carga de configuración,
el preprocesado ETL del dataset CheXpert, el split por paciente para evitar
data leakage, la construcción del modelo y el bucle de entrenamiento.

Uso:
    python -m src.main
    python -m src.main --model resnet50 --epochs 5 --subset 1000
"""

import argparse
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.logging_config import get_logger
from src.models import CheXpertDataset, build_model, CHEXPERT_PATHOLOGY_COLS
from src.utils import (
    setup_environment,
    set_seed,
    aplicar_filtrado_proyecto,
    obtener_ruta_absoluta_train,
)
from src.train import train_model
from src.visualization import graficar_entrenamiento
from src.evaluate import evaluar_test
from src.model_registry import (
    cargar_registro,
    es_mejor,
    guardar_registro,
    registrar_experimento,
    promover,
    descartar,
    RUTA_HISTORIAL,
)

logger = get_logger(__name__)


def load_config(path: str = "config/config.yml") -> dict:
    """Carga y devuelve el archivo de configuración YAML del proyecto."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.

    Los argumentos opcionales sobreescriben los valores de config.yml en tiempo
    de ejecución sin modificar el fichero de configuración.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento CheXpert"
    )
    parser.add_argument(
        "--config", default="config/config.yml",
        help="Ruta al archivo de configuración YAML (por defecto: config/config.yml)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Nombre del backbone a usar. Sobreescribe model.name de config.yml."
             " Soportados: densenet121, resnet50, efficientnet_b0, efficientnet_b4"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Número máximo de épocas. Sobreescribe training.epochs de config.yml."
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limitar el conjunto de entrenamiento a N imágenes. Útil para validación rápida."
    )
    parser.add_argument(
        "--val-subset", type=int, default=None,
        help="Limitar el conjunto de validación a N imágenes. Para smoke tests rápidos de extremo a extremo."
    )
    return parser.parse_args()


def _patient_split(df: pd.DataFrame, train_ratio: float, seed: int):
    """
    Divide el dataset en train/validación a nivel de paciente.

    Garantiza que todas las imágenes de un mismo paciente pertenezcan
    exclusivamente a un subconjunto, evitando data leakage.

    El data leakage ocurre cuando imágenes del mismo paciente aparecen tanto
    en train como en validación. La red aprende variaciones de estilo individual
    (posición, artifacts, parámetros de adquisición) en lugar de patrones clínicos
    generalizables, produciendo una estimación optimista del rendimiento real.

    Devuelve dos listas de índices de filas del DataFrame.
    """
    rng = np.random.default_rng(seed)
    pacientes = df["Patient"].unique()
    rng.shuffle(pacientes)
    n_train = int(len(pacientes) * train_ratio)
    train_pats = set(pacientes[:n_train])
    train_idx = df.index[df["Patient"].isin(train_pats)].tolist()
    val_idx = df.index[~df["Patient"].isin(train_pats)].tolist()
    return train_idx, val_idx


def _fmt(x: Optional[float]) -> str:
    """Formatea una métrica que puede ser None (clase/conjunto no evaluable)."""
    return f"{x:.4f}" if x is not None else "n/a"


def _gestionar_promocion(
    cfg: dict, model_name: str, candidato_path: str, produccion_path: str,
    history: dict, test_metrics: dict
) -> None:
    """
    Decide si el modelo recién entrenado reemplaza al mejor modelo en producción.

    Compara la métrica de promoción (AUROC CheXpert-5 sobre el test silver, con F1-macro
    de desempate) contra el campeón registrado. Promueve el candidato a producción solo
    si mejora por encima de 'promotion_min_delta'; en caso contrario conserva el modelo
    actual y descarta el candidato. Registra siempre el experimento para auditoría.
    """
    # El modelo devuelto por train_model corresponde al epoch de mayor AUROC de
    # validación; sus métricas de validación se leen de ese epoch del historial.
    best_idx = int(np.argmax(history["val_auroc"]))
    val_metrics = {
        "auroc_macro": history["val_auroc"][best_idx],
        "f1_macro": history["val_f1"][best_idx],
        "accuracy": history["val_acc"][best_idx],
    }

    campeon = cargar_registro(model_name)
    # cargar_registro devuelve el registro completo; es_mejor compara solo las
    # métricas de test (donde está auroc_chexpert5 / f1_macro).
    campeon_test = campeon["test_metrics"] if campeon else None
    min_delta = cfg["training"].get("promotion_min_delta", 0.0)
    promovido = es_mejor(test_metrics, campeon_test, min_delta)

    registro = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backbone": model_name,
        "hiperparametros": {
            "epochs": cfg["training"]["epochs"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["batch_size"],
            "weight_decay": cfg["training"]["weight_decay"],
            "seed": cfg["training"]["seed"],
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": produccion_path,
        "promovido": promovido,
    }

    if promovido:
        promover(candidato_path, produccion_path)
        guardar_registro(model_name, registro)
        if campeon is None:
            logger.info(
                f"NUEVO MEJOR MODELO (primer registro) — AUROC CheXpert-5 (test): "
                f"{_fmt(test_metrics['auroc_chexpert5'])} | F1-macro: {test_metrics['f1_macro']:.4f}. "
                f"Guardado en {produccion_path}."
            )
        else:
            logger.info(
                f"NUEVO MEJOR MODELO — AUROC CheXpert-5 (test): "
                f"{_fmt(campeon['test_metrics'].get('auroc_chexpert5'))} -> "
                f"{_fmt(test_metrics['auroc_chexpert5'])} | F1-macro: "
                f"{campeon['test_metrics'].get('f1_macro', 0.0):.4f} -> {test_metrics['f1_macro']:.4f}. "
                f"Promovido a {produccion_path}."
            )
    else:
        descartar(candidato_path)
        logger.info(
            f"El nuevo modelo NO supera al actual (AUROC CheXpert-5 test "
            f"{_fmt(test_metrics['auroc_chexpert5'])} vs "
            f"{_fmt(campeon['test_metrics'].get('auroc_chexpert5'))}, min_delta={min_delta}). "
            f"Se conserva {produccion_path}; candidato descartado."
        )

    registrar_experimento(registro)
    logger.info(f"Experimento registrado en {RUTA_HISTORIAL}.")


def main():
    args = _parse_args()
    cfg = load_config(args.config)

    # Aplicar sobreescrituras desde CLI antes de cualquier uso de cfg.
    # Orden de precedencia: CLI > config.yml.
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    logger.info(
        f"Configuración cargada: backbone={cfg['model']['name']}, "
        f"épocas={cfg['training']['epochs']}, batch={cfg['training']['batch_size']}"
    )

    device, num_workers = setup_environment()
    set_seed(cfg["training"]["seed"])

    # ==================================================================
    # ETL
    # ==================================================================
    df = pd.read_csv(cfg["data"]["csv_path"])
    logger.info(f"CSV cargado: {len(df)} estudios en bruto")

    df, _ = aplicar_filtrado_proyecto(df)
    logger.info(f"Dataset tras ETL: {len(df)} estudios retenidos")

    # Extraer ID de paciente desde el campo Path (formato: .../patientXXXXX/...).
    # El ID se usa para el split: todas las imágenes de un paciente van al mismo subconjunto.
    df["Patient"] = df["Path"].str.split("/").str[2]

    df["Ruta_Absoluta"] = df["Path"].apply(
        lambda p: obtener_ruta_absoluta_train(
            p, cfg["data"]["images_root"], cfg["data"]["batches"]
        )
    )
    # Descartar estudios cuya imagen no se encuentra en disco (batch no descargado).
    df = df[df["Ruta_Absoluta"].notna()].reset_index(drop=True)
    logger.info(f"Imágenes localizadas en disco: {len(df)}")

    # Imputar NaN en columnas de patología a 0.0.
    # NaN en CheXpert significa que el radiólogo no evaluó esa patología,
    # no que esté ausente. Se trata como negativo (0.0) porque el modelo
    # necesita etiquetas binarias y la ausencia de evaluación es la señal
    # más débil posible, comparable a un negativo implícito.
    df[CHEXPERT_PATHOLOGY_COLS] = df[CHEXPERT_PATHOLOGY_COLS].fillna(0.0)

    # ==================================================================
    # Transforms
    # ==================================================================
    img_size = cfg["data"]["img_size"]

    # Data augmentation solo en entrenamiento, no en validación.
    # En validación se quiere una estimación determinista del rendimiento:
    # aplicar augmentation produciría métricas ruidosas que variarían entre
    # ejecuciones aunque el modelo sea el mismo.
    # RandomHorizontalFlip y RandomAffine simulan variabilidad natural de
    # posición del paciente y parámetros de adquisición sin introducir
    # artefactos clínicamente incorrectos (no rotaciones extremas, no zoom).
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ==================================================================
    # Split por paciente (evita data leakage)
    # ==================================================================
    train_idx, val_idx = _patient_split(
        df, cfg["data"]["train_split"], cfg["training"]["seed"]
    )
    logger.info(
        f"Split por paciente: {len(train_idx)} entrenamiento / {len(val_idx)} validación"
    )

    # Subconjunto opcional para validación rápida del pipeline completo
    # sin ejecutar un entrenamiento real sobre el dataset completo.
    if args.subset:
        train_idx = train_idx[:args.subset]
        logger.info(f"Subconjunto aplicado: usando {len(train_idx)} imágenes de entrenamiento")

    # El recorte de validación es independiente del de entrenamiento: la fase de
    # validación recorre todo val_idx por defecto, lo que en un smoke test domina el
    # tiempo total. Limitarla permite probar el flujo de extremo a extremo en segundos.
    if args.val_subset:
        val_idx = val_idx[:args.val_subset]
        logger.info(f"Subconjunto de validación aplicado: usando {len(val_idx)} imágenes de validación")

    train_ds = CheXpertDataset(
        df.loc[train_idx].reset_index(drop=True),
        transform=train_tf,
        etiquetas_cols=CHEXPERT_PATHOLOGY_COLS,
    )
    val_ds = CheXpertDataset(
        df.loc[val_idx].reset_index(drop=True),
        transform=eval_tf,
        etiquetas_cols=CHEXPERT_PATHOLOGY_COLS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True mantiene los tensores en memoria paginada bloqueada (page-locked).
        # En sistemas con GPU, esto acelera la transferencia CPU→GPU porque el DMA
        # puede leer directamente desde esa memoria sin necesidad de copias intermedias.
        # En CPU-only no hay beneficio apreciable pero tampoco penalización.
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["inference_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ==================================================================
    # Modelo
    # ==================================================================
    model_name = cfg["model"]["name"]
    model = build_model(
        model_name=model_name,
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        hidden_units=cfg["model"]["hidden_units"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # ==================================================================
    # Función de pérdida con pesos de clase (pos_weight)
    # ==================================================================
    # El dataset CheXpert está fuertemente desbalanceado: algunas patologías
    # aparecen en menos del 5% de los estudios. Sin corrección, el modelo
    # aprende a predecir siempre el negativo y obtiene alta accuracy ignorando
    # las clases minoritarias.
    # pos_weight = neg / pos escala el gradiente de los positivos para que
    # el modelo les preste tanta atención como a los negativos.
    # BCEWithLogitsLoss acepta pos_weight directamente: multiplica la pérdida
    # de los positivos de cada clase por su peso correspondiente.
    labels = df.loc[train_idx, CHEXPERT_PATHOLOGY_COLS].values
    neg = (labels == 0).sum(axis=0)
    pos = (labels == 1).sum(axis=0) + 1e-6   # +1e-6 evita división por cero en clases sin positivos
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg["training"]["scheduler_patience"]
    )

    # El checkpoint se nombra con el backbone para no mezclar arquitecturas distintas.
    # Las ejecuciones de prueba (--subset / --val-subset) guardan con sufijo '_subset' y
    # nunca tocan producción. Los entrenamientos reales guardan primero un checkpoint
    # CANDIDATO y solo se promueve a producción si supera al mejor modelo registrado.
    produccion_path = f"models/mejor_modelo_{model_name}.pth"
    es_prueba = bool(args.subset or args.val_subset)
    if es_prueba:
        save_path = f"models/mejor_modelo_{model_name}_subset.pth"
        logger.info("Ejecución de prueba: no se promociona ni se sobrescribe el modelo de producción.")
    else:
        save_path = f"models/_candidato_{model_name}.pth"
    logger.info(f"Backbone: {model_name} — checkpoint de este run: {save_path}")

    # ==================================================================
    # Entrenamiento
    # ==================================================================
    history, model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=cfg["training"]["epochs"],
        device=device,
        save_path=save_path,
    )
    graficar_entrenamiento(history)

    # ==================================================================
    # Evaluación final sobre el test silver-standard (valid de Stanford)
    # ==================================================================
    # El test (valid oficial anotado por radiólogos) es independiente del train
    # (pacientes disjuntos) y no participa en la selección del epoch —hecha por
    # AUROC de validación—. Sus métricas se usan para el gate de promoción.
    test_metrics = evaluar_test(cfg, model, device, cfg["model"]["num_classes"])

    # ==================================================================
    # Gate de promoción del mejor modelo (solo en entrenamientos reales)
    # ==================================================================
    if es_prueba:
        logger.info("Ejecución de prueba: omitido el gate de promoción del mejor modelo.")
    else:
        _gestionar_promocion(cfg, model_name, save_path, produccion_path, history, test_metrics)


if __name__ == "__main__":
    main()
