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

    # El checkpoint se guarda con el nombre del backbone para no sobreescribir
    # modelos de otras arquitecturas entrenados en la misma máquina.
    save_path = f"models/mejor_modelo_{model_name}.pth"
    logger.info(f"Backbone: {model_name} — checkpoint: {save_path}")

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


if __name__ == "__main__":
    main()
