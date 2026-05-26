import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models import CheXpertDataset, build_model, CHEXPERT_PATHOLOGY_COLS
from src.utils import (
    setup_environment,
    set_seed,
    aplicar_filtrado_proyecto,
    obtener_ruta_absoluta_train,
)
from src.train import train_model
from src.visualization import graficar_entrenamiento


def load_config(path: str = "config/config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _patient_split(df: pd.DataFrame, train_ratio: float, seed: int):
    """Split por paciente único para evitar data leakage entre train y validación."""
    rng = np.random.default_rng(seed)
    pacientes = df["Patient"].unique()
    rng.shuffle(pacientes)
    n_train = int(len(pacientes) * train_ratio)
    train_pats = set(pacientes[:n_train])
    train_idx = df.index[df["Patient"].isin(train_pats)].tolist()
    val_idx = df.index[~df["Patient"].isin(train_pats)].tolist()
    return train_idx, val_idx


def main():
    cfg = load_config()
    device, num_workers = setup_environment()
    set_seed(cfg["training"]["seed"])

    # --- ETL ---
    df = pd.read_csv(cfg["data"]["csv_path"])
    df, _ = aplicar_filtrado_proyecto(df)

    # Extraer ID de paciente desde el campo Path (formato: .../patientXXXXX/...)
    df["Patient"] = df["Path"].str.split("/").str[2]

    df["Ruta_Absoluta"] = df["Path"].apply(
        lambda p: obtener_ruta_absoluta_train(
            p, cfg["data"]["images_root"], cfg["data"]["batches"]
        )
    )
    df = df[df["Ruta_Absoluta"].notna()].reset_index(drop=True)
    df[CHEXPERT_PATHOLOGY_COLS] = df[CHEXPERT_PATHOLOGY_COLS].fillna(0.0)

    # --- Transforms ---
    img_size = cfg["data"]["img_size"]
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

    # --- Split por paciente (evita data leakage) ---
    train_idx, val_idx = _patient_split(
        df, cfg["data"]["train_split"], cfg["training"]["seed"]
    )
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
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["inference_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Modelo (backbone intercambiable via config.yml) ---
    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        hidden_units=cfg["model"]["hidden_units"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # --- Class weights: mayor gradiente para patologías poco frecuentes ---
    labels = df.loc[train_idx, CHEXPERT_PATHOLOGY_COLS].values
    neg = (labels == 0).sum(axis=0)
    pos = (labels == 1).sum(axis=0) + 1e-6
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

    # --- Entrenamiento ---
    history, model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=cfg["training"]["epochs"],
        device=device,
        save_path=cfg["model"]["checkpoint_path"],
    )
    graficar_entrenamiento(history)


if __name__ == "__main__":
    main()
