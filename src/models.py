# src/models.py
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models as tv_models
from PIL import Image


# =========================================================
# Constante de referencia: patologías CheXpert
# =========================================================

CHEXPERT_PATHOLOGY_COLS: List[str] = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Fracture', 'Support Devices'
]


# =========================================================
# Dataset de PyTorch para CheXpert
# =========================================================

class CheXpertDataset(Dataset):
    """
    Dataset de PyTorch para el dataset CheXpert.

    Carga imágenes en formato RGB desde rutas absolutas almacenadas en la
    columna 'Ruta_Absoluta' del DataFrame, y devuelve tensores de imagen
    junto con vectores de etiquetas multietiqueta en float32.

    Parámetros
    ----------
    dataframe : pd.DataFrame
        DataFrame con al menos las columnas 'Ruta_Absoluta' y las columnas
        de patologías especificadas en etiquetas_cols.
    transform : callable, optional
        Transformaciones de torchvision a aplicar sobre la imagen.
    etiquetas_cols : list of str, optional
        Columnas de patologías a incluir como etiquetas. Si no se especifica,
        se utiliza CHEXPERT_PATHOLOGY_COLS.
    """

    def __init__(
        self,
        dataframe,
        transform=None,
        etiquetas_cols: Optional[List[str]] = None
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.etiquetas_cols = etiquetas_cols if etiquetas_cols is not None else CHEXPERT_PATHOLOGY_COLS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # Localización física del archivo
        img_path = self.df.loc[idx, 'Ruta_Absoluta']

        # Carga y conversión a RGB.
        # Las imágenes originales son en escala de grises (1 canal). Se fuerza
        # la conversión a RGB (3 canales idénticos) porque las arquitecturas
        # preentrenadas en ImageNet (DenseNet, ResNet) exigen tensores [3, H, W].
        image = Image.open(img_path).convert('RGB')

        # Aplicación de transformaciones y normalización
        if self.transform:
            image = self.transform(image)

        # Vectorización de las etiquetas clínicas
        labels = self.df.loc[idx, self.etiquetas_cols].values.astype(np.float32)
        labels = torch.tensor(labels)

        return image, labels


# =========================================================
# Builder genérico de modelos
# =========================================================

SUPPORTED_MODELS = ("densenet121", "resnet50", "efficientnet_b0", "efficientnet_b4")


def _classifier_head(
    in_features: int, hidden_units: int, dropout: float, num_classes: int
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, num_classes),
    )


def build_model(
    model_name: str = "densenet121",
    num_classes: int = 13,
    dropout: float = 0.5,
    hidden_units: int = 1024,
    pretrained: bool = True,
) -> nn.Module:
    """Generic multilabel classifier builder.
    Supported backbones: densenet121, resnet50, efficientnet_b0, efficientnet_b4."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"'{model_name}' no soportado. Elige entre {SUPPORTED_MODELS}")

    weights = "DEFAULT" if pretrained else None

    if model_name == "densenet121":
        base = tv_models.densenet121(weights=weights)
        base.classifier = _classifier_head(
            base.classifier.in_features, hidden_units, dropout, num_classes
        )
    elif model_name == "resnet50":
        base = tv_models.resnet50(weights=weights)
        base.fc = _classifier_head(
            base.fc.in_features, hidden_units, dropout, num_classes
        )
    elif model_name.startswith("efficientnet"):
        base = getattr(tv_models, model_name)(weights=weights)
        base.classifier[-1] = _classifier_head(
            base.classifier[-1].in_features, hidden_units, dropout, num_classes
        )

    return base


def _has_simple_head(state: dict, model_name: str) -> bool:
    """Return True if the checkpoint uses a plain nn.Linear output head."""
    if model_name == "densenet121":
        return "classifier.weight" in state and "classifier.0.weight" not in state
    if model_name == "resnet50":
        return "fc.weight" in state and "fc.0.weight" not in state
    if model_name.startswith("efficientnet"):
        return "classifier.1.weight" in state and "classifier.1.0.weight" not in state
    return False


def _build_simple_head_model(model_name: str, num_classes: int) -> nn.Module:
    """Build backbone + single nn.Linear — matches checkpoints trained without _classifier_head."""
    if model_name == "densenet121":
        base = tv_models.densenet121(weights=None)
        base.classifier = nn.Linear(base.classifier.in_features, num_classes)
    elif model_name == "resnet50":
        base = tv_models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        base = getattr(tv_models, model_name)(weights=None)
        base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"'{model_name}' no soportado")
    return base


def load_checkpoint(cfg: dict, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a checkpoint into the correct architecture and return an eval-mode model.

    Detects automatically whether the checkpoint was saved with a plain
    nn.Linear head (original notebook training) or with _classifier_head
    (build_model), and builds the matching architecture.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]

    if _has_simple_head(state, model_name):
        model = _build_simple_head_model(model_name, num_classes)
    else:
        model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            dropout=cfg["model"]["dropout"],
            hidden_units=cfg["model"]["hidden_units"],
            pretrained=False,
        )

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_grad_cam_layer(model: nn.Module, model_name: str) -> list:
    """Devuelve la capa target de GradCAM para cada backbone soportado."""
    if model_name == "densenet121":
        return [model.features[-1]]
    if model_name == "resnet50":
        return [model.layer4[-1]]
    if model_name.startswith("efficientnet"):
        return [model.features[-1]]
    raise ValueError(f"No hay capa GradCAM definida para '{model_name}'")