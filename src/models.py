# src/models.py
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# =========================================================
# Constante de referencia: patologías CheXpert
# =========================================================

CHEXPERT_PATHOLOGY_COLS: List[str] = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
]
"""
Lista canónica de las 14 etiquetas diagnósticas del dataset CheXpert.
Se utiliza como valor por defecto en CheXpertDataset cuando no se especifican
columnas de etiquetas explícitamente.
"""


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