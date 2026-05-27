# src/models.py
#
# Este módulo centraliza todo lo relacionado con los modelos de red neuronal:
#   1. Las listas de patologías que el modelo puede detectar.
#   2. La clase Dataset que lee las imágenes y sus etiquetas del disco.
#   3. Las funciones que construyen la arquitectura de la red.
#   4. Las funciones que cargan un modelo ya entrenado desde un archivo .pth.
#   5. La función que selecciona la capa correcta para GradCAM según el backbone.
#
# El diseño permite cambiar el backbone (DenseNet, ResNet, EfficientNet) editando
# solo config.yml, sin tocar ningún archivo de código.

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models as tv_models
from PIL import Image


# ==============================================================================
# SECCIÓN 1: LISTAS DE PATOLOGÍAS
# ==============================================================================

# Lista de 13 patologías activas en este proyecto.
# Se excluye 'Pleural Other' porque tiene muy baja prevalencia en el dataset
# de entrenamiento y no aporta señal estadística suficiente para ser aprendida.
# 'No Finding' se incluye para que el modelo pueda indicar explícitamente cuando
# no detecta ninguna anomalía (clase negativa global).
CHEXPERT_PATHOLOGY_COLS: List[str] = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Fracture', 'Support Devices'
]

# Lista completa de 14 patologías según el estándar original de CheXpert.
# Se mantiene para compatibilidad con checkpoints entrenados desde el notebook
# original (T02_Analisis_DenseNet121.ipynb), que sí incluía 'Pleural Other'.
CHEXPERT_PATHOLOGY_COLS_14: List[str] = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


def get_pathology_labels(num_classes: int) -> List[str]:
    """Return the pathology label list matching the number of model output classes."""
    # Seleccionamos la lista correcta en función del número de salidas del modelo.
    # Esto evita que un modelo de 14 clases etiquete sus predicciones con los nombres
    # de 13 clases (lo que produciría una asignación desplazada y resultados erróneos).
    if num_classes == 13:
        return CHEXPERT_PATHOLOGY_COLS
    if num_classes == 14:
        return CHEXPERT_PATHOLOGY_COLS_14
    raise ValueError(f"No hay lista de etiquetas definida para {num_classes} clases")


# ==============================================================================
# SECCIÓN 2: DATASET DE PYTORCH
# ==============================================================================

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
        # reset_index asegura que los índices sean 0, 1, 2... tras cualquier
        # filtrado previo del DataFrame. Sin esto, self.df.loc[idx] podría
        # fallar si los índices originales tienen huecos.
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.etiquetas_cols = etiquetas_cols if etiquetas_cols is not None else CHEXPERT_PATHOLOGY_COLS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # 1. Obtener la ruta física de la imagen en el disco.
        #    La columna 'Ruta_Absoluta' fue construida en utils.py a partir de
        #    la ruta relativa del CSV y el directorio raíz del dataset.
        img_path = self.df.loc[idx, 'Ruta_Absoluta']

        # 2. Cargar la imagen y convertirla a RGB.
        #    Las radiografías de CheXpert son originalmente en escala de grises
        #    (1 solo canal de color). Las redes preentrenadas en ImageNet (DenseNet,
        #    ResNet, EfficientNet) esperan tensores de 3 canales [3, H, W].
        #    convert('RGB') replica el canal de grises 3 veces sin perder información.
        image = Image.open(img_path).convert('RGB')

        # 3. Aplicar transformaciones (resize, normalización, data augmentation).
        #    Las transformaciones son distintas en train y validación: en train se
        #    aplican flips y rotaciones aleatorias para aumentar la variabilidad del
        #    conjunto de datos y reducir el sobreajuste.
        if self.transform:
            image = self.transform(image)

        # 4. Extraer el vector de etiquetas de las columnas de patologías.
        #    Cada columna tiene valor 0.0 (ausente) o 1.0 (presente).
        #    Se convierte a float32 porque BCEWithLogitsLoss espera ese tipo.
        labels = self.df.loc[idx, self.etiquetas_cols].values.astype(np.float32)
        labels = torch.tensor(labels)

        return image, labels


# ==============================================================================
# SECCIÓN 3: CONSTRUCCIÓN DE MODELOS
# ==============================================================================

# Backbones soportados. Añadir uno nuevo requiere un bloque elif en build_model()
# y una entrada en get_grad_cam_layer().
SUPPORTED_MODELS = ("densenet121", "resnet50", "efficientnet_b0", "efficientnet_b4")


def _classifier_head(
    in_features: int, hidden_units: int, dropout: float, num_classes: int
) -> nn.Module:
    # Cabeza de clasificación personalizada que reemplaza la capa final del backbone.
    # La arquitectura Linear → ReLU → Dropout → Linear tiene dos ventajas frente
    # a un único Linear:
    #   - La capa oculta permite aprender representaciones no lineales de alto nivel.
    #   - El Dropout actúa como regularizador, reduciendo el sobreajuste en datasets
    #     de imágenes médicas donde el número de muestras es limitado.
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),   # compresión: features → representación compacta
        nn.ReLU(),                               # no-linealidad: permite aprender patrones complejos
        nn.Dropout(dropout),                     # regularización: apaga neuronas aleatorias en train
        nn.Linear(hidden_units, num_classes),    # proyección final: representación → una logit por patología
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

    # "DEFAULT" usa los pesos más recientes publicados por torchvision para cada
    # arquitectura. Con pretrained=False se crea la arquitectura con pesos aleatorios,
    # lo que es necesario al cargar un checkpoint propio (los pesos se cargarán después).
    weights = "DEFAULT" if pretrained else None

    if model_name == "densenet121":
        # DenseNet-121: red con conexiones densas entre capas. Su capa de
        # clasificación original se llama 'classifier' (un único Linear).
        # La reemplazamos por nuestra cabeza de 4 capas.
        base = tv_models.densenet121(weights=weights)
        base.classifier = _classifier_head(
            base.classifier.in_features, hidden_units, dropout, num_classes
        )
    elif model_name == "resnet50":
        # ResNet-50: red con conexiones residuales (skip connections). Su capa
        # de clasificación final se llama 'fc' (fully connected).
        base = tv_models.resnet50(weights=weights)
        base.fc = _classifier_head(
            base.fc.in_features, hidden_units, dropout, num_classes
        )
    elif model_name.startswith("efficientnet"):
        # EfficientNet: familia de redes diseñadas por escalado compuesto.
        # Su clasificador es una secuencia; la última capa (índice -1) es el Linear.
        base = getattr(tv_models, model_name)(weights=weights)
        base.classifier[-1] = _classifier_head(
            base.classifier[-1].in_features, hidden_units, dropout, num_classes
        )

    return base


# ==============================================================================
# SECCIÓN 4: CARGA DE CHECKPOINTS
# ==============================================================================

def _has_simple_head(state: dict, model_name: str) -> bool:
    """
    Determina si el checkpoint usa una cabeza de salida simple (nn.Linear directo).

    Los checkpoints entrenados desde el notebook original usan nn.Linear sin
    capas intermedias. Los entrenados con build_model() usan _classifier_head
    (4 capas: Linear-ReLU-Dropout-Linear). La distinción se hace inspeccionando
    las claves del state_dict sin cargar el modelo.
    """
    # La presencia de 'classifier.weight' (sin índice numérico) indica cabeza simple.
    # La presencia de 'classifier.0.weight' indica que hay una secuencia (Sequential),
    # lo que corresponde a _classifier_head (índice 0 = primer Linear).
    if model_name == "densenet121":
        return "classifier.weight" in state and "classifier.0.weight" not in state
    if model_name == "resnet50":
        return "fc.weight" in state and "fc.0.weight" not in state
    if model_name.startswith("efficientnet"):
        return "classifier.1.weight" in state and "classifier.1.0.weight" not in state
    return False


def _build_simple_head_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Construye el backbone con una cabeza de salida simple (nn.Linear).

    Se usa cuando el checkpoint detectado proviene del notebook original,
    que no usa _classifier_head sino un único Linear como capa de salida.
    """
    # weights=None porque los pesos vendrán del checkpoint; no hay que descargar
    # los pesos de ImageNet aquí, lo que ahorraría tiempo y evita una descarga.
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


def _infer_num_classes(state: dict, model_name: str) -> int:
    """
    Infiere num_classes leyendo la forma del tensor de pesos de la capa de salida.

    Se prueban primero las claves de _classifier_head (índice 3) y luego las de
    la cabeza simple, para que el mismo código sirva con ambos formatos de checkpoint.

    Raises:
        ValueError: si no se encuentra ninguna clave reconocida en el state_dict.
    """
    # El tensor de pesos de la capa de salida tiene shape [num_classes, in_features].
    # shape[0] nos da directamente el número de clases, sea 13 o 14.
    # Se prueban primero las claves de la cabeza compleja (_classifier_head, índice 3)
    # antes que las de la cabeza simple, para mayor precisión en la detección.
    candidates: dict[str, list[str]] = {
        "densenet121": ["classifier.3.weight", "classifier.weight"],
        "resnet50":    ["fc.3.weight",          "fc.weight"],
    }
    if model_name in candidates:
        for key in candidates[model_name]:
            if key in state:
                return state[key].shape[0]
    if model_name.startswith("efficientnet"):
        for key in ["classifier.1.3.weight", "classifier.1.weight"]:
            if key in state:
                return state[key].shape[0]
    raise ValueError(f"No se puede inferir num_classes para '{model_name}'")


def load_checkpoint(
    cfg: dict, checkpoint_path: str, device: torch.device
) -> tuple[nn.Module, int]:
    """
    Carga un checkpoint en la arquitectura correcta y devuelve (model, num_classes).

    Infiere num_classes del shape de los pesos del checkpoint, e identifica
    automáticamente si se usó cabeza simple (nn.Linear) o cabeza compuesta
    (_classifier_head de build_model). Compatible con checkpoints del notebook
    original (14 clases, cabeza simple) y con los generados por main.py (13 clases,
    cabeza compuesta).

    Raises:
        FileNotFoundError: si el checkpoint no existe en la ruta indicada.
        RuntimeError: si el state_dict no es compatible con la arquitectura inferida.
    """
    # 1. Cargar el state_dict (diccionario de tensores con los pesos) desde el disco.
    #    map_location=device evita cargar en GPU cuando el checkpoint fue guardado en GPU
    #    pero la máquina actual solo tiene CPU (escenario habitual en inferencia).
    #    weights_only=True es una medida de seguridad: impide que torch.load ejecute
    #    código arbitrario embebido en el archivo (protección contra archivos maliciosos).
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_name = cfg["model"]["name"]

    # 2. Inferir el número de clases directamente del checkpoint.
    #    Esto es crítico: si construyéramos el modelo con num_classes=13 pero el
    #    checkpoint tiene pesos para 14 clases, load_state_dict lanzaría un error
    #    de "size mismatch". Al leerlo del checkpoint nos aseguramos de que siempre
    #    coincidan.
    num_classes = _infer_num_classes(state, model_name)

    # 3. Construir la arquitectura correcta según el formato detectado.
    #    Los checkpoints del notebook original tienen cabeza simple (nn.Linear),
    #    los de main.py tienen cabeza compleja (_classifier_head).
    if _has_simple_head(state, model_name):
        model = _build_simple_head_model(model_name, num_classes)
    else:
        model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            dropout=cfg["model"]["dropout"],
            hidden_units=cfg["model"]["hidden_units"],
            pretrained=False,  # los pesos vienen del checkpoint, no de ImageNet
        )

    # 4. Cargar los pesos en la arquitectura y moverla al dispositivo correcto.
    model.load_state_dict(state)
    model.to(device)

    # 5. Poner el modelo en modo evaluación.
    #    Esto desactiva el Dropout y usa las estadísticas globales del BatchNorm
    #    (en lugar de las del batch actual). Es obligatorio para obtener predicciones
    #    deterministas y correctas fuera del entrenamiento.
    model.eval()
    return model, num_classes


# ==============================================================================
# SECCIÓN 5: CAPA TARGET PARA GRADCAM
# ==============================================================================

def get_grad_cam_layer(model: nn.Module, model_name: str) -> list:
    """
    Devuelve la lista con la capa convolucional target para GradCAM.

    La capa elegida es la última del bloque de características de cada backbone:
    DenseNet-121 → features[-1] (DenseBlock + BatchNorm final),
    ResNet-50    → layer4[-1] (último bloque residual),
    EfficientNet → features[-1] (último bloque MBConv).

    Raises:
        ValueError: si model_name no está entre los backbones soportados.
    """
    # GradCAM necesita una capa convolucional (no la capa de clasificación) porque
    # los mapas de activación de las capas convolucionales retienen información
    # espacial. La capa de clasificación (Linear) ya ha colapsado esa información.
    # Elegimos la ÚLTIMA capa convolucional porque captura las características
    # de más alto nivel (las más relacionadas con la clase a explicar).
    if model_name == "densenet121":
        return [model.features[-1]]    # DenseBlock4 + BatchNorm2d final
    if model_name == "resnet50":
        return [model.layer4[-1]]      # último BasicBlock/Bottleneck
    if model_name.startswith("efficientnet"):
        return [model.features[-1]]    # último bloque MBConv
    raise ValueError(f"No hay capa GradCAM definida para '{model_name}'")