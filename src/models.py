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

from typing import Callable, Dict, List, Optional, Tuple

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

# Las 5 patologías de la competición oficial de CheXpert (Irvin et al., 2019).
# Se usan como criterio de promoción del mejor modelo porque están bien representadas
# en el conjunto de test silver (valid de Stanford) —soportes 30-110 positivos—, lo que
# da una AUROC estable, y porque son el estándar de comparación con la literatura.
CHEXPERT_COMPETITION_5: List[str] = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'
]

# Configuraciones de clases seleccionables por experimento (config-driven).
# Cada configuración fija EXPLÍCITAMENTE sus clases activas —derivadas de la lista
# canónica de 13 para evitar divergencias de nomenclatura— y la política anti-ruido
# de eliminación de estudios cuyas únicas etiquetas positivas pertenecen a clases
# descartadas (convertirlos en negativos introduciría ruido de etiqueta):
#   - "ninguno":       no se elimina ningún estudio (no hay clases descartadas).
#   - "orfanos":       elimina estudios cuya ÚNICA etiqueta positiva era de una clase
#                      descartada (no toca estudios ya sin positivos en full13).
#   - "sin_positivos": elimina todo estudio que quede sin ninguna etiqueta positiva
#                      en las clases activas (incluye los ya negativos en origen).
_NOFRACTURE_EXCLUIDAS = ("Fracture",)
_MIN5PCT_EXCLUIDAS = ("Enlarged Cardiomediastinum", "Lung Lesion", "Pneumonia", "Fracture")

CLASS_CONFIGS: Dict[str, Dict] = {
    "full13": {
        "cols": list(CHEXPERT_PATHOLOGY_COLS),
        "anti_ruido": "ninguno",
    },
    "nofracture12": {
        "cols": [c for c in CHEXPERT_PATHOLOGY_COLS if c not in _NOFRACTURE_EXCLUIDAS],
        "anti_ruido": "orfanos",
    },
    "min5pct9": {
        "cols": [c for c in CHEXPERT_PATHOLOGY_COLS if c not in _MIN5PCT_EXCLUIDAS],
        "anti_ruido": "sin_positivos",
    },
}


def get_active_pathology_cols(class_config: str) -> List[str]:
    """Devuelve las columnas de patología activas para una configuración de clases."""
    if class_config not in CLASS_CONFIGS:
        raise ValueError(
            f"class_config '{class_config}' no definido. Opciones: {list(CLASS_CONFIGS)}"
        )
    return list(CLASS_CONFIGS[class_config]["cols"])


def parse_checkpoint_filename(filename: str) -> Tuple[str, Optional[str]]:
    """
    Extrae (backbone, class_config) del nombre de un checkpoint.

    Reconoce el patrón del proyecto `mejor_modelo_<backbone>_<class_config>.pth` y, por
    compatibilidad, el formato anterior sin configuración de clases
    (`mejor_modelo_<backbone>.pth`), en cuyo caso class_config es None. Respeta los
    backbones cuyo nombre contiene guion bajo (p. ej. `convnext_tiny`) detectando el
    sufijo de configuración entre las claves conocidas de CLASS_CONFIGS.

    Ejemplos
    --------
    >>> parse_checkpoint_filename("mejor_modelo_convnext_tiny_min5pct9.pth")
    ('convnext_tiny', 'min5pct9')
    >>> parse_checkpoint_filename("mejor_modelo_densenet121.pth")
    ('densenet121', None)
    """
    stem = filename.replace("\\", "/").split("/")[-1]
    if stem.endswith(".pth"):
        stem = stem[:-4]
    for prefijo in ("mejor_modelo_", "_candidato_"):
        if stem.startswith(prefijo):
            stem = stem[len(prefijo):]
            break
    if stem.endswith("_subset"):
        stem = stem[: -len("_subset")]
    for cc in CLASS_CONFIGS:
        if stem.endswith("_" + cc):
            return stem[: -(len(cc) + 1)], cc
    return stem, None


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
SUPPORTED_MODELS = ("densenet121", "vgg16_bn", "resnet50", "convnext_tiny", "swin_t")


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
    Supported backbones: densenet121, vgg16_bn, resnet50, convnext_tiny, swin_t."""
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
    elif model_name == "vgg16_bn":
        # VGG-16 con BatchNorm: red secuencial clásica (pila de convoluciones sin
        # conexiones de salto). El BatchNorm la hace mucho más estable de entrenar
        # que la VGG original. Su clasificador es una secuencia cuya última capa
        # (índice -1) es el Linear de salida.
        base = tv_models.vgg16_bn(weights=weights)
        base.classifier[-1] = _classifier_head(
            base.classifier[-1].in_features, hidden_units, dropout, num_classes
        )
    elif model_name == "resnet50":
        # ResNet-50: red con conexiones residuales (skip connections). Su capa
        # de clasificación final se llama 'fc' (fully connected).
        base = tv_models.resnet50(weights=weights)
        base.fc = _classifier_head(
            base.fc.in_features, hidden_units, dropout, num_classes
        )
    elif model_name == "convnext_tiny":
        # ConvNeXt-Tiny: CNN moderna inspirada en transformers. Su clasificador es
        # LayerNorm → Flatten → Linear; la última capa (índice -1) es el Linear.
        base = tv_models.convnext_tiny(weights=weights)
        base.classifier[-1] = _classifier_head(
            base.classifier[-1].in_features, hidden_units, dropout, num_classes
        )
    elif model_name == "swin_t":
        # Swin Transformer Tiny: transformer de visión jerárquico con ventanas
        # desplazadas. Su cabeza de clasificación se llama 'head' (un Linear).
        base = tv_models.swin_t(weights=weights)
        base.head = _classifier_head(
            base.head.in_features, hidden_units, dropout, num_classes
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
    if model_name == "vgg16_bn":
        return "classifier.6.weight" in state and "classifier.6.0.weight" not in state
    if model_name == "resnet50":
        return "fc.weight" in state and "fc.0.weight" not in state
    if model_name == "convnext_tiny":
        return "classifier.2.weight" in state and "classifier.2.0.weight" not in state
    if model_name == "swin_t":
        return "head.weight" in state and "head.0.weight" not in state
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
    elif model_name == "vgg16_bn":
        base = tv_models.vgg16_bn(weights=None)
        base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, num_classes)
    elif model_name == "resnet50":
        base = tv_models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
    elif model_name == "convnext_tiny":
        base = tv_models.convnext_tiny(weights=None)
        base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, num_classes)
    elif model_name == "swin_t":
        base = tv_models.swin_t(weights=None)
        base.head = nn.Linear(base.head.in_features, num_classes)
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
        "densenet121":   ["classifier.3.weight",   "classifier.weight"],
        "vgg16_bn":      ["classifier.6.3.weight", "classifier.6.weight"],
        "resnet50":      ["fc.3.weight",            "fc.weight"],
        "convnext_tiny": ["classifier.2.3.weight", "classifier.2.weight"],
        "swin_t":        ["head.3.weight",          "head.weight"],
    }
    if model_name in candidates:
        for key in candidates[model_name]:
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
    Devuelve la lista con la capa target para GradCAM.

    La capa elegida es la última del bloque de características de cada backbone:
    DenseNet-121 → features[-1] (DenseBlock + BatchNorm final),
    VGG16-BN     → última Conv2d de features,
    ResNet-50    → layer4[-1] (último bloque residual),
    ConvNeXt     → features[-1] (última etapa CNBlock),
    Swin-T       → norm (LayerNorm final; requiere reshape, ver get_grad_cam_reshape).

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
    if model_name == "vgg16_bn":
        # features termina en MaxPool2d; se busca la última Conv2d real de la pila.
        convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
        return [convs[-1]]
    if model_name == "resnet50":
        return [model.layer4[-1]]      # último BasicBlock/Bottleneck
    if model_name == "convnext_tiny":
        return [model.features[-1]]    # última etapa CNBlock
    if model_name == "swin_t":
        # LayerNorm final, cuya salida (B, H, W, C) conserva la rejilla espacial de
        # tokens; GradCAM la reorganiza a (B, C, H, W) con get_grad_cam_reshape.
        return [model.norm]
    raise ValueError(f"No hay capa GradCAM definida para '{model_name}'")


def get_grad_cam_reshape(model_name: str) -> Optional[Callable]:
    """
    Devuelve la función reshape_transform para GradCAM, o None para las CNN.

    Los transformers de visión (Swin) emiten las activaciones de la capa target con
    los tokens en formato channels-last (B, H, W, C); GradCAM espera mapas de
    activación convolucionales (B, C, H, W). Para `swin_t` se permutan los ejes; las
    CNN no requieren ninguna transformación.
    """
    if model_name == "swin_t":
        return lambda tensor: tensor.permute(0, 3, 1, 2)
    return None