"""Implementación manual de Grad-CAM sin dependencias de OpenCV."""
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps


def compute_grad_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int,
    reshape_transform: Optional[Callable] = None,
) -> np.ndarray:
    """
    Calcula el mapa de calor Grad-CAM para una clase objetivo.

    Args:
        model: Modelo de PyTorch en modo evaluación.
        input_tensor: Tensor de entrada con batch dimension [1, C, H, W].
        target_layer: Capa convolucional donde calcular Grad-CAM.
        target_class: Índice de la clase objetivo.
        reshape_transform: Función opcional para reshaping (transformers).

    Returns:
        Mapa de calor normalizado [0, 1] redimensionado al tamaño de la entrada, forma (H, W).
    """
    activations = None
    gradients = None

    def _save_grad(grad):
        nonlocal gradients
        gradients = grad.detach()

    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output.detach()
        # Solo registrar el hook de gradiente si el tensor lo admite. Bajo inference_mode/no_grad
        # (inferencia normal) el tensor no requiere gradiente; si un hook quedara registrado de una
        # llamada previa sobre el modelo cacheado, register_hook lanzaría RuntimeError. Este guard
        # hace la inferencia robusta aunque el hook no se hubiera retirado.
        if not output.requires_grad:
            return output
        # Se captura el gradiente con un hook de TENSOR (no de módulo): así se evita
        # register_full_backward_hook, que falla cuando la capa objetivo va seguida de una
        # operación in-place (p. ej. F.relu(features, inplace=True) en DenseNet).
        output.register_hook(_save_grad)
        # Devolver un clon hace que esa operación in-place posterior actúe sobre el clon y no
        # sobre el tensor del que depende el gradiente -> evita el error "view modified inplace".
        return output.clone()

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad()
        output = model(input_tensor)
        target_score = output[0, target_class]
        target_score.backward()

        # Transformers (Swin): reordenar tokens (B, H, W, C) -> (B, C, H, W).
        if reshape_transform is not None:
            activations = reshape_transform(activations)
            gradients = reshape_transform(gradients)

        # Pesos por canal = global average pooling de los gradientes.
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)

        # Subir el CAM a la resolución de la imagen de entrada: las activaciones de la capa
        # objetivo tienen menor resolución espacial (p. ej. 7x7) que los 224x224 de entrada.
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]

        # Normalizar a [0, 1].
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()


def apply_colormap(cam: np.ndarray) -> np.ndarray:
    """
    Aplica el colormap 'jet' (matplotlib, sin OpenCV) a un mapa de calor [0, 1].

    Vectorizado (sin bucles): rápido incluso a 224x224 en CPU.

    Args:
        cam: Mapa de calor normalizado [0, 1] de forma (H, W).

    Returns:
        Imagen RGB con el colormap aplicado (H, W, 3) en uint8.
    """
    rgba = colormaps["jet"](np.clip(cam, 0.0, 1.0))   # (H, W, 4) float en [0, 1]
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Superpone el heatmap sobre la imagen original.

    Args:
        image: Imagen original RGB (H, W, 3) en uint8 [0, 255].
        heatmap: Mapa de calor RGB (H, W, 3) en uint8.
        alpha: Transparencia del heatmap (0-1).

    Returns:
        Imagen combinada (H, W, 3) en uint8.
    """
    image_float = image.astype(np.float32) / 255.0
    heatmap_float = heatmap.astype(np.float32) / 255.0
    blended = (1 - alpha) * image_float + alpha * heatmap_float
    return np.clip(blended * 255, 0, 255).astype(np.uint8)
