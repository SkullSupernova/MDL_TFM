"""Implementación manual de Grad-CAM sin dependencias de OpenCV."""
import numpy as np
import torch
from typing import Callable, Optional


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
        Mapa de calor normalizado [0, 1] de forma (H, W).
    """
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
    
    # Registrar hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        model.zero_grad()
        output = model(input_tensor)
        
        # Obtener el score de la clase objetivo
        target_score = output[0, target_class]
        
        # Backward pass
        target_score.backward()
        
        # Procesar activaciones y gradientes
        if reshape_transform is not None:
            activations = reshape_transform(activations)
            gradients = reshape_transform(gradients)
        
        # Calcular pesos por canal (global average pooling de gradientes)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Ponderar activaciones por pesos y sumar canales
        cam = (weights * activations).sum(dim=1).squeeze(0)
        
        # Aplicar ReLU y normalizar
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        return cam.cpu().numpy()
    
    finally:
        # Eliminar hooks
        handle_forward.remove()
        handle_backward.remove()


def apply_colormap(cam: np.ndarray) -> np.ndarray:
    """
    Aplica un colormap jet manualmente sin OpenCV.
    
    Args:
        cam: Mapa de calor normalizado [0, 1] de forma (H, W).
    
    Returns:
        Imagen RGB con colormap aplicado (H, W, 3) en uint8.
    """
    # Crear colormap jet manualmente
    h, w = cam.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalizar a [0, 255]
    cam_normalized = (cam * 255).astype(np.uint8)
    
    # Aplicar colormap jet simplificado
    for i in range(h):
        for j in range(w):
            val = cam_normalized[i, j]
            if val < 64:
                rgb[i, j] = [0, 0, val * 4]
            elif val < 128:
                rgb[i, j] = [0, (val - 64) * 4, 255]
            elif val < 192:
                rgb[i, j] = [(val - 128) * 4, 255, 255 - (val - 128) * 4]
            else:
                rgb[i, j] = [255, 255 - (val - 192) * 4, 0]
    
    return rgb


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
    # Convertir a float para la mezcla
    image_float = image.astype(np.float32) / 255.0
    heatmap_float = heatmap.astype(np.float32) / 255.0
    
    # Mezcla alfa
    blended = (1 - alpha) * image_float + alpha * heatmap_float
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    return blended