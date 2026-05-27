"""
Aplicación web interactiva para clasificación multietiqueta de patologías torácicas.

Interfaz Streamlit que permite cargar una radiografía, seleccionar el modelo activo,
visualizar las probabilidades por patología con GradCAM y exportar los resultados.

Uso:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

# Streamlit adds src/ to sys.path when running this file directly;
# insert the project root so that 'from src.x import' resolves correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.models import get_grad_cam_layer, get_pathology_labels, load_checkpoint

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def _load_model():
    with open("config/config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint = cfg["model"]["checkpoint_path"]
    if not Path(checkpoint).exists():
        st.error(
            f"Checkpoint no encontrado: '{checkpoint}'. "
            "Entrena el modelo o actualiza 'model.checkpoint_path' en config/config.yml."
        )
        st.stop()

    device = torch.device("cpu")
    model, num_classes = load_checkpoint(cfg, checkpoint, device)
    target_layers = get_grad_cam_layer(model, cfg["model"]["name"])
    labels = get_pathology_labels(num_classes)
    return cfg, model, target_layers, device, labels


def _predict(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        logits = model(tensor.to(device))
        return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def _compute_grad_cam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    target_layers: list,
    class_idx: int,
    device: torch.device,
) -> np.ndarray:
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]
    # GradCAM requires gradient flow — no inference_mode context here
    mask = cam(input_tensor=tensor.to(device), targets=targets)
    return mask[0]  # (H, W) float32 in [0, 1]


def main() -> None:
    st.set_page_config(page_title="CheXpert Classifier", layout="wide")
    st.title("Clasificación de Patologías Torácicas")
    st.caption("DenseNet-121 — CheXpert | Clasificación multietiqueta de patologías torácicas")

    cfg, model, target_layers, device, labels = _load_model()
    default_threshold = float(cfg["training"]["threshold"])

    st.sidebar.header("Parámetros")
    threshold = st.sidebar.slider(
        "Umbral de clasificación", min_value=0.0, max_value=1.0,
        value=default_threshold, step=0.01,
    )
    gradcam_label = st.sidebar.selectbox("Patología para GradCAM", labels)

    uploaded = st.file_uploader("Cargar radiografía (JPEG o PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Carga una imagen para iniciar la inferencia.")
        return

    img_pil = Image.open(uploaded).convert("RGB")
    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0  # [0,1] requerido por show_cam_on_image

    tensor = _EVAL_TRANSFORM(img_pil).unsqueeze(0)

    probs = _predict(model, tensor, device)

    class_idx = labels.index(gradcam_label)
    mask = _compute_grad_cam(model, tensor, target_layers, class_idx, device)
    cam_image = show_cam_on_image(img_np, mask, use_rgb=True)  # (H, W, 3) uint8

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(img_resized, use_container_width=True)
    with col2:
        st.subheader(f"GradCAM — {gradcam_label}")
        st.image(cam_image, use_container_width=True)

    st.subheader("Probabilidades por patología")
    df = pd.DataFrame({
        "Patología": labels,
        "Probabilidad": probs,
        "Detectada": ["✓" if p >= threshold else "" for p in probs],
    })

    def _color_row(row: pd.Series) -> list[str]:
        color = "#d4edda" if row["Detectada"] == "✓" else "#f8f9fa"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        df.style.apply(_color_row, axis=1).format({"Probabilidad": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    detected = df.loc[df["Detectada"] == "✓", "Patología"].tolist()
    if detected:
        st.success(f"Patologías detectadas (umbral {threshold:.2f}): {', '.join(detected)}")
    else:
        st.warning(f"Ninguna patología supera el umbral de {threshold:.2f}.")


if __name__ == "__main__":
    main()
