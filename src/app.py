"""
Aplicación web interactiva para clasificación multietiqueta de patologías torácicas.

Interfaz Streamlit que permite cargar una radiografía, seleccionar el modelo activo,
visualizar las probabilidades por patología con GradCAM y exportar los resultados.

Uso:
    streamlit run src/app.py
"""

import re
import sys
from pathlib import Path

# Streamlit adds src/ to sys.path when running this file directly;
# insert the project root so that 'from src.x import' resolves correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
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


def _discover_models() -> dict[str, str]:
    """
    Busca checkpoints disponibles en el directorio models/.

    Devuelve un diccionario {etiqueta_display: ruta_str}. La etiqueta se extrae
    del patrón mejor_modelo_<name>.pth; si el nombre no sigue el patrón se usa
    el stem completo del fichero.
    """
    models_dir = Path("models")
    result = {}
    if not models_dir.exists():
        return result
    for p in sorted(models_dir.glob("*.pth")):
        m = re.match(r"mejor_modelo_(.+)\.pth", p.name)
        label = m.group(1) if m else p.stem
        result[label] = str(p)
    return result


@st.cache_resource
def _load_model(checkpoint_path: str, model_name: str):
    """
    Carga el modelo desde el checkpoint indicado.

    El par (checkpoint_path, model_name) actúa como clave de caché: Streamlit
    mantiene una instancia cargada por modelo y no recarga entre interacciones.
    """
    with open("config/config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["name"] = model_name

    if not Path(checkpoint_path).exists():
        st.error(
            f"Checkpoint no encontrado: '{checkpoint_path}'. "
            "Entrena el modelo o actualiza 'model.checkpoint_path' en config/config.yml."
        )
        st.stop()

    device = torch.device("cpu")
    model, num_classes = load_checkpoint(cfg, checkpoint_path, device)
    target_layers = get_grad_cam_layer(model, model_name)
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
    # GradCAM requiere flujo de gradientes — sin inference_mode aquí
    mask = cam(input_tensor=tensor.to(device), targets=targets)
    return mask[0]  # (H, W) float32 en [0, 1]


def main() -> None:
    st.set_page_config(page_title="CheXpert Classifier", layout="wide")
    st.title("Clasificación de Patologías Torácicas")

    # ── Descubrir modelos disponibles ──────────────────────────────────────────
    available_models = _discover_models()
    if not available_models:
        st.error(
            "No se encontraron checkpoints en models/. "
            "Ejecuta `python -m src.main` para entrenar un modelo."
        )
        st.stop()

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    st.sidebar.header("Parámetros")

    selected_label = st.sidebar.selectbox("Modelo", list(available_models.keys()))
    checkpoint_path = available_models[selected_label]

    cfg, model, target_layers, device, labels = _load_model(checkpoint_path, selected_label)

    st.caption(
        f"Backbone: **{selected_label}** — {len(labels)} patologías | "
        f"Checkpoint: `{checkpoint_path}`"
    )

    default_threshold = float(cfg["training"]["threshold"])
    threshold = st.sidebar.slider(
        "Umbral de clasificación", min_value=0.0, max_value=1.0,
        value=default_threshold, step=0.01,
    )
    gradcam_label = st.sidebar.selectbox("Patología para GradCAM", labels)

    # ── Carga de imagen ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Cargar radiografía (JPEG o PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Carga una imagen para iniciar la inferencia.")
        _render_history()
        return

    img_pil = Image.open(uploaded).convert("RGB")
    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    tensor = _EVAL_TRANSFORM(img_pil).unsqueeze(0)

    probs = _predict(model, tensor, device)

    class_idx = labels.index(gradcam_label)
    mask = _compute_grad_cam(model, tensor, target_layers, class_idx, device)
    cam_image = show_cam_on_image(img_np, mask, use_rgb=True)

    # ── Imágenes ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(img_resized, use_container_width=True)
    with col2:
        st.subheader(f"GradCAM — {gradcam_label}")
        st.image(cam_image, use_container_width=True)

    # ── Tabla y gráfico de probabilidades ────────────────────────────────────────
    st.subheader("Probabilidades por patología")

    df = pd.DataFrame({
        "Patología": labels,
        "Probabilidad": probs,
        "Detectada": probs >= threshold,
    })
    df_sorted = df.sort_values("Probabilidad", ascending=False).reset_index(drop=True)

    chart = (
        alt.Chart(df_sorted)
        .mark_bar()
        .encode(
            x=alt.X("Probabilidad:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilidad"),
            y=alt.Y("Patología:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.Detectada,
                alt.value("#28a745"),
                alt.value("#6c757d"),
            ),
            tooltip=["Patología", alt.Tooltip("Probabilidad:Q", format=".4f")],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Ver tabla de valores"):
        df_display = df_sorted.copy()
        df_display["Detectada"] = df_display["Detectada"].map({True: "✓", False: ""})

        def _color_row(row: pd.Series) -> list[str]:
            color = "#d4edda" if row["Detectada"] == "✓" else "#f8f9fa"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_display.style.apply(_color_row, axis=1).format({"Probabilidad": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # ── Resultado y exportar ─────────────────────────────────────────────────────
    detected = df.loc[df["Detectada"], "Patología"].tolist()
    if detected:
        st.success(f"Patologías detectadas (umbral {threshold:.2f}): {', '.join(detected)}")
    else:
        st.warning(f"Ninguna patología supera el umbral de {threshold:.2f}.")

    csv_data = df_sorted.drop(columns=["Detectada"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar resultados (CSV)",
        data=csv_data,
        file_name=f"resultados_{uploaded.name}.csv",
        mime="text/csv",
    )

    # ── Historial de sesión ──────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Modelo": selected_label,
        "Imagen": uploaded.name,
        "Hora": pd.Timestamp.now().strftime("%H:%M:%S"),
        "Detectadas": ", ".join(detected) if detected else "—",
        "Umbral": f"{threshold:.2f}",
    })

    _render_history()


def _render_history() -> None:
    """Muestra el historial de análisis acumulado en la sesión actual."""
    if not st.session_state.get("history"):
        return
    with st.expander(f"Historial de la sesión ({len(st.session_state.history)} análisis)"):
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
