"""
Aplicación web interactiva para clasificación multietiqueta de patologías torácicas.

Interfaz Streamlit que permite cargar una radiografía, seleccionar el modelo activo,
visualizar las probabilidades por patología, comparar la imagen original con el mapa de
calor Grad-CAM de las clases más probables y descargar un informe PDF profesional.

Uso:
    streamlit run src/app.py
"""

import sys
from pathlib import Path
from typing import Optional

# Streamlit modifica sys.path al arrancar: añade el directorio que contiene el script
# (es decir, 'src/') a sys.path. Eso rompe los imports 'from src.x import ...' porque
# Python ya no puede encontrar el paquete 'src' como subdirectorio del proyecto.
# Solución: insertar la raíz del proyecto (el padre de 'src/') al principio de sys.path
# antes de cualquier otro import de src.
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

from src.models import (
    get_active_pathology_cols,
    get_grad_cam_layer,
    get_grad_cam_reshape,
    get_pathology_labels,
    load_checkpoint,
    parse_checkpoint_filename,
)
from src.model_registry import cargar_registro
from src.report import build_report_pdf
from src.image_utils import validar_imagen_radiografia, empaquetar_imagenes_zip

# Transformación de evaluación estándar: sin data augmentation.
# Los valores de normalización son la media y desviación estándar de ImageNet
# por canal RGB, iguales a los usados durante el entrenamiento.
# Es imprescindible que coincidan; de lo contrario la distribución de entrada
# sería diferente a la que el modelo aprendió a procesar.
_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),          # redimensionar al tamaño esperado por la red
    transforms.ToTensor(),                   # PIL Image → Tensor float [0.0, 1.0]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Etiquetas legibles para los desplegables de la barra lateral. La selección de modelo se
# presenta en dos pasos (arquitectura -> clases entrenadas) en lugar de una lista plana,
# porque con varias arquitecturas y configuraciones la lista única resulta poco manejable.
_BACKBONE_LABELS = {
    "densenet121": "DenseNet-121",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_t": "Swin-Tiny",
    "vgg16_bn": "VGG16-BN",
}
_CLASS_CONFIG_LABELS = {
    "full13": "13 clases (full13)",
    "nofracture12": "12 clases (sin Fracture)",
    "min5pct9": "9 clases (prevalencia ≥5%)",
}
_CLASS_CONFIG_ORDER = ["full13", "nofracture12", "min5pct9"]


def _agrupar_modelos_por_arquitectura(available_models: dict) -> dict:
    """Agrupa los modelos descubiertos por backbone: {backbone: {class_config: info}}."""
    por_arquitectura: dict = {}
    for info in available_models.values():
        por_arquitectura.setdefault(info["backbone"], {})[info["class_config"]] = info
    return por_arquitectura


def _ordenar_class_configs(configs) -> list:
    """Ordena las configuraciones para el desplegable: full13, nofracture12, min5pct9 y luego el resto."""
    return sorted(
        configs,
        key=lambda c: _CLASS_CONFIG_ORDER.index(c) if c in _CLASS_CONFIG_ORDER else len(_CLASS_CONFIG_ORDER),
    )


def _tabla_comparacion(labels_a, probs_a, labels_b, probs_b, threshold):
    """
    Compara las probabilidades de dos modelos sobre las patologías que ambos predicen.

    Como dos `class_config` distintas tienen distinto conjunto de clases, solo se comparan
    las patologías **comunes** (en el orden de labels_a). Devuelve un DataFrame con columnas
    'Patología', 'Modelo A', 'Modelo B', 'delta' (|A−B|) y 'Coinciden' (ambos al mismo lado
    del umbral). Si no hay clases comunes, el DataFrame está vacío (con esas columnas).
    """
    idx_b = {lab: i for i, lab in enumerate(labels_b)}
    filas = []
    for i, lab in enumerate(labels_a):
        if lab not in idx_b:
            continue
        pa, pb = float(probs_a[i]), float(probs_b[idx_b[lab]])
        filas.append({
            "Patología": lab,
            "Modelo A": pa,
            "Modelo B": pb,
            "delta": abs(pa - pb),
            "Coinciden": bool((pa >= threshold) == (pb >= threshold)),
        })
    return pd.DataFrame(filas, columns=["Patología", "Modelo A", "Modelo B", "delta", "Coinciden"])


def _chart_probabilidades(labels, probs, threshold):
    """Devuelve el gráfico de barras horizontal de probabilidades por patología (Altair)."""
    df = pd.DataFrame({
        "Patología": labels,
        "Probabilidad": probs,
        "Detectada": np.asarray(probs) >= threshold,
    }).sort_values("Probabilidad", ascending=False).reset_index(drop=True)
    base = alt.Chart(df).encode(
        x=alt.X("Probabilidad:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilidad"),
        y=alt.Y("Patología:N", sort="-x", title=None),
    )
    barras = base.mark_bar().encode(
        color=alt.condition(alt.datum.Detectada, alt.value("#28a745"), alt.value("#6c757d")),
        tooltip=["Patología", alt.Tooltip("Probabilidad:Q", format=".2%")],
    )
    etiquetas = base.mark_text(align="left", baseline="middle", dx=3).encode(
        text=alt.Text("Probabilidad:Q", format=".1%"),
    )
    return (barras + etiquetas).properties(height=max(300, 24 * len(labels)))


def _discover_models() -> dict[str, dict]:
    """
    Busca checkpoints de producción en models/ y deduce su backbone y class_config.

    Solo considera los checkpoints de producción (`mejor_modelo_*.pth`), ignorando
    candidatos (`_candidato_*`) y los de smoke test (`*_subset`). Devuelve
    {etiqueta_display: {"path", "backbone", "class_config"}}; class_config es None
    para checkpoints en el formato anterior (sin configuración de clases en el nombre).
    """
    models_dir = Path("models")
    result = {}
    if not models_dir.exists():
        return result
    for p in sorted(models_dir.glob("mejor_modelo_*.pth")):
        if p.stem.endswith("_subset"):
            continue  # checkpoint de smoke test, no es de producción
        backbone, class_config = parse_checkpoint_filename(p.name)
        label = f"{backbone} · {class_config}" if class_config else backbone
        result[label] = {"path": str(p), "backbone": backbone, "class_config": class_config}
    return result


@st.cache_resource
def _load_model(checkpoint_path: str, backbone: str, class_config: Optional[str]):
    """
    Carga el modelo desde el checkpoint indicado y resuelve sus etiquetas de clase.

    La terna (checkpoint_path, backbone, class_config) actúa como clave de caché:
    Streamlit mantiene una instancia cargada por modelo y no recarga entre interacciones.
    Las etiquetas se derivan de class_config cuando está disponible; en checkpoints en
    formato antiguo (class_config None) se infieren del número de salidas del modelo.
    """
    # @st.cache_resource es el mecanismo de Streamlit para recursos costosos.
    # Streamlit re-ejecuta todo el script en cada interacción del usuario (slider,
    # botón, upload). Sin caché, el modelo se cargaría de nuevo en cada clic.
    with open("config/config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    # load_checkpoint necesita el backbone real (no la etiqueta de display) para
    # construir la arquitectura correcta.
    cfg["model"]["name"] = backbone
    if class_config:
        cfg["data"]["class_config"] = class_config

    if not Path(checkpoint_path).exists():
        st.error(
            f"Checkpoint no encontrado: '{checkpoint_path}'. "
            "Entrena el modelo o actualiza 'model.checkpoint_path' en config/config.yml."
        )
        st.stop()

    device = torch.device("cpu")
    model, num_classes = load_checkpoint(cfg, checkpoint_path, device)
    target_layers = get_grad_cam_layer(model, backbone)

    # Etiquetas: por configuración de clases si está; si no, por número de salidas.
    # El último fallback (nombres genéricos) protege ante un checkpoint cuyo nº de
    # clases no encaje con ninguna lista conocida, evitando un desalineado silencioso.
    labels = get_active_pathology_cols(class_config) if class_config else None
    if labels is None or len(labels) != num_classes:
        try:
            labels = get_pathology_labels(num_classes)
        except ValueError:
            labels = [f"Clase {i}" for i in range(num_classes)]
    return cfg, model, target_layers, device, labels


def _collect_model_metrics(backbone: str, class_config: Optional[str]) -> Optional[dict]:
    """
    Devuelve las métricas de test del campeón registrado para este modelo, o None.

    Busca en `models/best_model_registry.json` por la clave `<backbone>_<class_config>`
    (o solo `<backbone>` en formato antiguo). Permite enriquecer el informe con la
    fiabilidad validada del modelo cuando existe; si no hay registro, se omite.
    """
    clave = f"{backbone}_{class_config}" if class_config else backbone
    registro = cargar_registro(clave)
    return registro.get("test_metrics") if registro else None


def _predict(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    # inference_mode deshabilita el grafo de autodiferenciación completamente.
    # Es más eficiente que no_grad() para inferencia pura porque también desactiva
    # algunas comprobaciones internas de PyTorch. No se puede usar para GradCAM
    # (que necesita gradientes), por eso está en una función separada.
    with torch.inference_mode():
        logits = model(tensor.to(device))
        return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def _compute_grad_cam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    target_layers: list,
    class_idx: int,
    device: torch.device,
    reshape_transform=None,
) -> np.ndarray:
    # GradCAM calcula la importancia de cada región de la imagen midiendo cómo
    # cambian los gradientes de la clase objetivo en la capa convolucional elegida.
    # Para ello necesita que el grafo de gradientes esté activo: NO se puede usar
    # dentro de 'with torch.inference_mode()' ni 'with torch.no_grad()'.
    # La librería pytorch_grad_cam gestiona internamente el contexto de gradientes.
    # reshape_transform es necesario para transformers (Swin): convierte las
    # activaciones de tokens (B, H, W, C) al formato convolucional (B, C, H, W).
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(class_idx)]
    mask = cam(input_tensor=tensor.to(device), targets=targets)
    return mask[0]  # (H, W) float32 en [0, 1]; 0=no relevante, 1=muy relevante


def main() -> None:
    st.set_page_config(page_title="CheXpert Classifier", page_icon="🫁", layout="wide")
    st.title("🫁 Clasificación de Patologías Torácicas")
    st.caption(
        "Apoyo al cribado de patologías torácicas en radiografías frontales mediante "
        "aprendizaje profundo, con explicabilidad Grad-CAM e informe descargable. "
        "Herramienta de demostración: no sustituye el criterio de un profesional médico."
    )

    # ==================================================================
    # PASO 1: DESCUBRIR Y SELECCIONAR EL MODELO
    # ==================================================================
    available_models = _discover_models()
    if not available_models:
        st.error(
            "No se encontraron checkpoints en models/. "
            "Ejecuta `python -m src.main` para entrenar un modelo."
        )
        st.stop()

    # ==================================================================
    # PASO 2: SIDEBAR CON PARÁMETROS
    # ==================================================================
    st.sidebar.header("Parámetros")

    # Selección en dos pasos: arquitectura y luego configuración de clases. El segundo
    # desplegable es dinámico (solo las configuraciones disponibles para la arquitectura
    # elegida), porque la matriz arquitectura x config no está completa.
    por_arquitectura = _agrupar_modelos_por_arquitectura(available_models)
    backbone_sel = st.sidebar.selectbox(
        "Arquitectura", sorted(por_arquitectura),
        format_func=lambda b: _BACKBONE_LABELS.get(b, b),
    )
    configs_disponibles = por_arquitectura[backbone_sel]
    config_sel = st.sidebar.selectbox(
        "Clases entrenadas", _ordenar_class_configs(configs_disponibles),
        format_func=lambda c: _CLASS_CONFIG_LABELS.get(c, c or "formato antiguo"),
    )
    info = configs_disponibles[config_sel]
    checkpoint_path = info["path"]
    modelo_label = (
        f"{_BACKBONE_LABELS.get(backbone_sel, backbone_sel)} · "
        f"{_CLASS_CONFIG_LABELS.get(config_sel, config_sel or 'formato antiguo')}"
    )

    cfg, model, target_layers, device, labels = _load_model(
        checkpoint_path, info["backbone"], info["class_config"]
    )

    st.caption(
        f"Backbone: **{info['backbone']}**"
        + (f" · config `{info['class_config']}`" if info["class_config"] else "")
        + f" — {len(labels)} patologías | Checkpoint: `{checkpoint_path}`"
    )

    default_threshold = float(cfg["training"]["threshold"])
    # El slider permite ajustar el umbral de detección en tiempo real sin recargar el modelo.
    # Un umbral bajo detecta más patologías pero aumenta los falsos positivos.
    # Un umbral alto es más conservador pero puede perder patologías presentes.
    threshold = st.sidebar.slider(
        "Umbral de clasificación", min_value=0.0, max_value=1.0,
        value=default_threshold, step=0.01,
    )
    # Tope de paneles Grad-CAM: solo se generan mapas de las patologías por encima del
    # umbral (prob ≥ umbral); este valor limita cuántas se muestran, por coste de CPU
    # (cada panel es una pasada de Grad-CAM).
    max_panels = st.sidebar.slider(
        "Máximo de paneles (Grad-CAM)",
        min_value=1, max_value=len(labels),
        value=min(8, len(labels)), step=1,
        help=(
            "Solo se muestran las patologías cuya probabilidad supera el umbral, las más "
            "probables primero. Baja el umbral para ver más patologías; súbelo para ver "
            "solo las más confiadas."
        ),
    )

    if st.sidebar.button("Limpiar historial", use_container_width=True):
        st.session_state.history = []

    # Comparación opcional con un segundo modelo (B) sobre la misma imagen (F8). Replica el
    # selector de dos pasos; las claves (key=) evitan colisiones de estado con el modelo A.
    comparar = st.sidebar.checkbox("Comparar con un segundo modelo")
    info_b = None
    modelo_label_b = None
    if comparar:
        st.sidebar.caption("Segundo modelo (B)")
        backbone_b = st.sidebar.selectbox(
            "Arquitectura (B)", sorted(por_arquitectura),
            format_func=lambda b: _BACKBONE_LABELS.get(b, b), key="arch_b",
        )
        config_b = st.sidebar.selectbox(
            "Clases entrenadas (B)", _ordenar_class_configs(por_arquitectura[backbone_b]),
            format_func=lambda c: _CLASS_CONFIG_LABELS.get(c, c or "formato antiguo"), key="cfg_b",
        )
        info_b = por_arquitectura[backbone_b][config_b]
        modelo_label_b = (
            f"{_BACKBONE_LABELS.get(backbone_b, backbone_b)} · "
            f"{_CLASS_CONFIG_LABELS.get(config_b, config_b or 'formato antiguo')}"
        )

    # ==================================================================
    # PASO 3: CARGA DE IMAGEN
    # ==================================================================
    uploaded = st.file_uploader("Cargar radiografía (JPEG o PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Carga una imagen para iniciar la inferencia.")
        _render_history()
        return

    # Validación en el límite de entrada: descarta imágenes inservibles (tamaño/resolución)
    # y avisa si la imagen no parece una radiografía (en color), antes de gastar cómputo.
    img_original = Image.open(uploaded)
    validacion = validar_imagen_radiografia(img_original, n_bytes=getattr(uploaded, "size", None))
    for _err in validacion["errores"]:
        st.error(_err)
    if not validacion["ok"]:
        _render_history()
        st.stop()
    for _aviso in validacion["avisos"]:
        st.warning(_aviso)

    # Cargar la imagen original en PIL (RGB) para mostrarla y para GradCAM.
    img_pil = img_original.convert("RGB")

    # Redimensionar a 224x224 para la visualización: queremos mostrar la misma
    # resolución que ve el modelo, no la imagen original de alta resolución.
    img_resized = img_pil.resize((224, 224))

    # Convertir a array float [0,1] para show_cam_on_image de pytorch_grad_cam.
    # Esta función requiere exactamente ese rango; con valores [0,255] el mapa
    # de calor aparecería casi invisible o totalmente saturado.
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    # Aplicar las transformaciones del modelo (normalización ImageNet) y
    # añadir la dimensión de batch: [3, 224, 224] → [1, 3, 224, 224].
    tensor = _EVAL_TRANSFORM(img_pil).unsqueeze(0)

    # ==================================================================
    # PASO 4: INFERENCIA
    # ==================================================================
    probs = _predict(model, tensor, device)

    # Inferencia del segundo modelo (B) sobre la misma imagen, si la comparación está activa.
    # Todos los backbones usan la misma transformación de entrada (224x224, normalización
    # ImageNet), así que el mismo 'tensor' es válido para B.
    probs_b = None
    labels_b = None
    if info_b is not None:
        _, model_b, target_layers_b, _, labels_b = _load_model(
            info_b["path"], info_b["backbone"], info_b["class_config"]
        )
        probs_b = _predict(model_b, tensor, device)

    # Imagen original en uint8 [0,255] para reutilizarla en cada panel y en las descargas.
    original_uint8 = (img_np * 255).astype(np.uint8)
    idx_top = int(np.argmax(probs))
    # Detectadas en orden de probabilidad descendente (coherente con tabla y gráfica).
    detected = [labels[i] for i in np.argsort(probs)[::-1] if probs[i] >= threshold]

    # ==================================================================
    # PASO 5: RESUMEN
    # ==================================================================
    st.divider()
    st.subheader("Resumen")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patologías detectadas", f"{len(detected)}/{len(labels)}")
    c2.metric("Probabilidad máxima", f"{probs[idx_top] * 100:.1f}%")
    c3.metric("Patología principal", labels[idx_top])
    c4.metric("Umbral", f"{threshold:.2f}")
    if detected:
        st.success(f"Patologías detectadas (umbral {threshold:.2f}): {', '.join(detected)}")
    else:
        st.warning(f"Ninguna patología supera el umbral de {threshold:.2f}.")

    # ==================================================================
    # PASO 6: PROBABILIDADES POR PATOLOGÍA
    # (la gráfica va antes de la explicabilidad para que el usuario vea el
    #  resumen cuantitativo sin tener que bajar hasta el final)
    # ==================================================================
    st.divider()
    st.subheader("Probabilidades por patología")

    # df_sorted del modelo A: se reutiliza en el informe PDF (PASO 8).
    df = pd.DataFrame({
        "Patología": labels,
        "Probabilidad": probs,
        "Detectada": probs >= threshold,
    })
    df_sorted = df.sort_values("Probabilidad", ascending=False).reset_index(drop=True)

    if probs_b is None:
        # Un solo modelo: una gráfica + tabla.
        st.altair_chart(_chart_probabilidades(labels, probs, threshold), use_container_width=True)
        with st.expander("Ver tabla de valores"):
            df_display = df_sorted.copy()
            df_display["Detectada"] = df_display["Detectada"].map({True: "✓", False: ""})

            def _color_row(row: pd.Series) -> list[str]:
                # Verde si la patología fue detectada; gris claro en caso contrario.
                color = "#d4edda" if row["Detectada"] == "✓" else "#f8f9fa"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                df_display.style.apply(_color_row, axis=1).format({"Probabilidad": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )
    else:
        # Comparación: una gráfica por modelo (titulada con su nombre) + gráfica comparativa.
        st.markdown(f"##### {modelo_label}")
        st.altair_chart(_chart_probabilidades(labels, probs, threshold), use_container_width=True)
        st.markdown(f"##### {modelo_label_b}")
        st.altair_chart(_chart_probabilidades(labels_b, probs_b, threshold), use_container_width=True)

        st.markdown("##### Comparación (patologías comunes)")
        df_cmp = _tabla_comparacion(labels, probs, labels_b, probs_b, threshold)
        if df_cmp.empty:
            st.info("Los dos modelos no comparten patologías comparables.")
        else:
            n_ok = int(df_cmp["Coinciden"].sum())
            st.caption(
                f"{n_ok}/{len(df_cmp)} patologías comunes con la misma decisión a umbral "
                f"{threshold:.2f} · diferencia media |A−B| = {df_cmp['delta'].mean():.3f}"
            )
            # Barras agrupadas (yOffset = una barra por modelo en cada patología; Altair 5+).
            df_long = df_cmp.melt(
                id_vars="Patología", value_vars=["Modelo A", "Modelo B"],
                var_name="Modelo", value_name="Probabilidad",
            )
            cmp_chart = alt.Chart(df_long).mark_bar().encode(
                x=alt.X("Probabilidad:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilidad"),
                y=alt.Y("Patología:N", sort="-x", title=None),
                yOffset="Modelo:N",
                color=alt.Color("Modelo:N", title=None, scale=alt.Scale(
                    domain=["Modelo A", "Modelo B"], range=["#1f77b4", "#ff7f0e"])),
                tooltip=["Patología", "Modelo", alt.Tooltip("Probabilidad:Q", format=".2%")],
            ).properties(height=max(220, 26 * len(df_cmp)))
            st.altair_chart(cmp_chart, use_container_width=True)
            with st.expander("Ver tabla comparativa"):
                st.dataframe(
                    df_cmp.rename(columns={"delta": "|A−B|"}).style.format(
                        {"Modelo A": "{:.3f}", "Modelo B": "{:.3f}", "|A−B|": "{:.3f}"}
                    ),
                    use_container_width=True, hide_index=True,
                )

    # ==================================================================
    # PASO 7: EXPLICABILIDAD — original + Grad-CAM (de cada modelo si se compara)
    # ==================================================================
    st.divider()
    st.subheader("Explicabilidad visual")
    if probs_b is not None:
        st.caption(f"**A:** {modelo_label}  ·  **B:** {modelo_label_b}")
    # Se generan mapas Grad-CAM de las patologías por encima del umbral (modelo A), las más
    # probables primero y hasta max_panels. 'panels' (de A) se usa en el PDF/ZIP. En modo
    # comparación se genera además el Grad-CAM del modelo B para las patologías que comparte.
    panels = []
    if detected:
        n_show = min(len(detected), max_panels)
        orden = np.argsort(probs)[::-1][:n_show]
        # Los transformers (Swin) requieren un reshape de las activaciones para Grad-CAM;
        # las CNN devuelven None.
        reshape_a = get_grad_cam_reshape(info["backbone"])
        idx_b = {lab: i for i, lab in enumerate(labels_b)} if probs_b is not None else {}
        reshape_b = get_grad_cam_reshape(info_b["backbone"]) if probs_b is not None else None
        with st.spinner("Generando mapas Grad-CAM…"):
            for idx in orden:
                lab = labels[int(idx)]
                # ClassifierOutputTarget necesita un índice entero, no el nombre de la clase.
                mask = _compute_grad_cam(
                    model, tensor, target_layers, int(idx), device, reshape_transform=reshape_a
                )
                # show_cam_on_image usa un colormap jet (azul→rojo): el rojo marca las zonas
                # que más influyeron en la predicción de esa patología.
                heat = show_cam_on_image(img_np, mask, use_rgb=True)  # (H, W, 3) uint8
                heat_b = None
                if probs_b is not None and lab in idx_b:
                    mask_b = _compute_grad_cam(
                        model_b, tensor, target_layers_b, idx_b[lab], device, reshape_transform=reshape_b
                    )
                    heat_b = show_cam_on_image(img_np, mask_b, use_rgb=True)
                panels.append({
                    "label": lab,
                    "prob": float(probs[int(idx)]),
                    "heatmap": heat,
                    "heatmap_b": heat_b,
                })
        ocultas = len(detected) - len(panels)
        nota_tope = f" (se omiten {ocultas} por el tope de paneles)" if ocultas > 0 else ""
        st.caption(
            f"Solo se muestran las {len(panels)} patología(s) cuya probabilidad supera el "
            f"umbral de {threshold:.2f}{nota_tope}. Ajusta el umbral en la barra lateral. "
            "🔴 rojo = mayor influencia · 🔵 azul = menor influencia."
        )
        for p in panels:
            if probs_b is not None and p["label"] in idx_b:
                pb = float(probs_b[idx_b[p["label"]]])
                st.markdown(f"**{p['label']}** — A: {p['prob'] * 100:.1f}% · B: {pb * 100:.1f}%")
            else:
                st.markdown(f"**{p['label']}** — {p['prob'] * 100:.1f}%")
            if probs_b is None:
                col1, col2 = st.columns(2)
                col1.image(img_resized, use_container_width=True, caption="Original")
                col2.image(p["heatmap"], use_container_width=True, caption="Grad-CAM")
            else:
                col1, col2, col3 = st.columns(3)
                col1.image(img_resized, use_container_width=True, caption="Original")
                col2.image(p["heatmap"], use_container_width=True, caption="Grad-CAM · A")
                if p["heatmap_b"] is not None:
                    col3.image(p["heatmap_b"], use_container_width=True, caption="Grad-CAM · B")
                else:
                    col3.info("Patología no presente en el modelo B")
    else:
        st.info(
            f"Ninguna patología supera el umbral de {threshold:.2f}, así que no se muestran "
            "mapas Grad-CAM. Baja el umbral en la barra lateral para visualizar las "
            "patologías más probables."
        )

    # ==================================================================
    # PASO 8: EXPORTAR (informe PDF + imágenes ZIP)
    # ==================================================================
    st.divider()
    st.subheader("Exportar")

    # Informe PDF profesional con tablas, gráfica, paneles original+Grad-CAM y, si el
    # modelo tiene métricas de validación registradas, su fiabilidad. Se genera en
    # memoria a partir de los datos ya calculados (no recalcula Grad-CAM).
    filas = [
        {"patologia": r["Patología"], "probabilidad": float(r["Probabilidad"]), "detectada": bool(r["Detectada"])}
        for _, r in df_sorted.iterrows()
    ]
    contexto = {
        "titulo": "Informe de análisis de radiografía torácica",
        "modelo": info["backbone"],
        "class_config": info["class_config"],
        "checkpoint": checkpoint_path,
        "imagen_nombre": uploaded.name,
        "umbral": threshold,
        "filas": filas,
        "detectadas": detected,
        "original": original_uint8,
        "panels": panels,
        "metricas_modelo": _collect_model_metrics(info["backbone"], info["class_config"]),
    }
    col_pdf, col_zip = st.columns(2)
    with col_pdf:
        st.download_button(
            label="Descargar informe (PDF)",
            data=build_report_pdf(contexto),
            file_name=f"informe_{Path(uploaded.name).stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col_zip:
        st.download_button(
            label="Descargar imágenes (ZIP)",
            data=empaquetar_imagenes_zip(original_uint8, panels),
            file_name=f"imagenes_{Path(uploaded.name).stem}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    # ==================================================================
    # PASO 9: HISTORIAL DE SESIÓN
    # ==================================================================
    st.divider()
    # st.session_state persiste entre re-ejecuciones del script (causadas por
    # interacciones del usuario) pero se resetea al recargar la página.
    # Esto permite acumular el historial de análisis de la sesión actual.
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Modelo": modelo_label,
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
    hist_df = pd.DataFrame(st.session_state.history)
    with st.expander(f"Historial de la sesión ({len(st.session_state.history)} análisis)"):
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar historial (CSV)",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="historial_sesion.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
