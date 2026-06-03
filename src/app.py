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
) -> np.ndarray:
    # GradCAM calcula la importancia de cada región de la imagen midiendo cómo
    # cambian los gradientes de la clase objetivo en la capa convolucional elegida.
    # Para ello necesita que el grafo de gradientes esté activo: NO se puede usar
    # dentro de 'with torch.inference_mode()' ni 'with torch.no_grad()'.
    # La librería pytorch_grad_cam gestiona internamente el contexto de gradientes.
    cam = GradCAM(model=model, target_layers=target_layers)
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

    # Selector de modelo: lista todos los .pth disponibles en models/.
    # Al cambiar el modelo, @st.cache_resource devuelve la instancia ya cargada
    # si ese checkpoint ya fue procesado antes, o carga una nueva en caso contrario.
    selected_label = st.sidebar.selectbox("Modelo", list(available_models.keys()))
    info = available_models[selected_label]
    checkpoint_path = info["path"]

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
    # Tope de paneles Grad-CAM. Se explican las patologías detectadas (prob ≥ umbral);
    # si hay menos de min_panels, se completa hasta ese mínimo; nunca más que este tope.
    # Cada panel añade una pasada de Grad-CAM (coste en CPU).
    min_panels = min(5, len(labels))
    max_panels = st.sidebar.slider(
        "Máximo de paneles (Grad-CAM)",
        min_value=min_panels, max_value=len(labels),
        value=min(8, len(labels)), step=1,
        help=(
            "Se muestran las patologías detectadas (prob ≥ umbral); si hay menos de "
            f"{min_panels}, se completa hasta {min_panels}. Este valor es el tope máximo."
        ),
    )

    if st.sidebar.button("Limpiar historial", use_container_width=True):
        st.session_state.history = []

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
    # PASO 6: EXPLICABILIDAD — original (izq.) + Grad-CAM (der.) por patología
    # ==================================================================
    # Se explican las patologías DETECTADAS (prob ≥ umbral); si hay menos de min_panels se
    # completa hasta ese mínimo con las siguientes más probables, sin superar max_panels.
    # Como las detectadas son siempre las de mayor probabilidad, basta tomar las n_show
    # primeras del orden descendente. argsort ascendente + inversión = orden descendente.
    n_show = min(max(len(detected), min_panels), max_panels, len(labels))
    orden = np.argsort(probs)[::-1][:n_show]
    with st.spinner(f"Generando {len(orden)} mapas Grad-CAM…"):
        panels = []
        for idx in orden:
            # ClassifierOutputTarget necesita un índice entero, no el nombre de la clase.
            mask = _compute_grad_cam(model, tensor, target_layers, int(idx), device)
            # show_cam_on_image usa un colormap jet (azul→rojo): el rojo marca las zonas
            # que más influyeron en la predicción de esa patología.
            heat = show_cam_on_image(img_np, mask, use_rgb=True)  # (H, W, 3) uint8
            panels.append({
                "label": labels[int(idx)],
                "prob": float(probs[int(idx)]),
                "heatmap": heat,
            })

    st.divider()
    st.subheader(f"Explicabilidad visual — {len(panels)} patologías")
    st.caption(
        f"Patologías detectadas (prob ≥ umbral), con un mínimo de {min_panels}. "
        "Izquierda: radiografía original. Derecha: mapa de calor Grad-CAM. "
        "🔴 rojo = mayor influencia en la predicción · 🔵 azul = menor influencia."
    )
    for p in panels:
        st.markdown(f"**{p['label']}** — {p['prob'] * 100:.1f}%")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, use_container_width=True, caption="Original")
        with col2:
            st.image(p["heatmap"], use_container_width=True, caption="Grad-CAM")

    # ==================================================================
    # PASO 7: PROBABILIDADES POR PATOLOGÍA
    # ==================================================================
    st.divider()
    st.subheader("Probabilidades por patología")

    df = pd.DataFrame({
        "Patología": labels,
        "Probabilidad": probs,
        "Detectada": probs >= threshold,
    })
    # Ordenar de mayor a menor probabilidad para que las patologías más probables
    # aparezcan primero y sean más fáciles de identificar visualmente.
    df_sorted = df.sort_values("Probabilidad", ascending=False).reset_index(drop=True)

    # Gráfico de barras horizontal con Altair.
    # Se usa Altair (no matplotlib ni plotly) porque es la librería de gráficos
    # nativa de Streamlit: no requiere instalación extra y se renderiza en SVG
    # (vectorial, escalable sin pérdida de calidad).
    # El color verde indica "detectada" (por encima del umbral) y gris "no detectada".
    # Codificación común (eje X probabilidad, eje Y patología ordenada) compartida por
    # las barras y por las etiquetas de porcentaje superpuestas.
    base = alt.Chart(df_sorted).encode(
        x=alt.X("Probabilidad:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilidad"),
        y=alt.Y("Patología:N", sort="-x", title=None),
    )
    barras = base.mark_bar().encode(
        color=alt.condition(
            alt.datum.Detectada,
            alt.value("#28a745"),    # verde Bootstrap: patología detectada
            alt.value("#6c757d"),    # gris Bootstrap: no detectada
        ),
        tooltip=["Patología", alt.Tooltip("Probabilidad:Q", format=".2%")],
    )
    # Etiqueta con el porcentaje al final de cada barra. El formato ".1%" multiplica por
    # 100 y añade el símbolo (p. ej. 0.608 -> "60.8%").
    etiquetas = base.mark_text(align="left", baseline="middle", dx=3).encode(
        text=alt.Text("Probabilidad:Q", format=".1%"),
    )
    chart = (barras + etiquetas).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

    # Tabla numérica detallada, colapsada por defecto para no saturar la interfaz.
    with st.expander("Ver tabla de valores"):
        df_display = df_sorted.copy()
        df_display["Detectada"] = df_display["Detectada"].map({True: "✓", False: ""})

        def _color_row(row: pd.Series) -> list[str]:
            # Colorear toda la fila en verde si la patología fue detectada,
            # en gris claro en caso contrario, para facilitar la lectura rápida.
            color = "#d4edda" if row["Detectada"] == "✓" else "#f8f9fa"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_display.style.apply(_color_row, axis=1).format({"Probabilidad": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
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
