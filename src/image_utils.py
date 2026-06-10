"""
Utilidades de imagen para la app web, independientes de Streamlit (testeable):

- `validar_imagen_radiografia`: comprobación en el límite de entrada (tamaño, resolución y
  si la imagen parece una radiografía en escala de grises).
- `empaquetar_imagenes_zip`: empaqueta la radiografía original y los mapas de calor en un ZIP.
"""

import re
import zipfile
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Umbral de "color": diferencia media entre canales RGB (0-255) por encima de la cual
# se considera que la imagen no es una radiografía monocroma. Una imagen en escala de
# grises convertida a RGB tiene canales idénticos (diferencia 0).
_UMBRAL_COLOR = 12.0


def validar_imagen_radiografia(
    img: Image.Image,
    n_bytes: Optional[int] = None,
    lado_min: int = 64,
    max_mb: float = 10.0,
) -> Dict:
    """
    Valida una imagen de entrada antes de la inferencia.

    Parámetros
    ----------
    img : PIL.Image.Image
        Imagen ya abierta.
    n_bytes : int, optional
        Tamaño del fichero en bytes (p. ej. `uploaded.size`); si se aporta, se limita a max_mb.
    lado_min : int
        Lado mínimo aceptable en píxeles (ancho y alto).
    max_mb : float
        Tamaño máximo del fichero en megabytes.

    Devuelve
    --------
    dict con:
        - "ok" (bool): True si no hay errores bloqueantes.
        - "errores" (list[str]): motivos por los que la imagen no debe procesarse.
        - "avisos" (list[str]): advertencias no bloqueantes (p. ej. imagen en color).

    Ejemplo
    -------
    >>> validar_imagen_radiografia(Image.new("L", (224, 224)))["ok"]
    True
    """
    errores: List[str] = []
    avisos: List[str] = []

    if n_bytes is not None and n_bytes > max_mb * 1024 * 1024:
        errores.append(
            f"La imagen pesa {n_bytes / (1024 * 1024):.1f} MB; el máximo admitido es {max_mb:.0f} MB."
        )

    ancho, alto = img.size
    if ancho < lado_min or alto < lado_min:
        errores.append(
            f"Resolución {ancho}×{alto} px demasiado baja; mínimo {lado_min}×{lado_min} px."
        )

    # ¿parece radiografía? Las RX son monocromas: al pasarlas a RGB, los tres canales
    # son casi idénticos. Una diferencia media alta indica imagen en color (aviso, no error).
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    if arr.size:
        spread = float(
            np.mean(np.abs(arr[..., 0] - arr[..., 1])) + np.mean(np.abs(arr[..., 1] - arr[..., 2]))
        ) / 2
        if spread > _UMBRAL_COLOR:
            avisos.append(
                "La imagen parece estar en color; las radiografías suelen ser en escala de grises. "
                "Los resultados podrían no ser fiables."
            )

    return {"ok": not errores, "errores": errores, "avisos": avisos}


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _slug(texto: str) -> str:
    """Convierte una etiqueta en un nombre de fichero seguro."""
    return re.sub(r"[^A-Za-z0-9]+", "_", texto).strip("_") or "clase"


def empaquetar_imagenes_zip(original: np.ndarray, panels: List[Dict]) -> bytes:
    """
    Empaqueta la radiografía original y los mapas de calor en un ZIP en memoria.

    Parámetros
    ----------
    original : np.ndarray uint8 (H, W, 3)
        Radiografía original (224×224).
    panels : list[dict]
        Cada panel con "label" (str), "heatmap" (np.ndarray uint8, H, W, 3) y, en modo
        comparación, "heatmap_b" (np.ndarray uint8 o None) con el mapa del segundo modelo.

    Devuelve
    --------
    bytes del ZIP. Contiene `original.png` y, por panel, su mapa de calor. Si el panel lleva
    un `heatmap_b` no nulo (modo comparación) se exportan ambos modelos como
    `heatmap_<i>_<label>_A.png` y `heatmap_<i>_<label>_B.png`; en caso contrario un único
    `heatmap_<i>_<label>.png`.
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("original.png", _png_bytes(original))
        for i, p in enumerate(panels, start=1):
            label = _slug(p["label"])
            heatmap_b = p.get("heatmap_b")
            if heatmap_b is not None:
                zf.writestr(f"heatmap_{i:02d}_{label}_A.png", _png_bytes(p["heatmap"]))
                zf.writestr(f"heatmap_{i:02d}_{label}_B.png", _png_bytes(heatmap_b))
            else:
                zf.writestr(f"heatmap_{i:02d}_{label}.png", _png_bytes(p["heatmap"]))
    return buf.getvalue()
