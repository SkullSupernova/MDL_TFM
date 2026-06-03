"""
Pre-redimensionado del dataset CheXpert a tamaño fijo (por defecto 224x224).

El cuello de botella del entrenamiento en Windows (num_workers=0) es decodificar y
redimensionar los JPEG originales en caliente en cada época. Generar una copia ya
redimensionada del dataset elimina ese coste sin recurrir a multiprocessing: el
DataLoader solo decodifica imágenes pequeñas. La copia conserva la estructura de
directorios original, por lo que basta apuntar `data.images_root` / `data.test_images_root`
a las nuevas raíces para usarla.

El redimensionado a (size, size) reproduce exactamente la transformación Resize((224,224))
del pipeline (que deforma la relación de aspecto); la augmentation se sigue aplicando en
caliente sobre la imagen ya pequeña, sin pérdida de comportamiento.

Uso:
    python -m src.preprocess_resize                       # train (batches) + valid, a *_224
    python -m src.preprocess_resize --skip-valid
    python -m src.preprocess_resize --out-root D:/ruta_224 --size 224 --limit 50
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

from src.logging_config import get_logger

logger = get_logger(__name__)

# Pillow >= 9.1 expone el filtro vía enum Resampling; el alias antiguo sigue disponible
# pero está deprecado. Se resuelve una vez para no depender de la versión instalada.
try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _RESAMPLE = Image.LANCZOS


def resize_tree(
    src_root: str, dst_root: str, size: int = 224,
    quality: int = 95, limit: Optional[int] = None, overwrite: bool = False
) -> dict:
    """
    Redimensiona todos los .jpg bajo src_root y los escribe en dst_root preservando
    la ruta relativa. Idempotente: omite los que ya existen salvo overwrite=True.

    Parámetros
    ----------
    src_root : str
        Raíz a recorrer recursivamente en busca de imágenes .jpg.
    dst_root : str
        Raíz de salida; se crea la estructura de subdirectorios necesaria.
    size : int
        Lado del cuadrado de salida (size x size).
    quality : int
        Calidad JPEG de salida (1-95). 95 minimiza la pérdida por recompresión.
    limit : int, optional
        Máximo de imágenes a procesar (para pruebas rápidas). None = todas.
    overwrite : bool
        Si False, no reescribe imágenes ya presentes en dst_root.

    Devuelve
    --------
    dict con conteos: procesadas, omitidas, errores, total.
    """
    src = Path(src_root)
    dst = Path(dst_root)
    if not src.exists():
        raise FileNotFoundError(f"Raíz de origen no encontrada: {src}")

    procesadas = omitidas = errores = 0
    t0 = time.time()
    for i, ruta in enumerate(src.rglob("*.jpg")):
        if limit is not None and (procesadas + omitidas) >= limit:
            break
        destino = dst / ruta.relative_to(src)
        if destino.exists() and not overwrite:
            omitidas += 1
            continue
        try:
            destino.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(ruta) as img:
                # JPEG no admite modos como 'P'/'RGBA'; el Dataset convierte a RGB al
                # cargar, por lo que conservar el modo nativo (típicamente 'L') es válido
                # y produce ficheros más pequeños.
                if img.mode not in ("L", "RGB"):
                    img = img.convert("RGB")
                img.resize((size, size), _RESAMPLE).save(destino, "JPEG", quality=quality)
            procesadas += 1
        except Exception as exc:  # imagen corrupta o ilegible: se registra y se continúa
            errores += 1
            logger.warning(f"No se pudo procesar {ruta}: {exc}")

        if (procesadas + omitidas) % 2000 == 0 and (procesadas + omitidas) > 0:
            ips = procesadas / max(time.time() - t0, 1e-6)
            logger.info(
                f"{src.name}: {procesadas} procesadas, {omitidas} omitidas "
                f"({ips:.0f} img/s)"
            )

    total = procesadas + omitidas
    logger.info(
        f"{src.name} -> {dst}: {procesadas} procesadas, {omitidas} omitidas, "
        f"{errores} errores ({total} total) en {time.time() - t0:.1f}s"
    )
    return {"procesadas": procesadas, "omitidas": omitidas, "errores": errores, "total": total}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-redimensionado del dataset CheXpert")
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument(
        "--out-root", default=None,
        help="Raíz de salida del train. Por defecto: <images_root>_<size>."
    )
    parser.add_argument(
        "--out-valid-root", default=None,
        help="Raíz de salida del valid/test. Por defecto: <test_images_root>_<size>."
    )
    parser.add_argument("--size", type=int, default=None, help="Lado del cuadrado de salida.")
    parser.add_argument("--quality", type=int, default=95, help="Calidad JPEG (1-95).")
    parser.add_argument("--limit", type=int, default=None, help="Máx. imágenes por árbol (pruebas).")
    parser.add_argument("--overwrite", action="store_true", help="Reescribir imágenes existentes.")
    parser.add_argument("--skip-train", action="store_true", help="No procesar los batches de train.")
    parser.add_argument("--skip-valid", action="store_true", help="No procesar el conjunto valid/test.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    size = args.size or cfg["data"].get("img_size", 224)
    images_root = cfg["data"]["images_root"]
    test_images_root = cfg["data"]["test_images_root"]
    out_root = args.out_root or f"{images_root}_{size}"
    out_valid_root = args.out_valid_root or f"{test_images_root}_{size}"

    if not args.skip_train:
        for batch in cfg["data"]["batches"]:
            resize_tree(
                str(Path(images_root) / batch), str(Path(out_root) / batch),
                size=size, quality=args.quality, limit=args.limit, overwrite=args.overwrite,
            )

    if not args.skip_valid:
        resize_tree(
            test_images_root, out_valid_root,
            size=size, quality=args.quality, limit=args.limit, overwrite=args.overwrite,
        )

    logger.info("Pre-redimensionado completado. Apunta config.yml a las nuevas raíces:")
    logger.info(f"  data.images_root: \"{out_root}\"")
    logger.info(f"  data.test_images_root: \"{out_valid_root}\"")


if __name__ == "__main__":
    main()
