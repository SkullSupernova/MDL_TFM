"""
API REST de inferencia para clasificación multietiqueta de patologías torácicas.

Expone dos endpoints:
- GET  /health  — verificación de estado del servicio.
- POST /predict — inferencia sobre una imagen de radiografía (JPEG/PNG).

El modelo se carga una sola vez al arrancar el servidor mediante el contexto
de ciclo de vida (lifespan). La ruta del checkpoint y el backbone se leen de
config/config.yml.
"""

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from src.logging_config import get_logger
from src.models import get_pathology_labels, load_checkpoint

logger = get_logger(__name__)

# Ruta fija al archivo de configuración.
# Al ejecutar con uvicorn o docker, el directorio de trabajo es siempre /app (Docker)
# o la raíz del proyecto (local), por lo que esta ruta relativa funciona en ambos casos.
_CONFIG_PATH = "config/config.yml"

# Variables globales del módulo que se inicializan en el lifespan de FastAPI.
# Se usan variables globales (en lugar de inyección de dependencias) porque el modelo
# es un recurso pesado que solo debe cargarse una vez y compartirse entre todas las
# peticiones concurrentes sin coste adicional.
_model: torch.nn.Module | None = None
_cfg: dict = {}
_labels: list[str] = []
_device: torch.device = torch.device("cpu")  # la API siempre corre en CPU

# Transformación de evaluación: igual que en entrenamiento pero sin data augmentation.
# Los parámetros de normalización [0.485, 0.456, 0.406] y [0.229, 0.224, 0.225]
# son la media y desviación estándar de ImageNet por canal RGB. Usar los mismos
# valores que en entrenamiento es imprescindible: si no, la distribución de entrada
# del modelo sería diferente a la que vio durante el aprendizaje.
_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # tamaño esperado por los backbones preentrenados
    transforms.ToTensor(),                   # PIL Image [0,255] → Tensor [0.0, 1.0]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict) -> tuple[torch.nn.Module, int]:
    checkpoint = cfg["model"]["checkpoint_path"]
    # Validar la existencia del checkpoint antes de intentar cargarlo para dar
    # un mensaje de error claro. torch.load lanzaría un FileNotFoundError genérico.
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: '{checkpoint}'. "
            "Entrena el modelo primero o actualiza 'model.checkpoint_path' en config.yml."
        )
    return load_checkpoint(cfg, checkpoint, _device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El lifespan es el mecanismo de FastAPI para ejecutar código al arrancar
    # y al detener la aplicación. Todo lo que está ANTES del 'yield' se ejecuta
    # al inicio; todo lo que está DESPUÉS se ejecuta al apagar.
    #
    # Cargamos el modelo aquí (y no en el endpoint) porque:
    #   1. Cargar un modelo PyTorch tarda varios segundos: hacerlo en cada petición
    #      haría la API inutilizable.
    #   2. El modelo ocupa cientos de MB en RAM: solo debe existir una instancia.
    #   3. Si el checkpoint no existe, el servidor arranca con un error descriptivo
    #      en lugar de fallar silenciosamente en la primera petición.
    global _model, _cfg, _labels
    _cfg = _load_config()
    logger.info(f"Cargando modelo: {_cfg['model']['name']}")
    _model, num_classes = _load_model(_cfg)
    _labels = get_pathology_labels(num_classes)
    logger.info(f"Modelo listo: {_cfg['model']['name']} — {len(_labels)} clases")
    yield
    # Al apagar: liberar la referencia al modelo para que el garbage collector
    # pueda recuperar la memoria (especialmente relevante si el servidor se
    # recarga en caliente sin reiniciar el proceso).
    _model = None
    logger.info("Servidor detenido, modelo liberado")


app = FastAPI(
    title="CheXpert Pathology Classifier",
    description="Inferencia multietiqueta de patologías torácicas sobre radiografías AP/PA.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Verifica que el servicio está activo.

    Respuesta: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Imagen de radiografía (JPEG o PNG)"),
    threshold: float = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Umbral de clasificación (0–1). Por defecto usa el valor de config.yml.",
    ),
) -> JSONResponse:
    """
    Realiza inferencia multietiqueta sobre una radiografía torácica.

    Parámetros:
        file      — Imagen en formato JPEG o PNG (multipart/form-data).
        threshold — Umbral de decisión (0.0–1.0). Si se omite, usa training.threshold de config.yml.

    Respuesta exitosa (200):
        {
          "threshold": 0.5,
          "probabilities": {"No Finding": 0.12, "Edema": 0.87, ...},
          "detected_pathologies": ["Edema"]
        }

    Errores:
        422 — Tipo de archivo no soportado o imagen corrupta.
        503 — Modelo no disponible (fallo al cargar en el arranque).
    """
    # 1. Verificar que el modelo está disponible.
    #    Puede ser None si el lifespan falló al cargar el checkpoint.
    if _model is None:
        logger.error("Petición rechazada: modelo no disponible")
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    # 2. Validar el tipo de archivo antes de leerlo.
    #    Rechazamos tipos no soportados aquí (y no después de leer el contenido)
    #    para evitar transferencias innecesarias de datos grandes.
    if file.content_type not in ("image/jpeg", "image/png"):
        logger.warning(f"Tipo de archivo no soportado: {file.content_type} ({file.filename})")
        raise HTTPException(
            status_code=422,
            detail=f"Tipo de archivo no soportado: '{file.content_type}'. Se requiere image/jpeg o image/png.",
        )

    logger.info(f"Petición recibida: {file.filename} ({file.content_type})")

    # 3. Leer el contenido del archivo y decodificarlo como imagen.
    #    BytesIO permite tratar los bytes como un fichero en memoria sin
    #    guardarlo en disco, lo que es más rápido y no requiere gestión de archivos.
    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        logger.warning(f"No se pudo decodificar la imagen: {file.filename}")
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen.")

    # 4. Preprocesar la imagen y añadir la dimensión de batch.
    #    El modelo espera un tensor de forma [batch_size, 3, 224, 224].
    #    unsqueeze(0) convierte [3, 224, 224] en [1, 3, 224, 224].
    tensor = _eval_transform(img).unsqueeze(0).to(_device)

    # 5. Ejecutar la inferencia.
    #    inference_mode es más eficiente que no_grad porque también deshabilita
    #    la creación del grafo de autodiferenciación y algunas comprobaciones
    #    intermedias. Se usa solo para inferencia, nunca durante el entrenamiento.
    with torch.inference_mode():
        logits = _model(tensor)
        # sigmoid convierte los logits ilimitados a probabilidades [0, 1].
        # squeeze(0) elimina la dimensión de batch: [1, num_classes] → [num_classes].
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    # 6. Aplicar el umbral de detección y construir la respuesta.
    #    Si no se proporcionó umbral en la petición, se usa el valor de config.yml.
    #    Esto permite que los clientes usen su propio umbral sin modificar la config.
    thr = threshold if threshold is not None else _cfg["training"]["threshold"]
    probabilities: Dict[str, float] = {
        label: round(prob, 4)
        for label, prob in zip(_labels, probs)
    }
    detected: List[str] = [
        label for label, prob in probabilities.items() if prob >= thr
    ]

    logger.info(
        f"Resultado: {len(detected)}/{len(_labels)} patologías detectadas "
        f"(umbral={thr:.2f}, archivo={file.filename})"
    )

    return JSONResponse(content={
        "threshold": thr,
        "probabilities": probabilities,
        "detected_pathologies": detected,
    })
