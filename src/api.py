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

from src.models import get_pathology_labels, load_checkpoint

_CONFIG_PATH = "config/config.yml"

_model: torch.nn.Module | None = None
_cfg: dict = {}
_labels: list[str] = []
_device: torch.device = torch.device("cpu")
_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict) -> tuple[torch.nn.Module, int]:
    checkpoint = cfg["model"]["checkpoint_path"]
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: '{checkpoint}'. "
            "Entrena el modelo primero o actualiza 'model.checkpoint_path' en config.yml."
        )
    return load_checkpoint(cfg, checkpoint, _device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _cfg, _labels
    _cfg = _load_config()
    _model, num_classes = _load_model(_cfg)
    _labels = get_pathology_labels(num_classes)
    yield
    _model = None


app = FastAPI(
    title="CheXpert Pathology Classifier",
    description="Inferencia multietiqueta de patologías torácicas sobre radiografías AP/PA.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, str]:
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
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=422,
            detail=f"Tipo de archivo no soportado: '{file.content_type}'. Se requiere image/jpeg o image/png.",
        )

    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen.")

    tensor = _eval_transform(img).unsqueeze(0).to(_device)

    with torch.inference_mode():
        logits = _model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    thr = threshold if threshold is not None else _cfg["training"]["threshold"]
    probabilities: Dict[str, float] = {
        label: round(prob, 4)
        for label, prob in zip(_labels, probs)
    }
    detected: List[str] = [
        label for label, prob in probabilities.items() if prob >= thr
    ]

    return JSONResponse(content={
        "threshold": thr,
        "probabilities": probabilities,
        "detected_pathologies": detected,
    })
