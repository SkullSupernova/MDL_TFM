"""
Registro de modelos y gate de promoción del "mejor modelo".

Mantiene dos artefactos para que un nuevo entrenamiento solo reemplace al modelo de
producción cuando mejora de forma objetiva, y para poder auditar el proceso:

- Registro del campeón (`models/best_model_registry.json`): por backbone, las métricas
  del modelo actualmente en producción y sus hiperparámetros. Es la referencia contra
  la que se compara cada nuevo entrenamiento.
- Historial de experimentos (`logs/experiments.jsonl`): una línea JSON por entrenamiento
  real (append-only), para auditoría posterior.

Criterio de promoción para multietiqueta CheXpert: AUROC media de las 5 patologías
oficiales sobre el test silver (clave 'auroc_chexpert5'), con F1-macro como desempate
dentro de un margen 'min_delta'.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)

RUTA_REGISTRO = Path("models/best_model_registry.json")
RUTA_HISTORIAL = Path("logs/experiments.jsonl")

_METRICA_PRIMARIA = "auroc_chexpert5"
_METRICA_DESEMPATE = "f1_macro"


def cargar_registro(backbone: str, ruta: Path = RUTA_REGISTRO) -> Optional[dict]:
    """Devuelve el registro del campeón para un backbone, o None si no existe."""
    if not Path(ruta).exists():
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(backbone)


def es_mejor(nuevas: Dict, campeon: Optional[Dict], min_delta: float = 0.0) -> bool:
    """
    Decide si las métricas de test 'nuevas' superan a las del 'campeon'.

    AUROC CheXpert-5 como criterio primario; si la diferencia cae dentro de min_delta
    (empate técnico), desempata por F1-macro. Si no hay campeón previo, devuelve True.
    Si alguna AUROC primaria no es evaluable (None), decide por el desempate (F1).

    Ejemplo
    -------
    >>> es_mejor({'auroc_chexpert5': 0.82, 'f1_macro': 0.5},
    ...          {'auroc_chexpert5': 0.80, 'f1_macro': 0.6}, min_delta=0.005)
    True
    """
    if campeon is None:
        return True
    p_new = nuevas.get(_METRICA_PRIMARIA)
    p_old = campeon.get(_METRICA_PRIMARIA)
    if p_new is None or p_old is None:
        return nuevas.get(_METRICA_DESEMPATE, 0.0) > campeon.get(_METRICA_DESEMPATE, 0.0)
    if p_new > p_old + min_delta:
        return True
    if p_new < p_old - min_delta:
        return False
    # Empate técnico dentro de min_delta: desempatar por F1-macro.
    return nuevas.get(_METRICA_DESEMPATE, 0.0) > campeon.get(_METRICA_DESEMPATE, 0.0)


def guardar_registro(backbone: str, registro: dict, ruta: Path = RUTA_REGISTRO) -> None:
    """Actualiza (o crea) la entrada del backbone en el registro de campeones."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if Path(ruta).exists():
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[backbone] = registro
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def registrar_experimento(registro: dict, ruta: Path = RUTA_HISTORIAL) -> None:
    """Añade una línea JSON con el experimento al historial (append-only)."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def promover(candidato: str, produccion: str) -> None:
    """Reemplaza el checkpoint de producción por el candidato (mover atómico)."""
    os.replace(candidato, produccion)


def descartar(candidato: str) -> None:
    """Elimina un checkpoint candidato no promovido, si existe."""
    if os.path.exists(candidato):
        os.remove(candidato)
