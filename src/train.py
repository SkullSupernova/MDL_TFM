# src/train.py
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import amp

from src.utils import EarlyStopping, ModelCheckpoint, calculate_metrics
from src.logging_config import get_logger

logger = get_logger(__name__)


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    device: torch.device,
    save_path: str = "models/mejor_modelo_chexpert.pth"
) -> Tuple[Dict, torch.nn.Module]:
    """
    Bucle principal de entrenamiento para clasificación multietiqueta con CheXpert.

    Implementa Automatic Mixed Precision (AMP), monitorización por lotes,
    EarlyStopping y ModelCheckpoint. Al finalizar, restaura los pesos del
    mejor punto de validación y los guarda en disco.

    Parámetros
    ----------
    model : torch.nn.Module
        Arquitectura a entrenar.
    train_loader : DataLoader
        Iterador del conjunto de entrenamiento.
    val_loader : DataLoader
        Iterador del conjunto de validación.
    criterion : torch.nn.Module
        Función de pérdida (p. ej. BCEWithLogitsLoss).
    optimizer : torch.optim.Optimizer
        Optimizador (p. ej. AdamW).
    scheduler : torch.optim.lr_scheduler
        Planificador de tasa de aprendizaje.
    num_epochs : int
        Número máximo de épocas de entrenamiento.
    device : torch.device
        Dispositivo de cómputo (CPU o CUDA).
    save_path : str
        Ruta donde se guardarán los pesos del mejor modelo.

    Devuelve
    --------
    history : dict
        Historial de métricas por época: train_loss, val_loss, val_acc, val_f1.
    model : torch.nn.Module
        Modelo con los pesos restaurados al mejor punto de validación.
    """
    early_stopping = EarlyStopping(patience=6)
    model_checkpoint = ModelCheckpoint()

    # AMP solo disponible en CUDA; en CPU se usa autocast con device_type='cpu'
    device_type = device.type
    scaler = amp.GradScaler(device_type) if device_type == 'cuda' else None

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    model = model.to(device)
    logger.info(f"Inicio de entrenamiento en: {str(device).upper()}")
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Época {epoch + 1}/{num_epochs} | LR: {current_lr:.6f}")

        # =================== FASE DE ENTRENAMIENTO ===================
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with amp.autocast(device_type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                logger.info(f"  [Batch {i + 1:04d}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)

        # =================== FASE DE VALIDACIÓN ===================
        model.eval()
        val_running_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                with amp.autocast(device_type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)
        y_true = np.vstack(all_labels)
        y_pred = np.vstack(all_preds)
        metrics = calculate_metrics(y_true, y_pred)

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1_macro'])

        scheduler.step(val_epoch_loss)

        # =================== CONTROL (CALLBACKS) ===================
        logger.info(
            f"Época {epoch + 1} — Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['f1_macro']:.4f}"
        )

        saved = model_checkpoint(model, metrics['f1_macro'])
        if saved:
            logger.info(f"  Nuevo mejor modelo — F1: {metrics['f1_macro']:.4f}")

        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            logger.warning(f"Early Stopping activado en época {epoch + 1}.")
            break

    total_time = (time.time() - start_time) / 60
    logger.info(f"Entrenamiento finalizado en {total_time:.2f} minutos.")

    if model_checkpoint.best_model_state:
        model.load_state_dict(model_checkpoint.best_model_state)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Pesos restaurados al mejor F1 y guardados en: {save_path}")

    return history, model
