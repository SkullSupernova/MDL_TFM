# src/train.py
#
# Bucle principal de entrenamiento. Recibe un modelo ya construido y todos los
# componentes necesarios (dataloaders, criterio, optimizador, scheduler) y ejecuta
# el ciclo epoch → train → validate → checkpoint → early stopping.
#
# El módulo es independiente de la arquitectura concreta: no sabe si está
# entrenando un DenseNet o un ResNet, lo que permite reutilizarlo sin cambios.

import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import amp

from src.utils import EarlyStopping, ModelCheckpoint, calculate_metrics, auroc_macro
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
    # Inicializar los callbacks de control del entrenamiento.
    # EarlyStopping detiene el entrenamiento si la pérdida de validación no mejora
    # durante 'patience' épocas consecutivas, evitando el sobreajuste y el tiempo perdido.
    # ModelCheckpoint guarda en memoria (no en disco) los pesos del mejor modelo visto,
    # medido por F1-macro de validación.
    early_stopping = EarlyStopping(patience=6)
    model_checkpoint = ModelCheckpoint()

    # Configurar Automatic Mixed Precision (AMP).
    # AMP usa float16 para la mayor parte de los cálculos y float32 solo donde
    # la precisión es crítica. En CUDA esto reduce el uso de VRAM a la mitad y
    # acelera el entrenamiento ~2x. En CPU no hay beneficio y usamos autocast
    # sin GradScaler (el escalador de gradientes solo es necesario con float16 real).
    device_type = device.type
    scaler = amp.GradScaler(device_type) if device_type == 'cuda' else None

    # Historial de métricas: se devuelve al final para graficar la evolución.
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auroc': []}

    model = model.to(device)
    logger.info(f"Inicio de entrenamiento en: {str(device).upper()}")
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Época {epoch + 1}/{num_epochs} | LR: {current_lr:.6f}")

        # ==================================================================
        # FASE DE ENTRENAMIENTO
        # model.train() activa el Dropout y el modo de estadísticas locales
        # del BatchNorm. Es obligatorio llamarlo antes de cada fase de train
        # porque model.eval() lo desactivará en la fase de validación.
        # ==================================================================
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Limpiar los gradientes del paso anterior antes de calcular los nuevos.
            # set_to_none=True (el defecto en PyTorch moderno) es más eficiente que
            # rellenar con ceros porque evita una operación de escritura en memoria.
            optimizer.zero_grad()

            # Calcular la pérdida dentro del contexto AMP.
            # autocast convierte automáticamente las operaciones a float16 donde
            # es seguro hacerlo (capas lineales, convoluciones) y mantiene float32
            # donde la precisión importa (softmax, normalización).
            with amp.autocast(device_type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backpropagation y actualización de pesos.
            # Con scaler (CUDA + AMP): el escalador amplifica la pérdida antes del
            # backward para evitar underflow en float16, y la revierte antes de
            # actualizar los pesos.
            # Sin scaler (CPU): backprop estándar con float32.
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            # Log de progreso cada 100 batches y al final de la época.
            # Permite monitorizar el entrenamiento en tiempo real sin inundar
            # el log con una línea por cada batch.
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                logger.info(f"  [Batch {i + 1:04d}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)

        # ==================================================================
        # FASE DE VALIDACIÓN
        # model.eval() desactiva el Dropout y usa las estadísticas globales
        # del BatchNorm, garantizando predicciones deterministas.
        # torch.no_grad() desactiva el cálculo del grafo de gradientes, lo que
        # reduce el uso de memoria y acelera la inferencia ~2x.
        # ==================================================================
        model.eval()
        val_running_loss = 0.0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                with amp.autocast(device_type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                # Se guardan las probabilidades (sigmoid) para calcular tanto las
                # predicciones binarias (umbral 0.5) como la AUROC, que es independiente
                # del umbral y se usa para seleccionar el mejor epoch. El umbral
                # personalizado de config.yml se aplica solo en inferencia (api.py y app.py).
                # .float() es necesario: bajo autocast 'outputs' es bfloat16 (CPU) o
                # float16 (GPU), y numpy() no soporta esos tipos.
                all_probs.append(torch.sigmoid(outputs).float().cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)

        # Concatenar todos los batches en matrices únicas para calcular métricas globales.
        y_true = np.vstack(all_labels)
        y_prob = np.vstack(all_probs)
        y_pred = (y_prob > 0.5).astype(np.float32)
        metrics = calculate_metrics(y_true, y_pred)
        val_auroc, _ = auroc_macro(y_true, y_prob)

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1_macro'])
        history['val_auroc'].append(val_auroc)

        # Actualizar el scheduler con la pérdida de validación.
        # ReduceLROnPlateau divide la tasa de aprendizaje cuando la pérdida de
        # validación deja de mejorar durante 'patience' épocas. Esto permite
        # un entrenamiento más fino en las últimas etapas sin requerir un decay fijo.
        scheduler.step(val_epoch_loss)

        # ==================================================================
        # CONTROL DE ENTRENAMIENTO (CALLBACKS)
        # ==================================================================
        logger.info(
            f"Época {epoch + 1} — Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | "
            f"Val F1: {metrics['f1_macro']:.4f} | Val AUROC: {val_auroc:.4f}"
        )

        # Guardar en memoria los pesos si este epoch tiene la mejor AUROC de validación
        # hasta ahora. Se usa AUROC (no F1 a umbral 0.5) porque es independiente del
        # umbral y más robusta al fuerte desbalanceo de clases del dataset.
        saved = model_checkpoint(model, val_auroc)
        if saved:
            logger.info(f"  Nuevo mejor modelo — Val AUROC: {val_auroc:.4f}")

        # Comprobar si se activa el Early Stopping.
        # Si se ha superado la paciencia, se rompe el bucle y se pasa directamente
        # a restaurar los mejores pesos, sin seguir entrenando sobre el sobreajuste.
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            logger.warning(f"Early Stopping activado en época {epoch + 1}.")
            break

    total_time = (time.time() - start_time) / 60
    logger.info(f"Entrenamiento finalizado en {total_time:.2f} minutos.")

    # Restaurar los pesos del mejor epoch y guardarlos en disco.
    # Guardamos state_dict (solo los pesos, no la arquitectura completa) porque
    # es el formato estándar, más ligero y portable que guardar el modelo entero.
    if model_checkpoint.best_model_state:
        model.load_state_dict(model_checkpoint.best_model_state)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Pesos restaurados al mejor F1 y guardados en: {save_path}")

    return history, model
