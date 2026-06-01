# src/visualization.py
#
# Funciones de visualización para análisis post-entrenamiento: curvas de aprendizaje,
# inspección visual de predicciones, matrices de confusión y auditoría GradCAM.
#
# Todas las funciones de este módulo producen salida gráfica (matplotlib/seaborn).
# Las funciones que usan IPython.display (generar_auditoria_total) solo muestran
# salida en entornos Jupyter; en scripts CLI se renderizan los gráficos pero no el Markdown.

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    multilabel_confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from IPython.display import display, Markdown

from src.logging_config import get_logger

logger = get_logger(__name__)


def _mostrar_o_guardar(save_path: Optional[str]) -> None:
    """Guarda la figura activa en disco si se indica ruta; si no, la muestra (Jupyter)."""
    # En el pipeline CLI / tracker se guarda a PNG (headless); en notebook se muestra.
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def graficar_entrenamiento(hist: Optional[Dict], save_path: Optional[str] = None) -> None:
    """
    Genera gráficas de evolución de pérdida (Loss) y métricas de validación
    (Accuracy y F1-Macro) a partir del historial de entrenamiento.

    Parámetros
    ----------
    hist : dict or None
        Diccionario con claves 'train_loss', 'val_loss', 'val_acc', 'val_f1'.
        Si es None, la función emite un aviso y retorna sin graficar.
    """
    if hist is None:
        logger.warning("No hay historial disponible para graficar.")
        return

    epochs = range(1, len(hist['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Gráfico izquierdo: evolución de la pérdida.
    # La divergencia entre train_loss y val_loss indica sobreajuste: si train_loss
    # sigue bajando mientras val_loss sube o se estanca, el modelo está memorizando
    # el set de entrenamiento. El punto de Early Stopping debería estar cerca de
    # donde val_loss alcanza su mínimo.
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, hist['val_loss'], label='Val Loss', marker='s')
    plt.title('Evolución de la Pérdida (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (BCE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Gráfico derecho: métricas de validación.
    # Se comprueba si las claves existen para que la función sea compatible con
    # historiales parciales (p.ej. si el entrenamiento fue interrumpido antes de
    # calcular acc y f1 en la primera época).
    if 'val_acc' in hist and 'val_f1' in hist:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, hist['val_acc'], label='Validation Accuracy', marker='^', color='green')
        plt.plot(epochs, hist['val_f1'], label='Validation F1-Macro', marker='d', color='purple')
        plt.title('Rendimiento en Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Puntuación')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    # tight_layout ajusta automáticamente el espaciado entre subgráficos para que
    # los títulos y etiquetas no se solapen. Sin él, en figuras con múltiples
    # subplots los labels del eje x del gráfico superior a menudo se solapan
    # con el título del gráfico inferior.
    plt.tight_layout()
    _mostrar_o_guardar(save_path)


def inspeccion_visual(loader, model, classes: List[str], device: torch.device, n_images: int = 6) -> None:
    """
    Muestra una cuadrícula de radiografías con sus etiquetas reales y las
    predicciones del modelo superpuestas.

    Parámetros
    ----------
    loader : DataLoader
        Iterador del conjunto a inspeccionar.
    model : torch.nn.Module
        Modelo en modo evaluación.
    classes : list of str
        Nombres de las clases en el mismo orden que la salida del modelo.
    device : torch.device
        Dispositivo de cómputo.
    n_images : int
        Número de imágenes a mostrar (por defecto 6).
    """
    model.eval()
    images_shown = 0
    plt.figure(figsize=(16, 10))

    # Parámetros de normalización ImageNet usados en el pipeline de entrenamiento.
    # Para visualizar la imagen, es necesario invertir la normalización:
    #   imagen_original = imagen_normalizada * std + mean
    # Sin esta inversión, los píxeles tendrían valores negativos y fuera de [0,1],
    # y matplotlib mostraría colores incorrectos o artefactos de clipping.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            # sigmoid convierte los logits a probabilidades [0,1] por clase.
            probs = torch.sigmoid(outputs).cpu().numpy()

            for i in range(images.size(0)):
                if images_shown >= n_images:
                    break

                # Pasar de formato PyTorch (C, H, W) a matplotlib (H, W, C)
                # e invertir la normalización ImageNet para recuperar la imagen original.
                img = images[i].cpu().numpy().transpose((1, 2, 0))
                img = std * img + mean
                # clip previene valores fuera de [0,1] por errores de punto flotante
                # acumulados en la inversión de la normalización.
                img = np.clip(img, 0, 1)

                reales = [classes[j] for j, val in enumerate(labels[i]) if val == 1]
                predichas = [
                    f"{classes[j]} ({probs[i][j]:.2f})"
                    for j, p in enumerate(probs[i]) if p > 0.5
                ]

                plt.subplot(2, 3, images_shown + 1)
                # cmap='bone' es un colormap en escala de grises con ligero tinte azulado,
                # adecuado para radiografías: preserva los tonos de grises clínicos sin
                # añadir color artificial que podría distorsionar la interpretación.
                plt.imshow(img, cmap='bone')
                # Título azul si la predicción coincide exactamente con la realidad,
                # rojo si hay alguna discrepancia, para identificar errores de un vistazo.
                color = 'blue' if set(reales) == {p.split(' ')[0] for p in predichas} else 'red'
                plt.title(
                    f"REAL: {', '.join(reales) if reales else 'Sano'}\n"
                    f"PRED: {', '.join(predichas) if predichas else 'Sano'}",
                    fontsize=9,
                    color=color
                )
                plt.axis('off')
                images_shown += 1

            if images_shown >= n_images:
                break

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], save_path: Optional[str] = None
) -> None:
    """
    Genera una cuadrícula de matrices de confusión binarias, una por etiqueta.

    En clasificación multietiqueta no existe una única matriz de confusión:
    cada clase es un problema binario independiente (presente / ausente), y
    multilabel_confusion_matrix devuelve un array de forma (n_clases, 2, 2)
    donde mcm[i] es la matriz binaria de la clase i.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales de forma (n_muestras, n_clases).
    y_pred : np.ndarray
        Predicciones binarias de forma (n_muestras, n_clases).
    labels : list of str
        Nombres de las clases.
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    n_labels = len(labels)
    cols = 3
    # Calcular el número de filas necesario para acomodar todas las clases en 3 columnas.
    # (n_labels + cols - 1) // cols es el equivalente entero de ceil(n_labels / cols).
    rows = (n_labels + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    # ravel() convierte la matriz 2D de axes en un array 1D para indexar con un solo
    # índice en el bucle, evitando tener que gestionar índices de fila y columna por separado.
    axes = axes.ravel()

    for i in range(n_labels):
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'Matriz: {labels[i]}')
        axes[i].set_xlabel('Predicción')
        axes[i].set_ylabel('Realidad')
        axes[i].set_xticklabels(['Neg', 'Pos'])
        axes[i].set_yticklabels(['Neg', 'Pos'])

    # Ocultar los ejes sobrantes si el número de clases no es múltiplo de 3.
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    _mostrar_o_guardar(save_path)


def matriz_resumen_multietiqueta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Genera un heatmap resumen de sensibilidad, especificidad, tasa de omisión
    y tasa de falsa alarma para cada clase del problema multietiqueta.

    Las cuatro métricas clínicas tienen interpretación directa:
    - Sensibilidad (Recall): de todos los enfermos, ¿cuántos detecta el modelo?
      Alta sensibilidad es prioritaria en cribado: mejor alertar de más que omitir.
    - Tasa de omisión (Miss Rate = 1 - Sensibilidad): falsos negativos / todos los positivos reales.
      Crítico en contexto clínico: un FN puede significar no tratar una patología grave.
    - Especificidad: de todos los sanos, ¿cuántos clasifica correctamente como sanos?
      Alta especificidad reduce alarmas innecesarias y pruebas complementarias costosas.
    - Tasa de falsa alarma (Fall-out = 1 - Especificidad): falsos positivos / todos los negativos reales.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales de forma (n_muestras, n_clases).
    y_pred : np.ndarray
        Predicciones binarias de forma (n_muestras, n_clases).
    classes : list of str
        Nombres de las clases.
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    datos_resumen = []

    for i in range(len(classes)):
        # ravel() sobre una matriz 2x2 devuelve (TN, FP, FN, TP) en ese orden.
        # El orden es row-major: primero fila 0 (negativos reales: TN, FP),
        # luego fila 1 (positivos reales: FN, TP).
        tn, fp, fn, tp = mcm[i].ravel()
        sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
        tasa_omision = fn / (tp + fn) if (tp + fn) > 0 else 0
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
        falsa_alarma = fp / (tn + fp) if (tn + fp) > 0 else 0
        datos_resumen.append([sensibilidad, tasa_omision, especificidad, falsa_alarma])

    datos_resumen = np.array(datos_resumen)

    columnas = [
        'Sensibilidad\n(Detecta al enfermo)',
        'Tasa Omisión\n(Falso Negativo)',
        'Especificidad\n(Descarta al sano)',
        'Falsa Alarma\n(Falso Positivo)'
    ]

    plt.figure(figsize=(10, len(classes) * 0.8))
    ax = sns.heatmap(
        datos_resumen,
        annot=True,
        fmt='.1%',       # formato porcentaje con 1 decimal para legibilidad clínica
        cmap='Blues',
        xticklabels=columnas,
        yticklabels=classes,
        cbar=False,
        linewidths=1,
        linecolor='white'
    )

    # Resaltar en rojo las celdas de métricas negativas (tasa omisión y falsa alarma)
    # que superan el 20%. Este umbral es arbitrario pero clínicamente útil como
    # alerta visual: más del 20% de omisiones o falsas alarmas es preocupante.
    # Las columnas 1 (tasa_omision) y 3 (falsa_alarma) son las métricas de error.
    for i in range(datos_resumen.shape[0]):
        for j in [1, 3]:
            if datos_resumen[i, j] > 0.2:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='red', alpha=0.2))

    plt.title('Matriz Resumen de Rendimiento Clínico por Patología', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    _mostrar_o_guardar(save_path)


def plot_roc_curves(
    y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], save_path: Optional[str] = None
) -> None:
    """
    Dibuja las curvas ROC por clase (una línea por patología evaluable) con su AUROC.

    Solo se trazan las clases con positivos y negativos en y_true; las no evaluables
    (p. ej. Fracture en el test silver) se omiten del gráfico. Pensada para guardarse
    a disco como artefacto del experimento.
    """
    plt.figure(figsize=(9, 8))
    for i, lab in enumerate(labels):
        col = y_true[:, i]
        if col.min() == col.max():
            continue
        fpr, tpr, _ = roc_curve(col, y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc_score(col, y_prob[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
    plt.title('Curvas ROC por patología')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    _mostrar_o_guardar(save_path)


def plot_pr_curves(
    y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], save_path: Optional[str] = None
) -> None:
    """
    Dibuja las curvas precisión-recall por clase con su PR-AUC (average precision).

    Más informativas que ROC bajo desbalanceo. Se omiten las clases sin positivos.
    """
    plt.figure(figsize=(9, 8))
    for i, lab in enumerate(labels):
        col = y_true[:, i]
        if col.max() == 0:
            continue
        precision, recall, _ = precision_recall_curve(col, y_prob[:, i])
        plt.plot(recall, precision, label=f"{lab} (AP={average_precision_score(col, y_prob[:, i]):.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.title('Curvas Precisión-Recall por patología')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    _mostrar_o_guardar(save_path)


def generar_auditoria_total(
    loader,
    model: torch.nn.Module,
    classes: List[str],
    device: torch.device
) -> None:
    """
    Ejecuta una auditoría visual de explicabilidad mediante GradCAM para cada
    clase, mostrando ejemplos de verdaderos positivos (TP), falsos positivos (FP)
    y falsos negativos (FN).

    Para cada clase se busca un ejemplo representativo de cada tipo de error en
    el conjunto de test. Si no existe un tipo (p.ej. sin FP para una clase muy
    específica), el panel correspondiente queda vacío con un mensaje explicativo.

    Nota: esta función usa IPython.display.display(Markdown(...)) para los títulos.
    En scripts CLI los títulos no se mostrarán, pero los gráficos matplotlib sí.

    Parámetros
    ----------
    loader : DataLoader
        Iterador del conjunto de test.
    model : torch.nn.Module
        Modelo en modo evaluación. Debe exponer model.features para GradCAM.
    classes : list of str
        Nombres de las clases en el mismo orden que la salida del modelo.
    device : torch.device
        Dispositivo de cómputo.
    """
    model.eval()

    # model.features[-1] es la última capa convolucional del DenseNet-121.
    # GradCAM necesita los gradientes que fluyen hacia esta capa para calcular
    # la importancia de cada región espacial. Ver src/models.py:get_grad_cam_layer()
    # para la justificación de la elección de la capa target por backbone.
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Inicializar el diccionario de ejemplos: para cada clase, se busca un ejemplo
    # de TP, FP y FN. El valor None indica que aún no se ha encontrado.
    ejemplos = {cls: {'TP': None, 'FP': None, 'FN': None} for cls in classes}

    # Primera pasada: recopilar ejemplos representativos de cada tipo de error.
    # Se usa torch.no_grad() porque en esta fase solo se necesitan las probabilidades
    # para clasificar los ejemplos como TP/FP/FN. GradCAM se ejecuta después,
    # fuera del bloque no_grad(), porque necesita gradientes activos.
    logger.info("Buscando casos representativos de TP/FP/FN en el test set...")
    with torch.no_grad():
        for images, labels in loader:
            imgs_dev = images.to(device)
            outputs = model(imgs_dev)
            probs = torch.sigmoid(outputs).cpu().numpy()
            reales = labels.numpy()
            preds = (probs > 0.5).astype(float)

            for idx_cls, cls_name in enumerate(classes):
                for i in range(len(images)):
                    # Guardar el primer ejemplo encontrado de cada tipo.
                    # Se guarda el tensor CPU (no el de GPU) para poder reutilizarlo
                    # fuera del contexto del DataLoader sin mantener el GPU ocupado.
                    if preds[i, idx_cls] == 1 and reales[i, idx_cls] == 1 and ejemplos[cls_name]['TP'] is None:
                        ejemplos[cls_name]['TP'] = (images[i], probs[i, idx_cls])
                    if preds[i, idx_cls] == 1 and reales[i, idx_cls] == 0 and ejemplos[cls_name]['FP'] is None:
                        ejemplos[cls_name]['FP'] = (images[i], probs[i, idx_cls])
                    if preds[i, idx_cls] == 0 and reales[i, idx_cls] == 1 and ejemplos[cls_name]['FN'] is None:
                        ejemplos[cls_name]['FN'] = (images[i], probs[i, idx_cls])

            # Terminar la búsqueda en cuanto se tienen los 3 tipos para todas las clases,
            # evitando iterar el dataset completo innecesariamente.
            if all(all(v is not None for v in p.values()) for p in ejemplos.values()):
                break

    # Segunda pasada: generar GradCAM para cada ejemplo encontrado.
    for cls_name in classes:
        display(Markdown(f"### Análisis de Explicabilidad (Grad-CAM): `{cls_name}`"))
        _, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.suptitle(f"Evaluación Radiológica: {cls_name.upper()}", fontsize=18, fontweight='bold', y=1.05)

        tipos = [
            ('TP', 'ACIERTO (Verdadero Positivo)'),
            ('FP', 'FALSA ALARMA (Falso Positivo)'),
            ('FN', 'OMISIÓN (Falso Negativo)')
        ]

        for idx, (key, label) in enumerate(tipos):
            if ejemplos[cls_name][key] is not None:
                img_t, conf = ejemplos[cls_name][key]

                # Preparar la imagen para visualización: pasar de (C, H, W) a (H, W, C)
                # y normalizar al rango [0, 1] usando min-max del tensor concreto.
                # Nota: esto es diferente a invertir la normalización ImageNet (que usa
                # media y std globales). Aquí se usa min-max local porque show_cam_on_image
                # requiere float en [0, 1] y la inversión ImageNet puede producir valores
                # marginalmente fuera de ese rango por errores de punto flotante.
                img_np = img_t.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

                # GradCAM requiere dimensión de batch: (C, H, W) → (1, C, H, W).
                # A diferencia de la fase de búsqueda (torch.no_grad()), aquí NO se
                # usa no_grad porque GradCAM necesita el grafo de gradientes activo
                # para calcular la importancia espacial de la capa convolucional.
                input_tensor = img_t.unsqueeze(0).to(device)
                # Corrección: se usa el parámetro 'classes' en lugar de la variable
                # global 'cols_patologias_activas' que existía en el notebook original.
                target_idx = classes.index(cls_name)

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(target_idx)]
                )[0, :]  # eliminar la dimensión de batch: (1, H, W) → (H, W)

                # Superponer el mapa de calor GradCAM sobre la imagen original.
                # use_rgb=True porque nuestra imagen ya está en RGB (PyTorch usa RGB,
                # no BGR como OpenCV). Sin este parámetro, los canales se invertirían.
                vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                axes[idx].imshow(vis)
                axes[idx].set_title(
                    f"[{cls_name}]\n{label}\nConfianza de la Red: {conf:.1%}",
                    fontweight='bold',
                    fontsize=12
                )
            else:
                axes[idx].text(
                    0.5, 0.5,
                    f"No hay ejemplos de {key}\npara {cls_name} en Test",
                    ha='center', va='center', fontsize=12
                )
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()
