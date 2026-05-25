# src/visualization.py
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import multilabel_confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from IPython.display import display, Markdown


def graficar_entrenamiento(hist: Optional[Dict]) -> None:
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
        print("No hay historial disponible para graficar.")
        return

    epochs = range(1, len(hist['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, hist['val_loss'], label='Val Loss', marker='s')
    plt.title('Evolución de la Pérdida (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (BCE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if 'val_acc' in hist and 'val_f1' in hist:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, hist['val_acc'], label='Validation Accuracy', marker='^', color='green')
        plt.plot(epochs, hist['val_f1'], label='Validation F1-Macro', marker='d', color='purple')
        plt.title('Rendimiento en Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Puntuación')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


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

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            for i in range(images.size(0)):
                if images_shown >= n_images:
                    break

                img = images[i].cpu().numpy().transpose((1, 2, 0))
                img = std * img + mean
                img = np.clip(img, 0, 1)

                reales = [classes[j] for j, val in enumerate(labels[i]) if val == 1]
                predichas = [
                    f"{classes[j]} ({probs[i][j]:.2f})"
                    for j, p in enumerate(probs[i]) if p > 0.5
                ]

                plt.subplot(2, 3, images_shown + 1)
                plt.imshow(img, cmap='bone')
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


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> None:
    """
    Genera una cuadrícula de matrices de confusión binarias, una por etiqueta.

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
    rows = (n_labels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.ravel()

    for i in range(n_labels):
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'Matriz: {labels[i]}')
        axes[i].set_xlabel('Predicción')
        axes[i].set_ylabel('Realidad')
        axes[i].set_xticklabels(['Neg', 'Pos'])
        axes[i].set_yticklabels(['Neg', 'Pos'])

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def matriz_resumen_multietiqueta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str]
) -> None:
    """
    Genera un heatmap resumen de sensibilidad, especificidad, tasa de omisión
    y tasa de falsa alarma para cada clase del problema multietiqueta.

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
        fmt='.1%',
        cmap='Blues',
        xticklabels=columnas,
        yticklabels=classes,
        cbar=False,
        linewidths=1,
        linecolor='white'
    )

    for i in range(datos_resumen.shape[0]):
        for j in [1, 3]:
            if datos_resumen[i, j] > 0.2:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='red', alpha=0.2))

    plt.title('Matriz Resumen de Rendimiento Clínico por Patología', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.show()


def generar_auditoria_total(
    loader,
    model: torch.nn.Module,
    classes: List[str],
    device: torch.device
) -> None:
    """
    Ejecuta una auditoría visual de explicabilidad mediante Grad-CAM para cada
    clase, mostrando ejemplos de verdaderos positivos (TP), falsos positivos (FP)
    y falsos negativos (FN).

    Parámetros
    ----------
    loader : DataLoader
        Iterador del conjunto de test.
    model : torch.nn.Module
        Modelo en modo evaluación. Debe exponer model.features para Grad-CAM.
    classes : list of str
        Nombres de las clases en el mismo orden que la salida del modelo.
    device : torch.device
        Dispositivo de cómputo.
    """
    model.eval()

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    ejemplos = {cls: {'TP': None, 'FP': None, 'FN': None} for cls in classes}

    print("Buscando pacientes representativos en el Test Set para Auditoría Visual...")
    with torch.no_grad():
        for images, labels in loader:
            imgs_dev = images.to(device)
            outputs = model(imgs_dev)
            probs = torch.sigmoid(outputs).cpu().numpy()
            reales = labels.numpy()
            preds = (probs > 0.5).astype(float)

            for idx_cls, cls_name in enumerate(classes):
                for i in range(len(images)):
                    if preds[i, idx_cls] == 1 and reales[i, idx_cls] == 1 and ejemplos[cls_name]['TP'] is None:
                        ejemplos[cls_name]['TP'] = (images[i], probs[i, idx_cls])
                    if preds[i, idx_cls] == 1 and reales[i, idx_cls] == 0 and ejemplos[cls_name]['FP'] is None:
                        ejemplos[cls_name]['FP'] = (images[i], probs[i, idx_cls])
                    if preds[i, idx_cls] == 0 and reales[i, idx_cls] == 1 and ejemplos[cls_name]['FN'] is None:
                        ejemplos[cls_name]['FN'] = (images[i], probs[i, idx_cls])

            if all(all(v is not None for v in p.values()) for p in ejemplos.values()):
                break

    for cls_name in classes:
        display(Markdown(f"### Análisis de Explicabilidad (Grad-CAM): `{cls_name}`"))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Evaluación Radiológica: {cls_name.upper()}", fontsize=18, fontweight='bold', y=1.05)

        tipos = [
            ('TP', 'ACIERTO (Verdadero Positivo)'),
            ('FP', 'FALSA ALARMA (Falso Positivo)'),
            ('FN', 'OMISIÓN (Falso Negativo)')
        ]

        for idx, (key, label) in enumerate(tipos):
            if ejemplos[cls_name][key] is not None:
                img_t, conf = ejemplos[cls_name][key]

                img_np = img_t.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

                input_tensor = img_t.unsqueeze(0).to(device)
                # Corrección: se usa el parámetro 'classes' en lugar de la variable
                # global 'cols_patologias_activas' que existía en el notebook original.
                target_idx = classes.index(cls_name)

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(target_idx)]
                )[0, :]
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