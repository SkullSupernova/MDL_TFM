# src/utils.py
import os
import re
import copy
import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from IPython.display import display, Markdown


# =========================================================
# Configuración del entorno
# =========================================================

def setup_environment() -> Tuple[torch.device, int]:
    """
    Configura el dispositivo de cómputo y los parámetros del DataLoader.
    Las decisiones priorizan estabilidad sobre paralelización agresiva en Windows.
    """
    print("\n=== Detección de Hardware ===")

    cpu_cores = os.cpu_count()
    if cpu_cores is not None:
        print(f"Núcleos lógicos CPU  : {cpu_cores}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Dispositivo          : GPU ({gpu_name})")
        print(f"VRAM Disponible      : {vram_gb:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Dispositivo          : CPU")

    if os.name == 'nt':
        num_workers = 0
        print("Configuración OS     : Windows (num_workers=0)")
    else:
        num_workers = 2
        print("Configuración OS     : Unix/Linux (num_workers=2)")

    return device, num_workers


def set_seed(seed: int = 42) -> None:
    """Fija las semillas para garantizar reproducibilidad en CPU y GPU."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\nSemilla global fijada en: {seed}")


# =========================================================
# Filtrado ETL del dataset CheXpert (blacklist)
# =========================================================

def filtrar_chexpert_dataset(
    df_ini: pd.DataFrame,
    cols_patologias: Optional[List[str]] = None,
    excluir_positivos_en: Optional[List[str]] = None,
    excluir_vistas: Optional[List[str]] = None,
    excluir_posiciones: Optional[List[str]] = None,
    excluir_valores_globales: Optional[List[float]] = None,
    excluir_incertidumbre: bool = True,
    eliminar_inconsistencias_nofinding: bool = True,
    filtrar_edad_min: Optional[float] = None,
    filtrar_edad_max: Optional[float] = None,
    filtrar_sexo: Optional[List[str]] = None,
    validar_columnas: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Aplica un pipeline de filtrado ETL sobre el dataset CheXpert mediante
    criterios de exclusión (blacklist). Devuelve el DataFrame filtrado y un
    reporte de métricas de retención por criterio.
    """
    excluir_positivos_en = excluir_positivos_en or []
    excluir_vistas = excluir_vistas or []
    excluir_posiciones = excluir_posiciones or []
    excluir_valores_globales = excluir_valores_globales or []
    filtrar_sexo = filtrar_sexo or []

    if cols_patologias is None:
        cols_patologias = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]

    if validar_columnas:
        columnas_requeridas = cols_patologias + ['Frontal/Lateral', 'AP/PA', 'Age', 'Sex']
        faltantes = [col for col in columnas_requeridas if col not in df_ini.columns]
        if faltantes:
            raise ValueError(f"Columnas faltantes en el DataFrame: {faltantes}")

    df_filtrado = df_ini.copy()
    total_original = len(df_filtrado)

    perdida_por_positivos = 0
    perdida_por_vista = 0
    perdida_por_posicion = 0
    perdida_por_valores = 0
    perdida_por_inconsistencia = 0
    perdida_por_edad = 0
    perdida_por_sexo = 0

    # Filtro A: exclusión por positividad en patologías no deseadas
    if excluir_positivos_en:
        total_antes = len(df_filtrado)
        mascara = ~(df_filtrado[excluir_positivos_en] == 1.0).any(axis=1)
        df_filtrado = df_filtrado[mascara]
        perdida_por_positivos = total_antes - len(df_filtrado)

    # Filtro B: exclusión por vista radiológica
    if excluir_vistas:
        total_antes = len(df_filtrado)
        mascara = ~df_filtrado['Frontal/Lateral'].isin(excluir_vistas)
        df_filtrado = df_filtrado[mascara]
        perdida_por_vista = total_antes - len(df_filtrado)

    # Filtro C: exclusión por posición
    if excluir_posiciones:
        total_antes = len(df_filtrado)
        mascara = ~df_filtrado['AP/PA'].isin(excluir_posiciones)
        df_filtrado = df_filtrado[mascara]
        perdida_por_posicion = total_antes - len(df_filtrado)

    # Filtro D: exclusión por valores globales e incertidumbre (-1.0)
    if excluir_valores_globales or excluir_incertidumbre:
        total_antes = len(df_filtrado)
        mascara = pd.Series(False, index=df_filtrado.index)

        if excluir_valores_globales:
            mascara = mascara | df_filtrado[cols_patologias].isin(excluir_valores_globales).any(axis=1)
            if any(pd.isna(v) for v in excluir_valores_globales):
                mascara = mascara | df_filtrado[cols_patologias].isna().any(axis=1)

        if excluir_incertidumbre:
            mascara = mascara | (df_filtrado[cols_patologias] == -1.0).any(axis=1)

        df_filtrado = df_filtrado[~mascara]
        perdida_por_valores = total_antes - len(df_filtrado)

    # Filtro E: eliminación de inconsistencias diagnósticas en 'No Finding'
    if eliminar_inconsistencias_nofinding:
        total_antes = len(df_filtrado)
        otras_patologias = [col for col in cols_patologias if col != 'No Finding']
        mascara = (
            (df_filtrado['No Finding'] == 1.0) &
            (df_filtrado[otras_patologias] == 1.0).any(axis=1)
        )
        df_filtrado = df_filtrado[~mascara]
        perdida_por_inconsistencia = total_antes - len(df_filtrado)

    # Filtro F: segmentación por rango etario
    if filtrar_edad_min is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] >= filtrar_edad_min]
        perdida_por_edad += total_antes - len(df_filtrado)

    if filtrar_edad_max is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] <= filtrar_edad_max]
        perdida_por_edad += total_antes - len(df_filtrado)

    # Filtro G: segmentación por sexo
    if filtrar_sexo:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Sex'].isin(filtrar_sexo)]
        perdida_por_sexo = total_antes - len(df_filtrado)

    total_final = len(df_filtrado)
    total_perdido = total_original - total_final
    porcentaje_retencion = (total_final / total_original) * 100 if total_original > 0 else 0
    porcentaje_perdido = (total_perdido / total_original) * 100 if total_original > 0 else 0

    reporte = {
        "total_original": total_original,
        "total_final": total_final,
        "total_perdido": total_perdido,
        "porcentaje_retencion": porcentaje_retencion,
        "porcentaje_perdido": porcentaje_perdido,
        "perdida_por_positivos": perdida_por_positivos,
        "perdida_por_vista": perdida_por_vista,
        "perdida_por_posicion": perdida_por_posicion,
        "perdida_por_valores": perdida_por_valores,
        "perdida_por_inconsistencia": perdida_por_inconsistencia,
        "perdida_por_edad": perdida_por_edad,
        "perdida_por_sexo": perdida_por_sexo,
    }

    return df_filtrado, reporte


# =========================================================
# Filtrado ETL del dataset CheXpert (whitelist)
# =========================================================

def filtrar_chexpert_dataset_whitelist(
    df_ini: pd.DataFrame,
    cols_patologias: Optional[List[str]] = None,
    permitir_positivos_solo_en: Optional[List[str]] = None,
    incluir_vistas: Optional[List[str]] = None,
    incluir_posiciones: Optional[List[str]] = None,
    valores_permitidos: Optional[List[float]] = None,
    eliminar_inconsistencias_nofinding: bool = True,
    filtrar_edad_min: Optional[float] = None,
    filtrar_edad_max: Optional[float] = None,
    incluir_sexo: Optional[List[str]] = None,
    validar_columnas: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Aplica un pipeline de filtrado ETL sobre el dataset CheXpert mediante
    criterios de inclusión (whitelist). Devuelve el DataFrame filtrado y un
    reporte de métricas de retención por criterio.
    """
    incluir_vistas = incluir_vistas or []
    incluir_posiciones = incluir_posiciones or []
    incluir_sexo = incluir_sexo or []

    if cols_patologias is None:
        cols_patologias = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]

    if validar_columnas:
        columnas_requeridas = cols_patologias + ['Frontal/Lateral', 'AP/PA', 'Age', 'Sex']
        faltantes = [col for col in columnas_requeridas if col not in df_ini.columns]
        if faltantes:
            raise ValueError(f"Columnas requeridas no detectadas en la estructura de datos: {faltantes}")

    df_filtrado = df_ini.copy()
    total_original = len(df_filtrado)

    perdida_por_positivos = 0
    perdida_por_vista = 0
    perdida_por_posicion = 0
    perdida_por_valores = 0
    perdida_por_inconsistencia = 0
    perdida_por_edad = 0
    perdida_por_sexo = 0

    # Filtro A: whitelist de patologías con positivos permitidos
    if permitir_positivos_solo_en is not None:
        total_antes = len(df_filtrado)
        patologias_prohibidas = [col for col in cols_patologias if col not in permitir_positivos_solo_en]
        if patologias_prohibidas:
            mascara_prohibidas = (df_filtrado[patologias_prohibidas] == 1.0).any(axis=1)
            df_filtrado = df_filtrado[~mascara_prohibidas]
        perdida_por_positivos = total_antes - len(df_filtrado)

    # Filtro B: whitelist de vistas radiológicas
    if incluir_vistas:
        total_antes = len(df_filtrado)
        mascara = df_filtrado['Frontal/Lateral'].isin(incluir_vistas)
        df_filtrado = df_filtrado[mascara]
        perdida_por_vista = total_antes - len(df_filtrado)

    # Filtro C: whitelist de posiciones
    if incluir_posiciones:
        total_antes = len(df_filtrado)
        mascara = df_filtrado['AP/PA'].isin(incluir_posiciones)
        df_filtrado = df_filtrado[mascara]
        perdida_por_posicion = total_antes - len(df_filtrado)

    # Filtro D: whitelist de valores permitidos
    if valores_permitidos is not None:
        total_antes = len(df_filtrado)
        permitir_nan = any(pd.isna(v) for v in valores_permitidos)
        valores_limpios = [v for v in valores_permitidos if not pd.isna(v)]
        mascara_validos = df_filtrado[cols_patologias].isin(valores_limpios)
        if permitir_nan:
            mascara_validos = mascara_validos | df_filtrado[cols_patologias].isna()
        filas_invalidas = (~mascara_validos).any(axis=1)
        df_filtrado = df_filtrado[~filas_invalidas]
        perdida_por_valores = total_antes - len(df_filtrado)

    # Filtro E: eliminación de inconsistencias diagnósticas en 'No Finding'
    if eliminar_inconsistencias_nofinding:
        total_antes = len(df_filtrado)
        otras_patologias = [col for col in cols_patologias if col != 'No Finding']
        mascara = (
            (df_filtrado['No Finding'] == 1.0) &
            (df_filtrado[otras_patologias] == 1.0).any(axis=1)
        )
        df_filtrado = df_filtrado[~mascara]
        perdida_por_inconsistencia = total_antes - len(df_filtrado)

    # Filtro F: segmentación por rango etario
    if filtrar_edad_min is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] >= filtrar_edad_min]
        perdida_por_edad += total_antes - len(df_filtrado)

    if filtrar_edad_max is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] <= filtrar_edad_max]
        perdida_por_edad += total_antes - len(df_filtrado)

    # Filtro G: whitelist demográfica por sexo
    if incluir_sexo:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Sex'].isin(incluir_sexo)]
        perdida_por_sexo = total_antes - len(df_filtrado)

    total_final = len(df_filtrado)
    total_perdido = total_original - total_final
    porcentaje_retencion = (total_final / total_original) * 100 if total_original > 0 else 0
    porcentaje_perdido = (total_perdido / total_original) * 100 if total_original > 0 else 0

    reporte = {
        "total_original": total_original,
        "total_final": total_final,
        "total_perdido": total_perdido,
        "porcentaje_retencion": porcentaje_retencion,
        "porcentaje_perdido": porcentaje_perdido,
        "perdida_por_positivos": perdida_por_positivos,
        "perdida_por_vista": perdida_por_vista,
        "perdida_por_posicion": perdida_por_posicion,
        "perdida_por_valores": perdida_por_valores,
        "perdida_por_inconsistencia": perdida_por_inconsistencia,
        "perdida_por_edad": perdida_por_edad,
        "perdida_por_sexo": perdida_por_sexo,
    }

    return df_filtrado, reporte


def aplicar_filtrado_proyecto(df_ini: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Encapsula los parámetros de filtrado específicos del TFM en formato whitelist
    para estandarizar la ejecución del pipeline ETL sobre el dataset CheXpert.
    """
    permitir_positivos_solo_en = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Fracture', 'Support Devices'
    ]
    incluir_vistas = ['Frontal']
    incluir_posiciones = ['AP']
    valores_permitidos = [0.0, 1.0, np.nan]
    eliminar_inconsistencias_nofinding = True
    filtrar_edad_min = None
    filtrar_edad_max = None
    incluir_sexo = ['Male', 'Female']

    return filtrar_chexpert_dataset_whitelist(
        df_ini=df_ini,
        permitir_positivos_solo_en=permitir_positivos_solo_en,
        incluir_vistas=incluir_vistas,
        incluir_posiciones=incluir_posiciones,
        valores_permitidos=valores_permitidos,
        eliminar_inconsistencias_nofinding=eliminar_inconsistencias_nofinding,
        filtrar_edad_min=filtrar_edad_min,
        filtrar_edad_max=filtrar_edad_max,
        incluir_sexo=incluir_sexo,
    )


def auditar_dataset(df: pd.DataFrame, columnas_auditar: Optional[List[str]] = None) -> None:
    """
    Genera un reporte de auditoría en formato Markdown mostrando el total de
    muestras y la distribución de valores por columna.

    Nota: esta función usa IPython.display y solo produce salida visible dentro
    de un entorno Jupyter Notebook o JupyterLab. En un script CLI no produce
    ninguna salida.
    """
    total_filas = len(df)

    if columnas_auditar is None:
        columnas_auditar = [
            'Frontal/Lateral', 'AP/PA', 'Sex',
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]

    columnas_auditar = [col for col in columnas_auditar if col in df.columns]

    reporte_md = f"### Auditoría del Dataset (Post-Procesamiento)\n"
    reporte_md += f"**Total de muestras (imágenes) disponibles:** `{total_filas:,}`\n\n"
    reporte_md += "| Columna / Variable | Distribución de Valores (Conteo y Porcentaje) |\n"
    reporte_md += "| :--- | :--- |\n"

    for col in columnas_auditar:
        conteos = df[col].value_counts(dropna=False)
        detalles = []
        for valor, cantidad in conteos.items():
            str_valor = "NaN (Nulo)" if pd.isna(valor) else str(valor)
            porcentaje = (cantidad / total_filas) * 100
            detalles.append(f"**{str_valor}**: {cantidad:,} ({porcentaje:.1f}%)")
        str_detalles = " <br> ".join(detalles)
        reporte_md += f"| **{col}** | {str_detalles} |\n"

    display(Markdown(reporte_md))


# =========================================================
# Utilidades de mapeo de rutas del dataset
# =========================================================

def obtener_ruta_absoluta_train(
    ruta_csv: str,
    directorio_raiz_train: str,
    posibles_batches: List[str]
) -> Optional[str]:
    """
    Construye la ruta absoluta de una imagen de entrenamiento buscando en los
    subdirectorios de batch disponibles. Devuelve None si no se encuentra el archivo.

    Parámetros
    ----------
    ruta_csv : str
        Valor de la columna 'Path' del CSV de CheXpert.
    directorio_raiz_train : str
        Ruta absoluta al directorio raíz del dataset de entrenamiento.
    posibles_batches : list of str
        Lista de nombres de subdirectorios de batch a explorar.
    """
    match = re.search(r'(patient\d+/study\d+/.*\.jpg)', ruta_csv)
    if match:
        parte_relativa = match.group(1).replace('/', os.sep)
        for batch in posibles_batches:
            ruta_prueba = os.path.join(directorio_raiz_train, batch, parte_relativa)
            if os.path.exists(ruta_prueba):
                return ruta_prueba
    return None


def mapear_ruta_valid_definitiva(ruta_csv: str, directorio_raiz_valid: str) -> str:
    """
    Construye la ruta absoluta de una imagen del conjunto de validación (Gold Standard).

    Parámetros
    ----------
    ruta_csv : str
        Valor de la columna 'Path' del CSV de validación de CheXpert.
    directorio_raiz_valid : str
        Ruta absoluta al directorio raíz del conjunto de validación.
    """
    parte_relativa = ruta_csv.split('CheXpert-v1.0/')[-1]
    ruta_limpia = parte_relativa.replace('/', os.sep)
    return os.path.join(directorio_raiz_valid, ruta_limpia)


# =========================================================
# Métricas de evaluación
# =========================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula accuracy y F1-macro para clasificación multietiqueta.
    y_pred debe ser un array binario (0 o 1) tras aplicar el umbral de decisión.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {'accuracy': acc, 'f1_macro': f1}


# =========================================================
# Utilidades de inspección del modelo
# =========================================================

def mostrar_info_modelo(model: torch.nn.Module) -> None:
    """Imprime un resumen técnico de parámetros totales y entrenables del modelo."""
    print("-" * 30)
    print("RESUMEN DEL MODELO")
    print("-" * 30)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales:     {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Tipo de modelo:         {type(model).__name__}")
    print("-" * 30)


# =========================================================
# Clases auxiliares de entrenamiento
# =========================================================

class EarlyStopping:
    """
    Detiene el entrenamiento si la pérdida de validación no mejora durante
    un número de épocas consecutivas igual a 'patience'.
    """

    def __init__(self, patience: int = 6, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ModelCheckpoint:
    """
    Guarda en memoria el estado del modelo con el mejor F1-Score macro
    observado durante el entrenamiento.
    """

    def __init__(self) -> None:
        self.best_f1 = -float('inf')
        self.best_model_state = None

    def __call__(self, model: torch.nn.Module, f1_val: float) -> bool:
        if f1_val > self.best_f1:
            self.best_f1 = f1_val
            self.best_model_state = copy.deepcopy(model.state_dict())
            return True
        return False