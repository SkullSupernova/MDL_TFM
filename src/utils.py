# src/utils.py
#
# Utilidades transversales del proyecto: configuración del entorno, pipeline ETL
# del dataset CheXpert, métricas, inspección del modelo y callbacks de entrenamiento.
#
# Este módulo es el único punto donde se definen los criterios de limpieza del dataset,
# lo que permite cambiar la estrategia de filtrado sin tocar el notebook ni main.py.

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

    # Windows no soporta el modelo fork() de multiprocessing que usa PyTorch para
    # lanzar los worker del DataLoader. Con num_workers > 0 en Windows, cada worker
    # debe serializar y deserializar el dataset completo mediante pickle al arrancar,
    # lo que causa fallos o cuelgues en datasets complejos con lambdas o transforms.
    # num_workers=0 deshabilita el paralelismo y carga los batches en el proceso
    # principal: más lento, pero estable y sin restricciones de pickling.
    # En Linux/Mac se usa fork() sin coste de serialización, por eso num_workers=2.
    if os.name == 'nt':
        num_workers = 0
        print("Configuración OS     : Windows (num_workers=0)")
    else:
        num_workers = 2
        print("Configuración OS     : Unix/Linux (num_workers=2)")

    return device, num_workers


def set_seed(seed: int = 42) -> None:
    """Fija las semillas para garantizar reproducibilidad en CPU y GPU."""
    # Los experimentos de deep learning tienen varias fuentes de aleatoriedad
    # independientes; es necesario fijar todas ellas para que dos ejecuciones
    # con la misma semilla produzcan exactamente los mismos resultados:
    #
    #   PYTHONHASHSEED  — orden de las claves en diccionarios y conjuntos de Python
    #   numpy           — operaciones de muestreo (data augmentation, splits)
    #   random          — módulo estándar usado por algunas transforms de torchvision
    #   torch CPU       — inicialización de pesos, dropout
    #   torch CUDA      — operaciones en GPU (una por dispositivo, y globalmente)
    #
    # deterministic=True y benchmark=False en cuDNN hacen que el backend de CUDA
    # use algoritmos deterministas en lugar de los más rápidos (que pueden ser
    # no deterministas por paralelismo a nivel hardware).
    # Nota: deterministic=True puede reducir el rendimiento en ~5-10 % en GPU.
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

    El enfoque blacklist define qué excluir; todo lo que no coincide se retiene.
    Es más permisivo que whitelist y adecuado para depuración iterativa del dataset.

    Parámetros
    ----------
    df_ini : pd.DataFrame
        DataFrame bruto con las columnas del CSV de CheXpert.
    cols_patologias : list of str, optional
        Columnas de patología a considerar. Si es None, se usan las 14 estándar.
    excluir_positivos_en : list of str, optional
        Excluir filas con valor 1.0 en alguna de estas columnas de patología.
        Útil para eliminar una patología completa del dataset (p.ej. 'Pleural Other').
    excluir_vistas : list of str, optional
        Valores de 'Frontal/Lateral' a excluir (p.ej. ['Lateral']).
    excluir_posiciones : list of str, optional
        Valores de 'AP/PA' a excluir (p.ej. ['PA']).
    excluir_valores_globales : list of float, optional
        Valores numéricos a excluir en cualquier columna de patología.
    excluir_incertidumbre : bool
        Si True, excluye filas con valor -1.0 en cualquier patología.
    eliminar_inconsistencias_nofinding : bool
        Si True, excluye filas donde 'No Finding' = 1.0 y alguna otra patología = 1.0.
    filtrar_edad_min, filtrar_edad_max : float, optional
        Rango etario de retención.
    filtrar_sexo : list of str, optional
        Valores de 'Sex' a excluir.
    validar_columnas : bool
        Si True, lanza ValueError si faltan columnas requeridas.

    Devuelve
    --------
    df_filtrado : pd.DataFrame
    reporte : dict con métricas de retención por criterio
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

    # Trabajar sobre una copia para no mutar el DataFrame original,
    # lo que podría causar efectos secundarios si el caller lo reutiliza.
    df_filtrado = df_ini.copy()
    total_original = len(df_filtrado)

    perdida_por_positivos = 0
    perdida_por_vista = 0
    perdida_por_posicion = 0
    perdida_por_valores = 0
    perdida_por_inconsistencia = 0
    perdida_por_edad = 0
    perdida_por_sexo = 0

    # Filtro A: exclusión por positividad en patologías no deseadas.
    # Elimina estudios que contienen una patología que no forma parte del
    # conjunto de clases del TFM (p.ej. 'Pleural Other', excluida del modelo de 13 clases).
    # Retener estos estudios introduciría etiquetas positivas sin clase correspondiente
    # en el modelo, lo que podría confundir el entrenamiento.
    if excluir_positivos_en:
        total_antes = len(df_filtrado)
        mascara = ~(df_filtrado[excluir_positivos_en] == 1.0).any(axis=1)
        df_filtrado = df_filtrado[mascara]
        perdida_por_positivos = total_antes - len(df_filtrado)

    # Filtro B: exclusión por vista radiológica.
    # Las radiografías laterales tienen una distribución visual muy diferente a las
    # frontales: los modelos entrenados en frontales no generalizan bien a laterales.
    # El TFM usa exclusivamente vistas frontales para mantener coherencia con la
    # literatura de referencia (DenseNet-121 de Irvin et al., 2019).
    if excluir_vistas:
        total_antes = len(df_filtrado)
        mascara = ~df_filtrado['Frontal/Lateral'].isin(excluir_vistas)
        df_filtrado = df_filtrado[mascara]
        perdida_por_vista = total_antes - len(df_filtrado)

    # Filtro C: exclusión por posición (AP/PA).
    # Las proyecciones AP (antero-posterior) y PA (postero-anterior) difieren en la
    # magnificación del corazón y los vasos. PA es la proyección estándar clínica;
    # AP se usa cuando el paciente no puede ponerse de pie (más graves, más artefactos).
    if excluir_posiciones:
        total_antes = len(df_filtrado)
        mascara = ~df_filtrado['AP/PA'].isin(excluir_posiciones)
        df_filtrado = df_filtrado[mascara]
        perdida_por_posicion = total_antes - len(df_filtrado)

    # Filtro D: exclusión por valores especiales e incertidumbre.
    # CheXpert usa -1.0 para indicar incertidumbre diagnóstica (el radiólogo no pudo
    # determinar con certeza si la patología estaba presente o ausente).
    # Incluir estos estudios añadiría ruido de etiqueta (label noise) al entrenamiento.
    # La estrategia de descartar -1.0 es la más conservadora; alternativas son
    # tratarlos como positivos (upper bound) o negativos (lower bound), pero ambas
    # introducen sesgo sistemático.
    if excluir_valores_globales or excluir_incertidumbre:
        total_antes = len(df_filtrado)
        mascara = pd.Series(False, index=df_filtrado.index)

        if excluir_valores_globales:
            mascara = mascara | df_filtrado[cols_patologias].isin(excluir_valores_globales).any(axis=1)
            # isin() no detecta NaN (Python trata NaN != NaN), por lo que se comprueba
            # explícitamente si el caller incluye NaN en la lista de valores a excluir.
            if any(pd.isna(v) for v in excluir_valores_globales):
                mascara = mascara | df_filtrado[cols_patologias].isna().any(axis=1)

        if excluir_incertidumbre:
            mascara = mascara | (df_filtrado[cols_patologias] == -1.0).any(axis=1)

        df_filtrado = df_filtrado[~mascara]
        perdida_por_valores = total_antes - len(df_filtrado)

    # Filtro E: eliminación de inconsistencias diagnósticas en 'No Finding'.
    # Un estudio no puede ser simultáneamente 'No Finding'=1 y tener otra patología=1.
    # Esta contradicción ocurre en el CSV original debido a errores de etiquetado
    # automático del CheXbert. Retener estas filas contaminaría el aprendizaje de
    # ambas clases: el modelo no podría aprender qué significa realmente 'No Finding'.
    if eliminar_inconsistencias_nofinding:
        total_antes = len(df_filtrado)
        otras_patologias = [col for col in cols_patologias if col != 'No Finding']
        mascara = (
            (df_filtrado['No Finding'] == 1.0) &
            (df_filtrado[otras_patologias] == 1.0).any(axis=1)
        )
        df_filtrado = df_filtrado[~mascara]
        perdida_por_inconsistencia = total_antes - len(df_filtrado)

    # Filtros F y G: segmentación demográfica por edad y sexo.
    # No se aplican en el pipeline principal del TFM (None por defecto), pero
    # se mantienen disponibles para análisis de subgrupos o estudios de sesgo.
    if filtrar_edad_min is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] >= filtrar_edad_min]
        perdida_por_edad += total_antes - len(df_filtrado)

    if filtrar_edad_max is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] <= filtrar_edad_max]
        perdida_por_edad += total_antes - len(df_filtrado)

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

    El enfoque whitelist define explícitamente qué retener; todo lo demás se descarta.
    Es más restrictivo que blacklist y reduce el riesgo de incluir datos inesperados.
    Este es el pipeline activo en el TFM, invocado a través de aplicar_filtrado_proyecto().

    Parámetros
    ----------
    df_ini : pd.DataFrame
        DataFrame bruto con las columnas del CSV de CheXpert.
    cols_patologias : list of str, optional
        Columnas de patología a considerar. Si es None, se usan las 14 estándar.
    permitir_positivos_solo_en : list of str, optional
        Solo se permite que haya positivos (1.0) en estas columnas. Estudios con
        positivos en otras columnas son descartados.
    incluir_vistas : list of str, optional
        Valores de 'Frontal/Lateral' a retener (p.ej. ['Frontal']).
    incluir_posiciones : list of str, optional
        Valores de 'AP/PA' a retener (p.ej. ['AP']).
    valores_permitidos : list of float, optional
        Únicos valores numéricos aceptados en las columnas de patología.
        Cualquier fila con un valor no incluido en esta lista es descartada.
    eliminar_inconsistencias_nofinding : bool
        Si True, excluye filas donde 'No Finding'=1 y alguna otra patología=1.
    filtrar_edad_min, filtrar_edad_max : float, optional
        Rango etario de retención.
    incluir_sexo : list of str, optional
        Valores de 'Sex' a retener.
    validar_columnas : bool
        Si True, lanza ValueError si faltan columnas requeridas.

    Devuelve
    --------
    df_filtrado : pd.DataFrame
    reporte : dict con métricas de retención por criterio
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

    # Filtro A: whitelist de patologías con positivos permitidos.
    # Identifica las columnas cuyo valor 1.0 NO está permitido (el complemento
    # de permitir_positivos_solo_en) y descarta los estudios que las activan.
    # Esto es equivalente a decir "quiero estudios que, si tienen algún positivo,
    # sea solo en las clases del TFM, no en las excluidas".
    if permitir_positivos_solo_en is not None:
        total_antes = len(df_filtrado)
        patologias_prohibidas = [col for col in cols_patologias if col not in permitir_positivos_solo_en]
        if patologias_prohibidas:
            mascara_prohibidas = (df_filtrado[patologias_prohibidas] == 1.0).any(axis=1)
            df_filtrado = df_filtrado[~mascara_prohibidas]
        perdida_por_positivos = total_antes - len(df_filtrado)

    # Filtro B: whitelist de vistas radiológicas.
    if incluir_vistas:
        total_antes = len(df_filtrado)
        mascara = df_filtrado['Frontal/Lateral'].isin(incluir_vistas)
        df_filtrado = df_filtrado[mascara]
        perdida_por_vista = total_antes - len(df_filtrado)

    # Filtro C: whitelist de posiciones.
    if incluir_posiciones:
        total_antes = len(df_filtrado)
        mascara = df_filtrado['AP/PA'].isin(incluir_posiciones)
        df_filtrado = df_filtrado[mascara]
        perdida_por_posicion = total_antes - len(df_filtrado)

    # Filtro D: whitelist de valores permitidos en columnas de patología.
    # En el TFM se permiten [0.0, 1.0, NaN]: binario limpio más NaN (ausencia de
    # etiqueta, que se imputa a 0.0 en el Dataset). Se descartan los -1.0 (incertidumbre)
    # y cualquier otro valor anómalo.
    # La comprobación de NaN es necesaria porque isin() no detecta float('nan'):
    # dos NaN no son iguales en IEEE 754 (nan != nan es True por definición).
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

    # Filtro E: eliminación de inconsistencias en 'No Finding' (ver filtrar_chexpert_dataset).
    if eliminar_inconsistencias_nofinding:
        total_antes = len(df_filtrado)
        otras_patologias = [col for col in cols_patologias if col != 'No Finding']
        mascara = (
            (df_filtrado['No Finding'] == 1.0) &
            (df_filtrado[otras_patologias] == 1.0).any(axis=1)
        )
        df_filtrado = df_filtrado[~mascara]
        perdida_por_inconsistencia = total_antes - len(df_filtrado)

    if filtrar_edad_min is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] >= filtrar_edad_min]
        perdida_por_edad += total_antes - len(df_filtrado)

    if filtrar_edad_max is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] <= filtrar_edad_max]
        perdida_por_edad += total_antes - len(df_filtrado)

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

    Decisiones de diseño fijadas aquí:
    - 'Pleural Other' se excluye de permitir_positivos_solo_en porque el modelo
      trabaja con 13 clases (sin 'Pleural Other'). Incluir estudios con esa clase
      activa introduciría ejemplos con patología real pero sin clase modelo asignada.
    - Solo vistas frontales y proyección AP: coherencia con el diseño del TFM.
    - valores_permitidos=[0.0, 1.0, NaN]: descarta incertidumbre (-1.0) sin necesidad
      de especificarla explícitamente (whitelist más seguro que blacklist aquí).
    - eliminar_inconsistencias_nofinding=True: ver filtrar_chexpert_dataset_whitelist.
    """
    permitir_positivos_solo_en = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Fracture', 'Support Devices'
    ]
    incluir_vistas = ['Frontal']
    incluir_posiciones = ['AP']
    # NaN se incluye porque muchos estudios tienen etiqueta ausente (no positivo ni negativo)
    # en patologías que simplemente no fueron evaluadas. Se imputarán a 0.0 en el Dataset.
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
    ninguna salida porque IPython.display.display() no tiene efecto fuera de un
    kernel de IPython.
    """
    total_filas = len(df)

    if columnas_auditar is None:
        columnas_auditar = [
            'Frontal/Lateral', 'AP/PA', 'Sex',
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]

    # Filtrar columnas que no están en el DataFrame sin lanzar error,
    # para que la función sea robusta ante distintos estados del pipeline ETL.
    columnas_auditar = [col for col in columnas_auditar if col in df.columns]

    reporte_md = f"### Auditoría del Dataset (Post-Procesamiento)\n"
    reporte_md += f"**Total de muestras (imágenes) disponibles:** `{total_filas:,}`\n\n"
    reporte_md += "| Columna / Variable | Distribución de Valores (Conteo y Porcentaje) |\n"
    reporte_md += "| :--- | :--- |\n"

    for col in columnas_auditar:
        # dropna=False incluye NaN en el conteo, lo que es importante para
        # detectar valores faltantes que de otro modo quedarían invisibles.
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

    CheXpert distribuye las imágenes en múltiples subdirectorios de batch
    (batch1, batch2, ...) porque el dataset completo supera 400 GB. La ruta
    en el CSV es relativa y no incluye el nombre del batch, por lo que es
    necesario probar cada uno hasta encontrar el archivo.

    Parámetros
    ----------
    ruta_csv : str
        Valor de la columna 'Path' del CSV de CheXpert.
        Ejemplo: 'CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg'
    directorio_raiz_train : str
        Ruta absoluta al directorio raíz que contiene los subdirectorios de batch.
    posibles_batches : list of str
        Nombres de los subdirectorios de batch a probar (p.ej. ['batch1', 'batch2']).
    """
    # Extraer la parte relevante de la ruta (patient/study/imagen.jpg) ignorando
    # el prefijo variable del CSV ('CheXpert-v1.0/train/', 'valid/', etc.).
    # La regex captura desde 'patientXXXXX' hasta el final del nombre de archivo.
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

    El conjunto de validación de CheXpert tiene estructura plana (sin batches),
    por lo que la construcción de ruta es directa: basta con eliminar el prefijo
    'CheXpert-v1.0/' de la ruta del CSV y unirlo al directorio raíz local.

    Parámetros
    ----------
    ruta_csv : str
        Valor de la columna 'Path' del CSV de validación.
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
    Esta función no aplica umbral: eso es responsabilidad del caller (ver train.py).

    Se usa F1-macro (promedio no ponderado del F1 de cada clase) en lugar de
    micro o weighted porque:
    - Las 13 clases del TFM tienen distribuciones muy desbalanceadas.
    - Macro trata todas las clases por igual, penalizando el rendimiento pobre
      en clases minoritarias que serían ignoradas por micro o weighted.
    - Es la métrica de referencia en los benchmarks de CheXpert (Irvin et al., 2019).

    zero_division=0 evita warnings cuando una clase no tiene positivos reales
    ni predichos en un batch de validación (produce F1=0 para esa clase, que
    se incluye correctamente en el promedio macro sin lanzar excepción).
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {'accuracy': acc, 'f1_macro': f1}


# =========================================================
# Utilidades de inspección del modelo
# =========================================================

def mostrar_info_modelo(model: torch.nn.Module) -> None:
    """Imprime un resumen técnico de parámetros totales y entrenables del modelo."""
    # La distinción entre totales y entrenables es importante durante el fine-tuning:
    # si se congela el backbone (requires_grad=False en sus capas), los parámetros
    # entrenables serán mucho menos que el total, lo que acelera el entrenamiento
    # y reduce el riesgo de sobreajuste en datasets pequeños.
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

    Motivación: continuar entrenando después de que el modelo haya alcanzado
    su mínimo de validación solo empeora el sobreajuste y desperdicia tiempo.
    Con patience=6, el modelo tiene 6 oportunidades para salir de un plateau
    antes de ser detenido; esto es importante con schedulers ReduceLROnPlateau
    que pueden activar un descenso del LR que reactive la mejora.
    """

    def __init__(self, patience: int = 6, min_delta: float = 0.0) -> None:
        self.patience = patience
        # min_delta define un umbral mínimo de mejora: una reducción de la pérdida
        # menor que min_delta no cuenta como mejora real. Útil para ignorar
        # fluctuaciones numéricas en la pérdida que no corresponden a aprendizaje real.
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            # Primera época: inicializar la mejor pérdida sin incrementar el contador.
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # La pérdida no mejoró (es mayor o igual que best_loss - min_delta).
            # Se usa > en lugar de >= para permitir igualdad exacta como "sin mejora",
            # lo que es correcto cuando min_delta=0: cualquier pérdida que no baje
            # estrictamente incrementa el contador.
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # La pérdida mejoró suficientemente: resetear el contador y actualizar
            # el mejor valor. No se activa el early stop en este caso.
            self.best_loss = val_loss
            self.counter = 0


class ModelCheckpoint:
    """
    Guarda en memoria el estado del modelo con el mejor F1-Score macro
    observado durante el entrenamiento.

    Por qué en memoria (no en disco en cada epoch):
    - Guardar en disco en cada mejora crea muchos archivos intermedios y es lento.
    - Al final del entrenamiento se escribe exactamente un archivo con el mejor estado.
    - Si el proceso se interrumpe antes del final, se pierde el checkpoint, pero esto
      es aceptable en un entorno de investigación donde el entrenamiento es supervisado.

    Por qué deepcopy:
    - model.state_dict() devuelve una referencia a los tensores actuales del modelo.
    - Sin deepcopy, best_model_state apuntaría a los mismos tensores que se actualizan
      en cada paso de optimización, perdiendo el estado de la mejor época.
    - deepcopy crea copias independientes de todos los tensores: más memoria, pero
      garantiza que best_model_state refleja el estado exacto del momento de la copia.

    Por qué F1 (no val_loss):
    - La pérdida mide el error de clasificación promedio, no el rendimiento clínico.
    - En datasets desbalanceados, la pérdida puede bajar mientras el F1 en clases
      minoritarias empeora (el modelo aprende a predecir siempre la clase mayoritaria).
    - F1-macro da igual peso a todas las clases: el mejor checkpoint es el que mejor
      equilibra el rendimiento en todas las patologías, no solo en las más frecuentes.
    """

    def __init__(self) -> None:
        # Inicializar con -inf para que cualquier F1 real (incluso 0.0) se considere mejora
        # en la primera época y se guarde el estado inicial del modelo.
        self.best_f1 = -float('inf')
        self.best_model_state = None

    def __call__(self, model: torch.nn.Module, f1_val: float) -> bool:
        if f1_val > self.best_f1:
            self.best_f1 = f1_val
            self.best_model_state = copy.deepcopy(model.state_dict())
            return True
        return False
