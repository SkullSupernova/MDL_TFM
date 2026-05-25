import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict


def filtrar_chexpert_dataset(
    df_ini: pd.DataFrame,
    cols_patologias: Optional[List[str]] = None,
    excluir_positivos_en: Optional[List[str]] = None,
    excluir_vistas: Optional[List[str]] = None,
    excluir_posiciones: Optional[List[str]] = None,
    excluir_valores_globales: Optional[List[float]] = None,
    excluir_incertidumbre: bool = True,  # NUEVO
    eliminar_inconsistencias_nofinding: bool = True,
    filtrar_edad_min: Optional[float] = None,
    filtrar_edad_max: Optional[float] = None,
    filtrar_sexo: Optional[List[str]] = None,
    validar_columnas: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:

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

    # Validación de columnas
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

    # Filtro A
    if excluir_positivos_en:
        total_antes = len(df_filtrado)
        mascara = ~(df_filtrado[excluir_positivos_en] == 1.0).any(axis=1)
        df_filtrado = df_filtrado[mascara]
        perdida_por_positivos = total_antes - len(df_filtrado)

    # Filtro B
    if excluir_vistas:
        total_antes = len(df_filtrado)
        mascara = ~df_filtrado['Frontal/Lateral'].isin(excluir_vistas)
        df_filtrado = df_filtrado[mascara]
        perdida_por_vista = total_antes - len(df_filtrado)

    # Filtro C
    if excluir_posiciones:
        total_antes = len(df_filtrado)
        mascara = ~df_filtrado['AP/PA'].isin(excluir_posiciones)
        df_filtrado = df_filtrado[mascara]
        perdida_por_posicion = total_antes - len(df_filtrado)

    # Filtro D (modificado)
    if excluir_valores_globales or excluir_incertidumbre:
        total_antes = len(df_filtrado)

        mascara = pd.Series(False, index=df_filtrado.index)

        # Valores explícitos
        if excluir_valores_globales:
            mascara = mascara | df_filtrado[cols_patologias].isin(excluir_valores_globales).any(axis=1)

            if any(pd.isna(v) for v in excluir_valores_globales):
                mascara = mascara | df_filtrado[cols_patologias].isna().any(axis=1)

        # Incertidumbre (-1.0)
        if excluir_incertidumbre:
            mascara = mascara | (df_filtrado[cols_patologias] == -1.0).any(axis=1)

        df_filtrado = df_filtrado[~mascara]
        perdida_por_valores = total_antes - len(df_filtrado)

    # Filtro E
    if eliminar_inconsistencias_nofinding:
        total_antes = len(df_filtrado)
        otras_patologias = [col for col in cols_patologias if col != 'No Finding']

        mascara = (
            (df_filtrado['No Finding'] == 1.0) &
            (df_filtrado[otras_patologias] == 1.0).any(axis=1)
        )

        df_filtrado = df_filtrado[~mascara]
        perdida_por_inconsistencia = total_antes - len(df_filtrado)

    # Filtro F: edad
    if filtrar_edad_min is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] >= filtrar_edad_min]
        perdida_por_edad += total_antes - len(df_filtrado)

    if filtrar_edad_max is not None:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Age'] <= filtrar_edad_max]
        perdida_por_edad += total_antes - len(df_filtrado)

    # Filtro G: sexo
    if filtrar_sexo:
        total_antes = len(df_filtrado)
        df_filtrado = df_filtrado[df_filtrado['Sex'].isin(filtrar_sexo)]
        perdida_por_sexo = total_antes - len(df_filtrado)

    # Métricas
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
        "perdida_por_sexo": perdida_por_sexo
    }

    return df_filtrado, reporte