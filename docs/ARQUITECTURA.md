# Documento técnico: arquitectura del proyecto

Referencia de mantenimiento del clasificador de patologías torácicas (CheXpert). Describe la estructura del
código, el flujo de datos, el pipeline de entrenamiento, la evaluación, el sistema de experimentos, la
generación de informes y las decisiones de diseño. Para el uso del sistema, ver
[GUIA_USO.md](GUIA_USO.md); para la instalación, el [README](../README.md).

---

## 1. Visión general

Clasificación **multietiqueta** de 13 patologías torácicas a partir de radiografías frontales del dataset
CheXpert. La arquitectura de referencia es DenseNet-121; el pipeline soporta **cinco backbones
intercambiables** (DenseNet-121, VGG16-BN, ResNet-50, ConvNeXt-Tiny y Swin-Tiny), seleccionables desde
`config.yml` o `--model`. El sistema añade explicabilidad (Grad-CAM), una capa de evaluación sobre un test
"silver-standard", un gate de promoción del mejor modelo, un sistema de seguimiento de experimentos, una API
REST y una interfaz web. Los resultados de la comparación de arquitecturas (con intervalos de confianza
bootstrap) están en [COMPARATIVA_ARQUITECTURAS.md](COMPARATIVA_ARQUITECTURAS.md).

**Patologías activas (13):** No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Fracture, Support Devices.
Se excluye `Pleural Other` (baja prevalencia). Las 5 de la competición CheXpert (Atelectasis, Cardiomegaly,
Consolidation, Edema, Pleural Effusion) se usan como métrica principal de promoción.

---

## 2. Estructura de directorios

| Directorio | Función |
|---|---|
| `src/` | Código fuente (ver sección 3) |
| `config/config.yml` | Parámetros centralizados (datos, modelo, entrenamiento, experimentos) |
| `chexpert_csv/` | CSV de metadatos/etiquetas (`train_cheXbert.csv`) |
| `models/` | Checkpoints `.pth` (no versionados) |
| `experiments/` | Una carpeta autocontenida por run + `leaderboard.csv` (no versionado) |
| `logs/` | Logs y artefactos auxiliares (no versionado) |
| `muestras_busqueda/` | Radiografías de ejemplo para la web |
| `notebook/` | Cuadernos del análisis y prototipo (`T02_Analisis_DenseNet121.ipynb` es la referencia) |
| `test/` | Suite pytest (datos sintéticos, sin dataset real) |
| `docs/` | Esta documentación y la guía de uso |

---

## 3. Módulos de `src/` (función de cada fichero)

- **`models.py`** — Arquitecturas y modelo de datos. `CHEXPERT_PATHOLOGY_COLS` (13) y `CHEXPERT_COMPETITION_5`;
  `CLASS_CONFIGS` y `get_active_pathology_cols` (configuraciones de clases, sección 10); `build_model`
  (densenet121/vgg16_bn/resnet50/convnext_tiny/swin_t, cabeza Linear-ReLU-Dropout-Linear); `get_grad_cam_layer`
  (capa objetivo por backbone) y `get_grad_cam_reshape` (reshape de tokens para transformers como Swin);
  `load_checkpoint` (infiere nº de clases y detecta cabeza simple/compuesta);
  `parse_checkpoint_filename` (extrae backbone y class_config del nombre); `CheXpertDataset` (lee imágenes y
  etiquetas).
- **`utils.py`** — `setup_environment`/`set_seed`; ETL whitelist (`filtrar_chexpert_dataset_whitelist`,
  `aplicar_filtrado_proyecto`); selección de clases (`aplicar_seleccion_clases`); construcción del test silver
  (`construir_df_test_valid`, `mapear_ruta_valid_definitiva`, `obtener_ruta_absoluta_train`); métricas
  (`calculate_metrics`, `auroc_macro`, `auc_por_clase`, `pr_auc_macro`, `pr_auc_por_clase`,
  `distribucion_clases`, `contar_parametros`); callbacks `EarlyStopping` y `ModelCheckpoint`; `auditar_dataset`
  (utilidad de auditoría para notebooks).
- **`train.py`** — `train_model`: bucle AMP, `BCEWithLogitsLoss` con `pos_weight`, selección de la mejor época
  por AUROC de validación, early stopping y checkpoints reanudables (`resume_path`, `resume`).
- **`main.py`** — Orquestación CLI: carga de config, ETL, selección de clases, split por paciente, datasets,
  `pos_weight`, construcción del modelo, entrenamiento, evaluación de validación y test, gate de promoción y
  registro en el tracker. Flags: `--model`, `--class-config`, `--seed`, `--epochs`, `--subset`,
  `--val-subset`, `--tag`, `--resume`.
- **`evaluate.py`** — Evaluación sobre el test silver: `evaluate_model`, `calcular_metricas_completas`,
  `evaluar_loader`, `evaluar_test`. Independiente del entrenamiento (evalúa cualquier checkpoint).
- **`model_registry.py`** — Registro del campeón (`best_model_registry.json`), comparación `es_mejor` (gate) e
  historial append-only (`experiments.jsonl`). Indexado por par `(backbone, class_config)`.
- **`experiment_tracker.py`** — `ExperimentTracker`: genera `experiments/<run_id>/` y `leaderboard.csv`.
- **`bootstrap_ci.py`** — agregación post-hoc: lee las predicciones de test guardadas por cada run y calcula
  intervalos de confianza **bootstrap** (AUROC CheXpert-5, AUROC-macro, PR-AUC-macro) → `leaderboard_ci.csv`.
- **`report.py`** — `build_report_pdf`: informe PDF (reportlab) de un análisis de la web.
- **`image_utils.py`** — `validar_imagen_radiografia` (validación de entrada) y `empaquetar_imagenes_zip`.
- **`preprocess_resize.py`** — `resize_tree` y CLI de pre-redimensionado del dataset a 224x224.
- **`visualization.py`** — Grad-CAM, matrices de confusión, curvas de aprendizaje y ROC/PR (con `save_path`).
- **`api.py`** — API FastAPI (`/health`, `/predict`).
- **`app.py`** — Interfaz web Streamlit.
- **`logging_config.py`** — Logging centralizado a stdout y `logs/app.log` con rotación.

---

## 4. Flujo de datos

1. **Carga**: `train_cheXbert.csv` (etiquetas automáticas CheXbert; ver sección 13).
2. **ETL whitelist** (`aplicar_filtrado_proyecto`): retiene solo vista Frontal + proyección AP, valores
   permitidos `{0.0, 1.0, NaN}` (descarta incertidumbre `-1`), elimina inconsistencias de `No Finding`.
3. **Localización de imágenes**: se resuelven las rutas absolutas entre los batches; se descartan las no presentes.
4. **Imputación**: `NaN -> 0.0` (no mencionado = negativo implícito) en las 13 columnas.
5. **Selección de clases** (`aplicar_seleccion_clases`, solo train/val): reduce a las clases activas y elimina
   estudios huérfanos (anti-ruido, sección 10).
6. **Split por paciente** (`_patient_split`, 90/10): pacientes disjuntos entre train y validación; evita data
   leakage (que el modelo memorice características de un paciente presente en ambos conjuntos).
7. **Datasets y transforms**: `CheXpertDataset`; en train hay augmentation (flip horizontal, afín suave); en
   validación e inferencia solo resize + normalización ImageNet.
8. **Test silver** (`construir_df_test_valid`): el `valid` oficial de Stanford, filtrado Frontal+AP; se reducen
   columnas a las clases activas, pero no se eliminan imágenes.

---

## 5. Pipeline de entrenamiento

- **AMP** (Automatic Mixed Precision) en CUDA; en CPU se usa autocast sin GradScaler.
- **Pérdida**: `BCEWithLogitsLoss` con `pos_weight = neg/pos` por clase, para compensar el fuerte desbalanceo.
- **Optimización**: AdamW + `ReduceLROnPlateau` (sobre la pérdida de validación).
- **Selección del mejor modelo (dentro del run)**: por **AUROC de validación** (independiente del umbral, más
  robusta al desbalanceo que F1 a 0.5).
- **Early stopping**: paciencia 6 sobre la pérdida de validación.
- **Checkpoints reanudables**: en cada época se guarda el estado completo (modelo, optimizador, scheduler,
  historial, mejor estado y early stopping) en `models/_ckpt_<backbone>_<class_config>.pth`; con `--resume` un
  entrenamiento interrumpido continúa desde la época siguiente. El checkpoint se elimina al terminar.
- **Salida**: el mejor modelo se restaura y se guarda como checkpoint candidato; el gate decide la promoción.

`num_workers=0` en Windows (el modelo de multiprocessing de PyTorch da problemas de pickling/spawn);
para acelerar la carga se pre-redimensiona el dataset a 224x224 (`preprocess_resize.py`).

---

## 6. Sistema de evaluación

Tres conjuntos:

- **Train / Validación**: split por paciente de `train_cheXbert.csv` (etiquetas automáticas). La validación
  selecciona el mejor modelo dentro del run.
- **Test "silver-standard"**: el `valid` oficial de Stanford (anotado por radiólogos), filtrado Frontal+AP
  (aproximadamente 169 imágenes), con pacientes disjuntos del train. Es la referencia de evaluación final.

Métricas (`calcular_metricas_completas`): accuracy, F1-macro/micro, **AUROC-macro** sobre clases evaluables,
**AUROC de las 5 de CheXpert**, **PR-AUC-macro**, y por clase: AUROC, PR-AUC, precision, recall, F1 y soporte.
Una clase sin ambos valores (0 y 1) no tiene AUROC definida: se marca como **no evaluable** y se omite de los
promedios (caso de `Fracture`, con 0 positivos en el test silver).

---

## 7. Gate de promoción del mejor modelo

`_gestionar_promocion` (en `main.py`, solo en runs reales):

1. El entrenamiento guarda un checkpoint **candidato**.
2. Se evalúa en el test silver.
3. `es_mejor` compara contra el campeón registrado: criterio primario **AUROC CheXpert-5 (test)**, con
   **F1-macro** como desempate dentro de `promotion_min_delta` (0.005). Si supera el margen, se **promueve**
   (pasa a `models/mejor_modelo_<backbone>_<class_config>.pth`) y se actualiza el registro; si no, se conserva
   el actual y se descarta el candidato.
4. **Siempre** se añade el experimento al historial.

El registro y el gate se indexan por par **`(backbone, class_config)`**: modelos con distinto número de clases
tienen cabezas incompatibles y no deben competir por la misma plaza. Artefactos: `models/best_model_registry.json`,
`logs/experiments.jsonl`, `logs/test_metrics_<backbone>_<class_config>.csv`.

---

## 8. Sistema de seguimiento de experimentos

Cada entrenamiento real genera `experiments/<run_id>/` (run_id = `AAAAMMDD-HHMMSS_<backbone>_<class_config>[_tag]`):

- `config.yaml` (snapshot), `manifest.json` (git, entorno, hardware, nº de parámetros, sha256 del checkpoint,
  tiempos, mejor época, promoción), `dataset.json` (tamaños, distribución por clase, clases ausentes/no
  evaluables, reporte ETL, `pos_weight`, provenance, pacientes y solapamiento), `history.csv`,
  `metrics_{val,test}.json` (+ CSV por clase), `predictions/*.npz`, `error_analysis/*.csv` (peores FP/FN),
  `plots/*.png` (curvas de aprendizaje, matrices de confusión, ROC/PR, resumen clínico) y `report.md`.
- Índice cross-run: `experiments/leaderboard.csv` (una fila por run, con `backbone`, `class_config` y métricas).

---

## 9. Generación de informes (web)

`report.py: build_report_pdf(contexto) -> bytes` (reportlab). Secciones: cabecera (fecha, modelo,
configuración, umbral, imagen), resumen de hallazgos, tabla de probabilidades por clase, gráfica de barras
(matplotlib), paneles original + Grad-CAM de las clases explicadas, métricas de validación del modelo si están
registradas, y aviso legal. `image_utils.empaquetar_imagenes_zip` produce el ZIP de imágenes.

---

## 10. Configuraciones de clases

`CLASS_CONFIGS` (en `models.py`) define, de forma explícita y fija, el conjunto de clases activo y la política
anti-ruido de eliminación de estudios huérfanos (cuyas únicas etiquetas positivas pertenecen a clases descartadas):

| Config | Clases | Política anti-ruido |
|---|---|---|
| `full13` | 13 | `ninguno` |
| `nofracture12` | 12 (sin Fracture) | `orfanos` (elimina estudios cuyo único positivo era Fracture) |
| `min5pct9` | 9 (sin Enlarged Cardiomediastinum, Lung Lesion, Pneumonia, Fracture) | `sin_positivos` (elimina todo estudio sin positivo activo) |

`num_classes` se **deriva** de `len(active_cols)` (no del valor estático de config). La `class_config` activa
aparece en el `run_id`, el nombre del checkpoint y el `leaderboard.csv`. Las 5 de CheXpert sobreviven en las
tres configuraciones. La eliminación de estudios se aplica solo a train/val; el test reduce columnas, no imágenes.

---

## 11. Pre-redimensionado del dataset

`preprocess_resize.py` genera una copia del dataset a 224x224 (`resize_tree`), no destructiva (carpetas nuevas
con sufijo `_224`), idempotente y resumible. Elimina el cuello de botella de decodificar/redimensionar JPEG en
caliente, dejando la GPU alimentada. Tras ejecutarlo, se apunta `data.images_root` y `data.test_images_root` a
las raíces `_224`.

---

## 12. Decisiones de diseño y riesgos

Decisiones:

1. **13 clases** (sin `Pleural Other`) y test silver Frontal+AP: coherencia con la literatura de CheXpert y con
   la distribución de entrada del entrenamiento.
2. **Selección del mejor modelo por AUROC** (no F1 a 0.5): independiente del umbral, más robusta al desbalanceo.
3. **Gate por par `(backbone, class_config)`**: evita comparar modelos con cabezas incompatibles.
4. **Sistema de experimentos file-based**: trazabilidad y reproducibilidad sin dependencias externas.
5. **Pre-resize en lugar de `num_workers>0`**: solución robusta al cuello de botella de E/S en Windows.

Riesgos y mitigaciones:

- **Test silver pequeño (169 imágenes)**: clases como `Fracture` (0 positivos) o `Lung Lesion` no son evaluables;
  se marcan explícitamente y se omiten de los promedios.
- **Sesgo de selección**: usar el test para el gate hace que su métrica deje de ser estrictamente insesgada; se
  mitiga con `promotion_min_delta` y el historial completo, y debe documentarse en la memoria.
- **Etiquetas de train automáticas** (CheXbert): contienen ruido frente a las etiquetas de radiólogo del test.

---

## 13. Notas del dataset CheXpert

CheXpert (Stanford) contiene radiografías de tórax con etiquetas de 14 observaciones. Valores: positivo (1),
negativo (0), incierto (-1) y no mencionado (en blanco).

- **Etiquetas de entrenamiento** (`train_cheXbert.csv`): extraídas automáticamente de los informes con el
  etiquetador **CheXbert**, recomendado por Stanford frente al CheXpert labeler.
- **Etiquetas de validación** (`valid.csv`): anotadas por tres radiólogos certificados; las anotaciones se
  binarizan (presente e incierto-probable como positivo; ausente e incierto-improbable como negativo) y se toma
  el **voto mayoritario** como verdad de referencia. Por eso el `valid` se usa como test "silver-standard".

---

## 14. Limitaciones conocidas

- `api.py` asume un checkpoint de 13 o 14 clases (usa `get_pathology_labels`); para servir modelos de 12 o 9
  clases habría que derivar las etiquetas de la `class_config`, como ya hace la interfaz web (`app.py`).
- La comparación de arquitecturas se realiza con una sola semilla; la incertidumbre se estima por bootstrap
  sobre el test, no entre semillas.

---

## 15. Despliegue (Docker, GHCR y Streamlit Cloud)

La imagen Docker (`python:3.12-slim`, PyTorch CPU) empaqueta `src/` y `config/`. No instala librerías de
sistema: Grad-CAM es una implementación propia (`src/grad_cam.py`, NumPy + Matplotlib) y ninguna dependencia
usa OpenCV. El `.dockerignore` excluye `models/`, por lo que **el checkpoint no va dentro de la imagen**: se
monta como volumen, igual que `config/` (lo que permite cambiar el modelo servido sin reconstruir). Dos formas
de ejecutar:

- **Opción A — construir en local:** `docker compose build && docker compose up`. Servicios `api` (uvicorn,
  puerto 8000) y `webapp` (Streamlit, 8501), ambos montan `models/`, `config/` y `logs/`.
- **Opción B — imagen publicada en GHCR:** `docker compose -f docker-compose.ghcr.yml up` (usa
  `image: ghcr.io/skullsupernova/mdl_tfm:latest`; requiere `docker login ghcr.io`).

**CI/CD (`.github/workflows/`):** `ci.yml` ejecuta la suite pytest en CPU en cada push/PR; `docker-publish.yml`
construye y publica la imagen en GitHub Container Registry en push a `main`, tags `v*` o manualmente
(`workflow_dispatch`). El modelo servido por defecto es `models/mejor_modelo_densenet121_full13.pth`.
Guía de uso paso a paso (login, pull) en [../README.md](../README.md) §2.

**Streamlit Community Cloud:** la interfaz web está publicada en https://mdltfm-mvb.streamlit.app/ y redespliega
en cada push a `main`. A diferencia de Docker, en este despliegue se **versionan** los modelos DenseNet-121
(~31 MB) y ResNet-50 (~98 MB), bajo el límite de 100 MB de GitHub; ConvNeXt/Swin/VGG lo superan y no se
publican. No requiere `packages.txt` ni dependencias de sistema.
