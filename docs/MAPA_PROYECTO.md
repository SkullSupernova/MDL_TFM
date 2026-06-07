# Mapa exhaustivo del proyecto MDL_TFM

> Documento de referencia cruzada. Describe **dónde encontrar cada cosa** del proyecto de código
> (`d:\ProyectosPython\MDL_TFM`) para poder citarlo desde la redacción del TFM escrito. Todos los
> datos técnicos de este documento están verificados contra el código fuente, no contra resúmenes.
> Última verificación: 2026-06-07.

---

## 0. Qué es el proyecto en una frase

Clasificación **multietiqueta** de 13 patologías torácicas a partir de radiografías de tórax del dataset
**CheXpert** (Stanford), con explicabilidad **Grad-CAM**, evaluación sobre un test *silver-standard*, un gate
de promoción automático del mejor modelo, sistema de seguimiento de experimentos, **API REST** (FastAPI) e
**interfaz web** (Streamlit). Incluye una comparación de **5 arquitecturas** × **3 configuraciones de clases**.

---

## 1. Mapa de directorios (raíz)

| Ruta | Versionado | Contenido |
|------|:---:|-----------|
| `src/` | sí | Todo el código fuente (15 módulos). Ver §2. |
| `config/config.yml` | sí | Único punto de configuración (datos, modelo, entrenamiento, experimentos). Ver §3. |
| `test/` | sí | Suite pytest (datos sintéticos, sin dataset real). Ver §4. |
| `docs/` | sí | `ARQUITECTURA.md` (técnico), `GUIA_USO.md` (manual), este `MAPA_PROYECTO.md`. |
| `README.md` | sí | Documentación de usuario (instalación, quick start). |
| `CLAUDE.md` | sí | Instrucciones del repo (contexto, fases, reglas de estilo y commits). |
| `notebook/` | sí | Cuadernos del análisis/prototipo original. Ver §8. |
| `chexpert_csv/` | sí | CSV de etiquetas del dataset (`train_cheXbert.csv`). Ver §7. |
| `muestras_busqueda/` | sí | ~60 radiografías frontales de ejemplo para probar la web. |
| `.github/workflows/ci.yml` | sí | CI en GitHub Actions (pytest sobre CPU). |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | sí | Despliegue. Ver §9. |
| `requirements.txt`, `requirements-dev.txt` | sí | Dependencias prod / dev. |
| `pytest.ini` | sí | Configuración de pytest. |
| `models/` | **no** (`.gitignore`) | Checkpoints `.pth` (excepto `best_model_registry.json`, que sí se versiona). Ver §5. |
| `experiments/` | **no** | Una carpeta autocontenida por run + `leaderboard.csv`. Ver §6. |
| `logs/` | **no** | `app.log`, `experiments.jsonl`, `test_metrics_*.csv`. |
| `data/` | **no** | (Reservado; las imágenes reales viven fuera del repo, en `C:/CheXpertDataset/`.) |
| `.claude/ESTADO_PROYECTO.md` | **no** | Estado de continuidad entre sesiones (avances, pendientes, decisiones, §10 plan de comparación). |

---

## 2. Código fuente — `src/` (qué hace cada módulo y dónde están las funciones clave)

### 2.1. `models.py` — arquitecturas, dataset, clases, Grad-CAM
- **Listas de patologías:** `CHEXPERT_PATHOLOGY_COLS` (13 activas), `CHEXPERT_PATHOLOGY_COLS_14` (14 originales,
  compatibilidad), `CHEXPERT_COMPETITION_5` (las 5 oficiales: Atelectasis, Cardiomegaly, Consolidation, Edema,
  Pleural Effusion).
- **Configuraciones de clases:** `CLASS_CONFIGS` (dict fijo y explícito) y `get_active_pathology_cols()`.
- **Dataset:** `CheXpertDataset` (lee imagen del disco + etiquetas; aplica transforms).
- **Construcción del modelo:** `build_model()` — soporta `densenet121`, `vgg16_bn`, `resnet50`,
  `convnext_tiny`, `swin_t`. Cabeza de clasificación: `Linear → ReLU → Dropout → Linear`.
- **Carga de checkpoints:** `load_checkpoint()` (infiere nº de clases, detecta cabeza simple vs compuesta),
  `parse_checkpoint_filename()` (extrae `(backbone, class_config)` del nombre del fichero).
- **Grad-CAM:** `get_grad_cam_layer()` (capa objetivo por backbone), `get_grad_cam_reshape()` (reshape de
  tokens `(B,H,W,C)→(B,C,H,W)` para transformers como Swin; `None` para CNN).

### 2.2. `utils.py` — ETL, métricas, callbacks
- **Entorno:** `setup_environment()`, `set_seed()`.
- **ETL whitelist:** `filtrar_chexpert_dataset_whitelist()`, `aplicar_filtrado_proyecto()` — retiene Frontal+AP,
  valores `{0, 1, NaN}` (descarta incertidumbre `-1`), elimina inconsistencias de `No Finding`.
- **Selección de clases (anti-ruido):** `aplicar_seleccion_clases()` — reduce a clases activas y elimina
  estudios huérfanos según la política de la config (solo train/val).
- **Test silver:** `construir_df_test_valid()`, `mapear_ruta_valid_definitiva()`, `obtener_ruta_absoluta_train()`.
- **Métricas:** `calculate_metrics()`, `auroc_macro()`, `auc_por_clase()`, `pr_auc_macro()`,
  `pr_auc_por_clase()`, `distribucion_clases()`, `contar_parametros()`.
- **Callbacks:** `EarlyStopping`, `ModelCheckpoint`. **Auditoría:** `auditar_dataset()`.

### 2.3. `train.py` — bucle de entrenamiento
- `train_model()` — AMP (Automatic Mixed Precision) en CUDA; selección de la mejor época por **AUROC de
  validación**; early stopping; **checkpoints reanudables** (guarda por época modelo+optimizador+scheduler+
  historial+mejor estado en `models/_ckpt_<backbone>_<class_config>.pth`; `--resume` continúa).

### 2.4. `main.py` — orquestador CLI
- `main()` — pipeline completo. `_patient_split()` (split por paciente sin leakage), `_gestionar_promocion()`
  (gate de promoción), `_parse_args()`, `load_config()`.
- **Pérdida:** `BCEWithLogitsLoss` con `pos_weight = neg/pos` por clase (`main.py:442-444`).
- **Optimizador:** `AdamW` (`main.py:446`). **Scheduler:** `ReduceLROnPlateau` sobre pérdida de validación.
- **Augmentation (solo train):** `RandomHorizontalFlip(p=0.5)` + `RandomAffine(degrees=10, translate=(0.05,0.05))`
  (`main.py:345-346`). Validación/inferencia: solo `Resize(224)` + normalización ImageNet.
- **Flags CLI:** `--model`, `--class-config`, `--seed`, `--epochs`, `--batch-size`, `--inference-batch-size`,
  `--scheduler-patience`, `--subset`, `--val-subset`, `--tag`, `--resume`.

### 2.5. Evaluación y registro
- `evaluate.py` — evaluación sobre el test silver, independiente del entrenamiento (`evaluate_model`,
  `calcular_metricas_completas`, `evaluar_loader`, `evaluar_test`). Vuelca `logs/test_metrics_<backbone>_<config>.csv`.
- `model_registry.py` — `cargar_registro()`, comparación `es_mejor` (gate), historial append-only. Indexado por
  par `(backbone, class_config)`. Escribe `models/best_model_registry.json`.
- `experiment_tracker.py` — clase `ExperimentTracker`: genera todo `experiments/<run_id>/` y la fila de
  `leaderboard.csv`. Ver §6 para el contenido.

### 2.6. Visualización e informes
- `visualization.py` — `graficar_entrenamiento()` (curvas), `plot_confusion_matrices()`, `plot_roc_curves()`,
  `plot_pr_curves()`, `matriz_resumen_multietiqueta()` (resumen clínico), `inspeccion_visual()`.
- `report.py` — `build_report_pdf()` (informe PDF con reportlab para la web).
- `image_utils.py` — `validar_imagen_radiografia()`, `empaquetar_imagenes_zip()`.

### 2.7. Despliegue y soporte
- `api.py` — API FastAPI. Endpoints: `GET /health` (`api.py:107`), `POST /predict` (`api.py:117`). Modelo cargado
  una vez en `lifespan`; corre en **CPU**. Respuesta: `{threshold, probabilities, detected_pathologies}`.
- `app.py` — interfaz web Streamlit (uploader, selector de modelo, paneles original|Grad-CAM top-N, gráfica
  Altair, descargas PDF/ZIP/CSV, validación de imagen, tarjetas de resumen).
- `preprocess_resize.py` — `resize_tree()` + CLI: pre-redimensionado del dataset a 224×224 (resumible,
  idempotente, no destructivo; genera raíces `_224`). Elimina el cuello de botella de E/S.
- `logging_config.py` — `get_logger()` (stdout + `logs/app.log` con rotación).

---

## 3. Configuración — `config/config.yml`

Único punto de control. Para cambiar de arquitectura basta editar `model.name`.

| Sección | Claves principales |
|---------|--------------------|
| `data` | `csv_path`, `images_root`, `test_csv_path`, `test_images_root`, `batches`, `img_size` (224), `train_split` (0.9), **`class_config`** (full13/nofracture12/min5pct9) |
| `model` | **`name`** (backbone), `num_classes` (referencia; se deriva de la config), `dropout` (0.5), `hidden_units` (1024), `pretrained` (true), `checkpoint_path` |
| `training` | `batch_size` (64), `inference_batch_size` (256), `learning_rate` (1e-4), `weight_decay` (0.01), `epochs` (50), `early_stopping_patience` (8), `scheduler_patience` (2), `threshold` (0.5), `seed` (42), `promotion_min_delta` (0.005) |
| `pathologies` | las 13 clases activas |
| `experiments` | `root` (carpeta), `n_worst_cases` (20) |

> Las imágenes están **pre-redimensionadas**: `images_root` y `test_images_root` apuntan a raíces con sufijo `_224`.

---

## 4. Tests — `test/`

`test_models.py`, `test_utils.py`, `test_train.py`, `test_evaluate.py`, `test_registry.py`,
`test_experiment_tracker.py`, `test_image_utils.py`, `test_report.py`, `test_preprocess_resize.py`.
`conftest.py` (fixtures compartidas). Ejecutar: `.venv\Scripts\pytest.exe test/ -v`. No dependen del dataset
real (datos sintéticos / `tmp_path`). El README reporta **113 passed** (el plan de la fase de arquitecturas
menciona 118 tras añadir Swin/VGG; el número exacto depende del estado del repo).

---

## 5. Modelos — `models/`

| Patrón | Versionado | Significado |
|--------|:---:|-------------|
| `best_model_registry.json` | **sí** | Registro de campeones por par `(backbone, class_config)`: métricas val+test, hiperparámetros, ruta y sha256 del checkpoint, timestamp. |
| `mejor_modelo_<backbone>_<class_config>.pth` | no | Checkpoint de producción (campeón de ese par). |
| `mejor_modelo_densenet121.pth` | no | Checkpoint *throwaway* legado (formato antiguo, sin config). |
| `_candidato_<backbone>_<class_config>.pth` | no | Candidato en evaluación (antes del gate). |
| `_ckpt_<backbone>_<class_config>.pth` | no | Checkpoint por época (reanudable con `--resume`); se borra al terminar el run. |

> Tamaños aproximados: DenseNet-121 ~31 MB, ResNet-50/ConvNeXt/Swin ~100–110 MB, VGG16-BN ~2 GB (por su cabeza FC).

---

## 6. Experimentos — `experiments/<run_id>/`

`run_id = AAAAMMDD-HHMMSS_<backbone>_<class_config>[_tag]`. Carpeta autocontenida, gitignored. Contenido:

```
config.yaml                              snapshot de config.yml del run
manifest.json                            git commit, entorno/versiones, hardware, nº de parámetros,
                                         sha256 del checkpoint, tiempos, mejor época, promoción
dataset.json                             tamaños, distribución por clase, clases ausentes/no evaluables,
                                         reporte ETL, pos_weight, provenance, pacientes y solapamiento
history.csv                              loss/AUROC/F1 por época
metrics_val.json    / metrics_test.json        métricas globales (val y test)
metrics_val_per_class.csv / metrics_test_per_class.csv   AUROC/PR-AUC/P/R/F1/soporte por clase
predictions/val_predictions.npz / test_predictions.npz   predicciones crudas (numpy)
error_analysis/val_worst_cases.csv / test_worst_cases.csv   top-20 FP/FN con rutas de imagen
plots/  learning_curves.png · roc_curves_test.png · pr_curves_test.png ·
        confusion_matrices_val.png · confusion_matrices_test.png · clinical_summary_test.png
report.md                                informe legible por run (incluye clases no evaluables)
```

**Índice cross-run:** `experiments/leaderboard.csv` — una fila por run con columnas: `run_id, timestamp,
backbone, class_config, tag, epochs_ejecutadas, lr, batch_size, seed, val_auroc_best, test_auroc_chexpert5,
test_auroc_macro, test_pr_auc_macro, test_f1_macro, duracion_min, promovido, git_commit`.

---

## 7. Datos y dataset

- **Etiquetas de entrenamiento:** `chexpert_csv/train_cheXbert.csv` — etiquetas automáticas del etiquetador
  **CheXbert** (Stanford). También están `train_visualCheXbert.csv` y `CHEXPERT DEMO.xlsx`.
- **Test silver-standard:** `valid.csv` oficial de Stanford, anotado por **tres radiólogos** (voto mayoritario),
  filtrado Frontal+AP → **169 imágenes**. Ruta en `config.yml → data.test_csv_path`.
- **Imágenes (NO en el repo):** `C:/CheXpertDataset/chexpertchestxrays-u20210408` (originales) y
  `..._224` (copia pre-redimensionada, las que realmente se usan).
- **Tamaños tras ETL (config nofracture12):** 95.546 estudios → 94.659 (−887 huérfanos); split por paciente
  90/10 = **85.273 train / 9.386 val**. Test = 169.

**13 clases activas:** No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema,
Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Fracture, Support Devices.
(Se excluye `Pleural Other` de las 14 originales por baja prevalencia.)

**Valores de etiqueta CheXpert:** 1 (positivo), 0 (negativo), -1 (incierto, **descartado** por la whitelist),
en blanco/NaN (no mencionado → imputado a 0).

**Configuraciones de clases (`CLASS_CONFIGS`):**

| Config | Clases | Anti-ruido | Estudios eliminados (train/val) |
|--------|:---:|------------|:---:|
| `full13` | 13 | `ninguno` | — |
| `nofracture12` | 12 (sin Fracture) | `orfanos` (elimina estudios cuyo único positivo era Fracture) | −887 |
| `min5pct9` | 9 (sin Enlarged Cardiomediastinum, Lung Lesion, Pneumonia, Fracture) | `sin_positivos` (elimina todo estudio sin positivo activo) | −4622 |

Las 5 de CheXpert sobreviven en las tres configs. El anti-ruido solo afecta a train/val; el test reduce
columnas pero no elimina imágenes.

---

## 8. Notebooks — `notebook/` (origen histórico)

- `T02_Analisis_DenseNet121.ipynb` — **fuente de verdad original** del pipeline (DenseNet-121, EPOCHS=10,
  BATCH_SIZE=64, sin pre-resize, sin filtrado ETL whitelist → 223.414 imágenes, entrenamientos de 8 h+).
- `DL_VIS_Practica_final_MValbuena.ipynb` — práctica previa de la asignatura (EPOCHS=10, mismo orden de magnitud).
- `T01_Analisis_previo.ipynb` — análisis exploratorio (EDA). `T03_Preprocess.ipynb` — preprocesado.
- Artefactos legados: `mejor_modelo_chexpert.pth`, `historial_entrenamiento.pkl`.

> **Diferencia clave de tiempos** (útil para justificar en el TFM): el notebook entrena sobre las 223 k imágenes
> sin filtrar y redimensionando JPEG grandes en caliente → ~40 min/época. El pipeline `src/` filtra por ETL a
> ~85 k imágenes Frontal+AP y usa imágenes pre-redimensionadas a 224×224 → ~8–10 min/época (≈3–4× más rápido).

---

## 9. Despliegue

- **`Dockerfile`** — imagen `python:3.12-slim`, instala PyTorch **CPU** + dependencias; copia `src/` y `config/`.
- **`docker-compose.yml`** — dos servicios: `api` (uvicorn, puerto **8000**) y `webapp` (streamlit, puerto
  **8501**), ambos montan `models/` (read-only) y `logs/`. `webapp` depende de `api`.
- **Arranque local:** `streamlit run src/app.py` (web), `uvicorn src.api:app --reload` (API).
- **CI:** `.github/workflows/ci.yml` ejecuta la suite pytest en CPU (no despliega).
- **Limitación documentada:** `api.py` asume checkpoint de 13/14 clases (usa `get_pathology_labels`); para
  servir modelos de 12 o 9 clases habría que derivar las etiquetas de la `class_config` (la web ya lo hace).

---

## 10. Pipeline de entrenamiento (resumen para metodología del TFM)

1. **Carga** de `train_cheXbert.csv`.
2. **ETL whitelist** (`aplicar_filtrado_proyecto`): Frontal+AP, valores `{0,1,NaN}`, sin inconsistencias.
3. **Localización** de rutas absolutas entre batches; descarte de imágenes ausentes.
4. **Imputación** `NaN → 0`.
5. **Selección de clases** + anti-ruido (solo train/val).
6. **Split por paciente** 90/10 (pacientes disjuntos, sin data leakage).
7. **Datasets + transforms** (augmentation solo en train).
8. **Entrenamiento:** AMP, `BCEWithLogitsLoss` con `pos_weight=neg/pos`, **AdamW** (lr 1e-4, wd 0.01),
   `ReduceLROnPlateau`, selección de época por **AUROC de validación**, early stopping (paciencia 8).
9. **Evaluación** en val y en test silver (`evaluate.py`).
10. **Gate de promoción** (`_gestionar_promocion`): métrica primaria **AUROC CheXpert-5 (test)**, desempate
    F1-macro dentro de `promotion_min_delta` (0.005); indexado por `(backbone, class_config)`.
11. **Documentación** del run (`ExperimentTracker`) + fila en `leaderboard.csv`.

**Hardware:** NVIDIA RTX 4060 (8 GB), CUDA. **Protocolo de comparación:** 1 semilla por experimento +
pre-resize; incertidumbre por **bootstrap sobre el test** (no entre semillas).

---

## 11. Métricas (definiciones para el TFM)

- **AUROC** — área bajo la curva ROC; ordena positivos frente a negativos; independiente del umbral (0.5=azar).
- **AUROC CheXpert-5** — AUROC media de las 5 patologías oficiales; **métrica principal** de promoción.
- **AUROC-macro evaluable** — media sobre las clases con soporte ≥1 positivo (las no evaluables se omiten).
- **PR-AUC** — área precision-recall; más informativa bajo fuerte desbalanceo.
- **F1-macro** — media no ponderada del F1 por clase a umbral 0.5; desempate del gate.
- **Soporte** — nº de positivos reales; soporte 0 → clase **no evaluable** (p. ej. `Fracture` en el test silver).

---

## 12. Resultados experimentales (al 2026-06-07)

Comparación de arquitecturas — **AUROC CheXpert-5 sobre el test silver** (169 img). 1 semilla (42), batch 64.

| Arquitectura | Parámetros | nofracture12 (12) | min5pct9 (9) |
|--------------|:---:|:---:|:---:|
| **ConvNeXt-Tiny** | ~28 M | **0.8536** | **0.8505** |
| ResNet-50 | ~25 M | 0.8517 | 0.8422 |
| DenseNet-121 | ~7 M | 0.8497 | 0.8256 |
| Swin-Tiny | ~28 M | 0.8438 | _(en curso / pendiente)_ |
| VGG16-BN | ~138 M | _abandonado (~45 h)_ | — |

Detalle adicional por run (val_auroc_best / F1-macro / PR-AUC-macro / duración):
- DenseNet-121 nofracture12: 0.8183 / 0.4968 / 0.6136 / 131 min (early stop ép. 13).
- ConvNeXt-Tiny nofracture12: 0.8179 / 0.4809 / 0.6181 / 147 min (ép. 12).
- ResNet-50 nofracture12: 0.8159 / 0.5062 / 0.5992 / 138 min (ép. 13).
- Swin-Tiny nofracture12: 0.8195 / 0.5040 / 0.6162 / 186 min (ép. 17).
- DenseNet-121 min5pct9: 0.8252 / 0.5768 / 0.7038 / 96 min (ép. 14).
- ResNet-50 min5pct9: 0.8275 / 0.6051 / 0.7091 / 85 min (ép. 13).
- ConvNeXt-Tiny min5pct9: 0.8295 / 0.6022 / 0.7331 / 85 min (ép. 12).

**Observaciones para la discusión:**
- ConvNeXt-Tiny lidera la AUROC CheXpert-5 en ambas configs, pero las diferencias entre las cuatro
  arquitecturas son pequeñas (~0.01) sobre un test de 169 imágenes → previsible **solapamiento de los IC
  bootstrap** (Fase 5 pendiente); la conclusión honesta puede ser "equivalencia estadística".
- Reducir clases (full13 → nofracture12 → min5pct9) **sube mucho F1-macro y PR-AUC** (de ~0.48–0.50 / ~0.62 a
  ~0.60 / ~0.73) al eliminar clases raras difíciles, sin penalizar la AUROC CheXpert-5.

---

## 13. Estado del proyecto y trabajo pendiente

- **Fases 1–9 (implementación) + Fase 3 (pipeline de comparación):** completas.
- **Fase 4 (entrenamientos):** en curso — completadas las 4 arquitecturas rápidas en nofracture12 y la mayor
  parte de min5pct9; VGG16-BN abandonado por coste (~45 h). Seguimiento en `.claude/ESTADO_PROYECTO.md` §10.
- **Fase 5 (pendiente):** agregar `leaderboard.csv` con **intervalos de confianza bootstrap** sobre el test.
- **Fase 6 (pendiente):** informe comparativo + actualización de documentación.

> Para el detalle vivo de avances/pendientes/decisiones, la fuente es `.claude/ESTADO_PROYECTO.md`.
> Para el detalle técnico narrativo, `docs/ARQUITECTURA.md`. Para uso, `docs/GUIA_USO.md`.

---

## 14. Notas de precisión (discrepancias resueltas)

- El optimizador real es **AdamW** (`src/main.py:446`), no Adam (algún resumen lo cita impreciso).
- La augmentation de train es **flip horizontal + afín suave** (`src/main.py:345-346`), no solo flip.
- `num_classes` se **deriva** de `len(active_cols)` en tiempo de ejecución; el valor de `config.model.num_classes`
  es solo de referencia.
