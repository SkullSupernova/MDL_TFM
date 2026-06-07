# Clasificador de Patologías Torácicas (CheXpert)

Sistema de apoyo al cribado que analiza radiografías de tórax y estima la probabilidad de **13 patologías**
(edema, cardiomegalia, derrame pleural, neumonía, entre otras), con explicabilidad visual mediante **Grad-CAM**
e informe en PDF.

Aviso: herramienta de demostración y de investigación (Trabajo de Fin de Máster). No es un producto médico y
no sustituye el diagnóstico de un profesional sanitario.

El proyecto tiene tres componentes independientes:

- **Interfaz web** (Streamlit): sube una radiografía y obtén probabilidades, mapas de calor e informe PDF.
- **API REST** (FastAPI): inferencia para sistemas externos (imagen a JSON).
- **Pipeline de entrenamiento** (CLI): entrena y compara modelos desde la línea de comandos.

Documentación complementaria:

- **[docs/GUIA_USO.md](docs/GUIA_USO.md)**: manual de uso paso a paso (web, CLI, interpretación de resultados).
- **[docs/ARQUITECTURA.md](docs/ARQUITECTURA.md)**: documento técnico (estructura, flujo de datos, pipeline, evaluación, decisiones de diseño).

---

## Contenido

1. [Requisitos previos](#1-requisitos-previos)
2. [Instalación con Docker (recomendado)](#2-instalación-con-docker-recomendado)
3. [Instalación manual](#3-instalación-manual)
4. [Uso rápido de la interfaz web](#4-uso-rápido-de-la-interfaz-web)
5. [Uso de la API REST](#5-uso-de-la-api-rest)
6. [Entrenamiento de modelos](#6-entrenamiento-de-modelos)
7. [Estructura del proyecto](#7-estructura-del-proyecto)
8. [Configuración](#8-configuración)
9. [Tests](#9-tests)

---

## 1. Requisitos previos

- **Docker Desktop** (opción recomendada): [docker.com](https://www.docker.com/products/docker-desktop), o
- **Python 3.12** y **Git** (instalación manual). En Windows, marca la opción "Add Python to PATH".

El **modelo entrenado** (`.pth`) no se incluye en el repositorio por su tamaño; colócalo en `models/` o
entrénalo (sección 6). Las **imágenes del dataset** tampoco se versionan; solo se necesitan para entrenar.

---

## 2. Despliegue con Docker (recomendado)

Hay dos formas de ejecutar la imagen. **Ambas montan el modelo y la configuración como
volúmenes** (la imagen no incluye el `.pth`; ver la nota al final).

### Opción A — Construir la imagen localmente (desde el código)

```bash
git clone https://github.com/SkullSupernova/MDL_TFM.git
cd MDL_TFM
# El modelo entrenado debe estar en models/ (por ejemplo,
# models/mejor_modelo_densenet121_full13.pth, al que apunta config/config.yml).
docker compose build
docker compose up
```

### Opción B — Usar la imagen publicada en GHCR (sin construir)

La imagen se publica en GitHub Container Registry automáticamente al hacer `git push` a `main`
(workflow `.github/workflows/docker-publish.yml`). Para usarla sin compilar:

```bash
# 1. Autenticarse en GHCR (paquete privado): token de GitHub con permiso read:packages.
echo <TOKEN> | docker login ghcr.io -u <usuario_github> --password-stdin

# 2. Descargar la imagen ya construida.
docker pull ghcr.io/skullsupernova/mdl_tfm:latest

# 3. Ejecutar (necesitas las carpetas models/ y config/ en el directorio actual).
docker compose -f docker-compose.ghcr.yml up
```

> La Opción B solo funciona **después** de haber hecho `git push` a `main` al menos una vez (para
> que el workflow construya y publique la imagen). Mientras tanto, usa la Opción A.

### Acceso y parada (ambas opciones)

- Interfaz web: [http://localhost:8501](http://localhost:8501)
- API REST (documentación interactiva): [http://localhost:8000/docs](http://localhost:8000/docs)
- Detener: `Ctrl + C`, o `docker compose down` (Opción A) /
  `docker compose -f docker-compose.ghcr.yml down` (Opción B).

> **El modelo no va dentro de la imagen.** El `.dockerignore` excluye `models/`, así que el
> checkpoint se monta como volumen (`./models`) y la imagen se mantiene ligera (sin datos de
> entrenamiento). El modelo servido se elige en `config/config.yml` (`model.checkpoint_path`),
> también montado como volumen, de modo que puedes cambiarlo sin reconstruir la imagen.

---

## 3. Instalación manual

```bash
git clone https://github.com/SkullSupernova/MDL_TFM.git
cd MDL_TFM
python -m venv .venv
# Activar el entorno:
#   Windows PowerShell:  .venv\Scripts\Activate.ps1
#   Mac/Linux:           source .venv/bin/activate

# PyTorch solo CPU (ligero). Para GPU NVIDIA, omite esta línea y usa la de requirements.
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Lanzar la web o la API:

```bash
streamlit run src/app.py            # interfaz web
uvicorn src.api:app --reload        # API REST
```

---

## 4. Uso rápido de la interfaz web

1. **Selecciona el modelo** en la barra lateral (aparecen todos los `models/mejor_modelo_*.pth`).
2. **Ajusta el umbral** de detección y el **máximo de explicaciones** Grad-CAM.
3. **Sube una radiografía** (JPEG o PNG). Hay ejemplos en `muestras_busqueda/`.
4. Verás:
   - **Resumen** con tarjetas: patologías detectadas, probabilidad máxima, patología principal y umbral.
   - **Explicabilidad**: para las patologías detectadas (mínimo 5), la radiografía original junto a su mapa
     de calor Grad-CAM (las zonas cálidas indican mayor influencia en la predicción).
   - **Gráfico de probabilidades** por patología (con su porcentaje) y tabla detallada.
   - **Descargas**: informe PDF, archivo ZIP con las imágenes (original y mapas de calor) y CSV del historial.
5. Tema claro u oscuro desde el menú de la esquina superior derecha, en Settings, Theme.

Guía detallada: **[docs/GUIA_USO.md](docs/GUIA_USO.md)**.

---

## 5. Uso de la API REST

```bash
curl http://localhost:8000/health
# {"status": "ok"}

curl -X POST http://localhost:8000/predict -F "file=@ruta/a/imagen.jpg"
```

Respuesta de ejemplo:

```json
{
  "threshold": 0.5,
  "probabilities": {"Edema": 0.8743, "Pleural Effusion": 0.6102, "...": 0.0},
  "detected_pathologies": ["Edema", "Pleural Effusion"]
}
```

Umbral personalizado: añade `?threshold=0.7`. Documentación interactiva en
[http://localhost:8000/docs](http://localhost:8000/docs).

---

## 6. Entrenamiento de modelos

Requiere el dataset CheXpert local y las rutas configuradas en `config/config.yml`.

```bash
# Entrenamiento completo (DenseNet-121, 13 clases)
python -m src.main

# Validación rápida del pipeline (pocos minutos): subconjunto y 1 época
python -m src.main --epochs 1 --subset 500 --val-subset 200

# Otra arquitectura y otra configuración de clases
python -m src.main --model convnext_tiny --class-config nofracture12

# Reanudar un entrenamiento interrumpido
python -m src.main --resume
```

- **Arquitecturas:** `densenet121`, `vgg16_bn`, `resnet50`, `convnext_tiny`, `swin_t`.
- **Configuraciones de clases (`--class-config`):** `full13` (13), `nofracture12` (12, sin Fracture),
  `min5pct9` (9, solo prevalencia mayor o igual al 5 por ciento).
- Cada entrenamiento real evalúa sobre el test silver-standard, decide la promoción del mejor modelo y genera
  una carpeta autocontenida en `experiments/<run_id>/` (configuración, métricas, curvas, informe). El índice
  general es `experiments/leaderboard.csv`.
- El checkpoint final se guarda como `models/mejor_modelo_<backbone>_<class_config>.pth`.

Detalle del pipeline y la evaluación: **[docs/ARQUITECTURA.md](docs/ARQUITECTURA.md)**.

---

## 7. Estructura del proyecto

```
MDL_TFM/
├── .github/workflows/ci.yml     # CI: tests automáticos en cada push
├── chexpert_csv/                # Metadatos del dataset (CSV de etiquetas)
├── config/config.yml            # Todos los parámetros del proyecto
├── docs/                        # GUIA_USO.md (manual) y ARQUITECTURA.md (técnico)
├── models/                      # Modelos entrenados (.pth, no versionados)
├── muestras_busqueda/           # Radiografías de ejemplo
├── notebook/                    # Cuadernos del análisis y prototipo
├── experiments/                 # Resultados por experimento y leaderboard (no versionado)
├── src/
│   ├── api.py                   # API FastAPI (/health, /predict)
│   ├── app.py                   # Interfaz web Streamlit
│   ├── main.py                  # CLI de entrenamiento (ETL, train, eval, informe)
│   ├── models.py                # Arquitecturas, configuraciones de clases, carga de checkpoints
│   ├── train.py                 # Bucle de entrenamiento (AMP, early stopping, checkpoints reanudables)
│   ├── evaluate.py              # Evaluación sobre el test silver-standard y métricas
│   ├── model_registry.py        # Registro y gate de promoción del mejor modelo
│   ├── experiment_tracker.py    # Sistema de seguimiento de experimentos (file-based)
│   ├── report.py                # Generación del informe PDF
│   ├── image_utils.py           # Validación de imagen y empaquetado ZIP
│   ├── preprocess_resize.py     # Pre-redimensionado del dataset a 224x224
│   ├── visualization.py         # Grad-CAM, matrices de confusión, curvas ROC/PR
│   ├── utils.py                 # ETL del dataset, métricas, callbacks
│   └── logging_config.py        # Logging centralizado
├── test/                        # Suite de tests (pytest)
├── Dockerfile, docker-compose.yml          # construir imagen localmente
├── docker-compose.ghcr.yml                  # ejecutar imagen publicada en GHCR
├── requirements.txt, README.md
```

---

## 8. Configuración

Todos los parámetros están en `config/config.yml`:

| Parámetro | Descripción | Por defecto |
|---|---|---|
| `data.images_root` / `data.test_images_root` | Carpetas de imágenes (train / test silver) | rutas locales |
| `data.class_config` | Conjunto de clases activo: `full13` / `nofracture12` / `min5pct9` | `full13` |
| `data.img_size` | Tamaño de entrada en píxeles | `224` |
| `data.train_split` | Proporción de pacientes para entrenar | `0.9` |
| `model.name` | Arquitectura (`densenet121`, `vgg16_bn`, `resnet50`, `convnext_tiny`, `swin_t`) | `densenet121` |
| `training.epochs` / `training.batch_size` | Épocas máximas / tamaño de lote (CLI: `--epochs` / `--batch-size`) | `50` / `64` |
| `training.learning_rate` / `training.seed` | Tasa de aprendizaje / semilla | `0.0001` / `42` |
| `training.threshold` | Umbral de detección por defecto | `0.5` |
| `training.promotion_min_delta` | Margen mínimo para promover el mejor modelo | `0.005` |

---

## 9. Tests

No requieren el dataset real (usan datos sintéticos):

```bash
.venv\Scripts\pytest.exe test/ -v        # Windows
.venv/bin/pytest test/ -v                # Mac/Linux
```

Resultado esperado: **113 passed**.
