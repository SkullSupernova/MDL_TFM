# CheXpert Pathology Classifier

Clasificador multietiqueta de patologías torácicas mediante DenseNet-121 y el dataset CheXpert.
Incorpora explicabilidad visual con GradCAM, una API REST de inferencia y una interfaz web interactiva.
El backbone es intercambiable (ResNet-50, EfficientNet-B0/B4) editando únicamente `config/config.yml`.

---

## Requisitos previos

| Dependencia | Versión mínima |
|---|---|
| Python | 3.12 |
| PyTorch | 2.12.0 (CPU o CUDA) |
| torchvision | 0.27.0 |
| Docker (opcional) | 24.x |

El proyecto gestiona todas las dependencias Python en un entorno virtual.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd MDL_TFM

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar el entorno (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 4. Instalar PyTorch CPU (evita descarga del wheel CUDA de PyPI)
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu

# 5. Instalar el resto de dependencias
pip install -r requirements.txt
```

---

## Quick Start

### Entrenamiento

```bash
# Entrenamiento completo con la configuración de config/config.yml
python -m src.main

# Entrenamiento rápido (backbone alternativo, 2 épocas, 500 imágenes)
python -m src.main --model resnet50 --epochs 2 --subset 500
```

El checkpoint se guarda automáticamente en `models/mejor_modelo_<backbone>.pth`.

### API REST de inferencia

```bash
uvicorn src.api:app --reload
```

```bash
# Verificar estado
curl http://localhost:8000/health

# Inferencia sobre una imagen
curl -X POST http://localhost:8000/predict \
  -F "file=@ruta/a/imagen.jpg"
```

### Interfaz web (Streamlit)

```bash
streamlit run src/app.py
```

Abre `http://localhost:8501` en el navegador. Permite cargar una radiografía,
seleccionar el modelo activo, visualizar el mapa GradCAM y exportar los resultados.

### Docker Compose (API + Webapp)

```bash
# Construir imágenes
docker compose build

# Levantar ambos servicios con los modelos montados
docker compose up

# API:    http://localhost:8000
# Webapp: http://localhost:8501
```

Los checkpoints no se incluyen en la imagen Docker. Se montan como volumen:
edita `docker-compose.yml` si los pesos están en una ruta diferente a `./models`.

---

## Estructura de directorios

```
MDL_TFM/
├── config/
│   └── config.yml          # Hiperparámetros, rutas y configuración del modelo
├── models/                 # Checkpoints entrenados (excluidos de git)
├── src/
│   ├── api.py              # API REST FastAPI (GET /health, POST /predict)
│   ├── app.py              # Interfaz web Streamlit
│   ├── logging_config.py   # Configuración centralizada de logging
│   ├── main.py             # Pipeline CLI de entrenamiento
│   ├── models.py           # Arquitecturas, builder y utilidades de checkpoint
│   ├── train.py            # Bucle de entrenamiento con AMP y EarlyStopping
│   ├── utils.py            # ETL, métricas y clases auxiliares
│   └── visualization.py    # GradCAM, matrices de confusión y gráficas
├── test/                   # Suite de tests pytest
├── notebook/               # Notebook de análisis original (excluido de git)
├── Dockerfile              # Imagen Docker para API y Webapp
├── docker-compose.yml      # Orquestación de servicios
└── requirements.txt        # Dependencias con versiones pinned
```

---

## Configuración (`config/config.yml`)

| Clave | Descripción | Valor por defecto |
|---|---|---|
| `data.csv_path` | Ruta al CSV de metadatos CheXpert | `chexpert_csv/train_cheXbert.csv` |
| `data.images_root` | Directorio raíz del dataset de imágenes | `C:/CheXpertDataset/...` |
| `data.img_size` | Tamaño de redimensionado de imágenes | `224` |
| `data.train_split` | Proporción de pacientes para entrenamiento | `0.9` |
| `model.name` | Backbone activo | `densenet121` |
| `model.num_classes` | Número de clases de salida | `13` |
| `model.dropout` | Tasa de dropout en la cabeza de clasificación | `0.5` |
| `model.hidden_units` | Neuronas en la capa oculta de la cabeza | `1024` |
| `model.checkpoint_path` | Checkpoint predeterminado para inferencia | `models/mejor_modelo_densenet121.pth` |
| `training.epochs` | Épocas máximas de entrenamiento | `50` |
| `training.batch_size` | Tamaño de batch (entrenamiento) | `64` |
| `training.learning_rate` | Tasa de aprendizaje inicial | `0.0001` |
| `training.threshold` | Umbral de decisión para clasificación | `0.5` |
| `training.seed` | Semilla aleatoria para reproducibilidad | `42` |

Para cambiar el backbone basta con editar `model.name`. Los valores soportados son:
`densenet121`, `resnet50`, `efficientnet_b0`, `efficientnet_b4`.

---

## Patologías activas (13 clases)

```
No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax,
Pleural Effusion, Fracture, Support Devices
```

---

## Tests

```bash
# Ejecutar la suite completa
.venv\Scripts\pytest.exe test/ -v

# Con cobertura (requiere pytest-cov)
.venv\Scripts\pytest.exe test/ -v --cov=src --cov-report=term-missing
```

La suite cubre 47 tests en tres módulos: `test_models`, `test_utils` y `test_train`.
No depende del dataset real; usa imágenes sintéticas generadas con `tmp_path`.
