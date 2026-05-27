# CheXpert Pathology Classifier

Sistema de inteligencia artificial capaz de analizar radiografías de tórax y detectar automáticamente hasta 13 patologías clínicas, como edema pulmonar, cardiomegalia o neumonía.

El sistema incluye tres componentes independientes que se pueden usar por separado:
- **Interfaz web** — sube una radiografía y obtén el resultado visual en el navegador.
- **API REST** — integración para sistemas externos que envíen imágenes y reciban resultados en JSON.
- **Pipeline de entrenamiento** — re-entrena el modelo con nuevos datos desde línea de comandos.

---

## Contenido

1. [Requisitos previos](#1-requisitos-previos)
2. [Opción A — Uso con Docker (recomendado)](#2-opción-a--uso-con-docker-recomendado)
3. [Opción B — Instalación manual sin Docker](#3-opción-b--instalación-manual-sin-docker)
4. [Cómo usar la interfaz web](#4-cómo-usar-la-interfaz-web)
5. [Cómo usar la API REST](#5-cómo-usar-la-api-rest)
6. [Entrenamiento del modelo](#6-entrenamiento-del-modelo)
7. [Estructura del proyecto](#7-estructura-del-proyecto)
8. [Configuración avanzada](#8-configuración-avanzada)
9. [Ejecutar los tests](#9-ejecutar-los-tests)

---

## 1. Requisitos previos

### Si vas a usar Docker (Opción A)

Solo necesitas instalar **Docker Desktop**:

- **Windows / Mac**: descarga e instala desde [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: sigue las instrucciones de [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

Asegúrate de que Docker esté funcionando: abre una terminal y ejecuta `docker --version`. Si devuelve un número de versión, está listo.

### Si vas a instalar manualmente (Opción B)

- **Python 3.12** — descarga desde [https://www.python.org/downloads/](https://www.python.org/downloads/)
  - Durante la instalación en Windows, marca la opción **"Add Python to PATH"**
- **Git** — descarga desde [https://git-scm.com/downloads](https://git-scm.com/downloads)

---

## 2. Opción A — Uso con Docker (recomendado)

Esta opción no requiere instalar Python ni ninguna librería. Docker se encarga de todo.

### Paso 1: Descarga el proyecto

Abre una terminal (en Windows: busca "PowerShell" o "Símbolo del sistema") y ejecuta:

```bash
git clone https://github.com/SkullSupernova/MDL_TFM.git
cd MDL_TFM
```

Si no tienes Git, también puedes descargar el proyecto como archivo ZIP desde GitHub haciendo clic en el botón verde **"Code"** → **"Download ZIP"**, y descomprimirlo en una carpeta.

### Paso 2: Coloca el modelo entrenado

El modelo entrenado (archivo `.pth`) no se incluye en el repositorio por su tamaño. Colócalo en la carpeta `models/`:

```
MDL_TFM/
└── models/
    └── mejor_modelo_densenet121.pth   ← aquí
```

Si no tienes el archivo del modelo, consulta la sección [Entrenamiento del modelo](#6-entrenamiento-del-modelo).

### Paso 3: Construye y lanza los servicios

Desde la carpeta `MDL_TFM`, ejecuta:

```bash
docker compose build
docker compose up
```

La primera vez, `docker compose build` tardará varios minutos porque descarga Python y todas las librerías necesarias. Las siguientes veces será mucho más rápido.

### Paso 4: Abre la aplicación

Una vez que veas en la terminal mensajes como `Network URL: http://0.0.0.0:8501`, abre tu navegador y ve a:

- **Interfaz web**: [http://localhost:8501](http://localhost:8501)
- **API REST**: [http://localhost:8000/docs](http://localhost:8000/docs) (documentación interactiva)

### Para detener la aplicación

Pulsa `Ctrl + C` en la terminal donde está ejecutándose, o ejecuta en otra terminal:

```bash
docker compose down
```

---

## 3. Opción B — Instalación manual sin Docker

### Paso 1: Descarga el proyecto

```bash
git clone https://github.com/SkullSupernova/MDL_TFM.git
cd MDL_TFM
```

### Paso 2: Crea un entorno virtual de Python

Un entorno virtual es una "caja aislada" que instala las librerías del proyecto sin afectar al resto de tu sistema.

```bash
# Crear el entorno virtual
python -m venv .venv
```

```bash
# Activar el entorno virtual
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1

# En Windows (Símbolo del sistema / CMD):
.venv\Scripts\activate.bat

# En Mac / Linux:
source .venv/bin/activate
```

Sabrás que está activo porque verás `(.venv)` al inicio de la línea de comandos.

### Paso 3: Instala las dependencias

PyTorch (la librería de inteligencia artificial) tiene un instalador especial para evitar descargar versiones innecesariamente grandes:

```bash
# 1. Instala PyTorch solo para CPU (versión ligera, sin CUDA)
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Instala el resto de librerías del proyecto
pip install -r requirements.txt
```

> **Nota**: si tu ordenador tiene una tarjeta gráfica NVIDIA y quieres aprovecharla para entrenar más rápido, omite el primer comando y deja que `pip install -r requirements.txt` instale la versión con CUDA.

### Paso 4: Coloca el modelo entrenado

Igual que en la Opción A: copia el archivo `mejor_modelo_densenet121.pth` en la carpeta `models/`.

### Paso 5: Lanza la interfaz web

```bash
streamlit run src/app.py
```

O la API REST:

```bash
uvicorn src.api:app --reload
```

---

## 4. Cómo usar la interfaz web

Una vez que la aplicación esté en marcha y hayas abierto [http://localhost:8501](http://localhost:8501):

### Paso 1: Selecciona el modelo
En la barra lateral izquierda encontrarás el menú **"Modelo"**. Si has entrenado varios modelos con distintas arquitecturas, todos aparecerán aquí para que puedas compararlos.

### Paso 2: Ajusta el umbral de detección
El control deslizante **"Umbral de clasificación"** (por defecto 0.50) determina a partir de qué nivel de confianza se considera que una patología está presente. Un valor más bajo detecta más patologías pero puede dar más falsos positivos; uno más alto es más conservador.

### Paso 3: Selecciona la patología para GradCAM
El menú **"Patología para GradCAM"** elige qué zona de la radiografía ilumina el mapa de calor. Cada patología activa regiones distintas de la imagen.

### Paso 4: Carga la radiografía
Haz clic en **"Browse files"** y selecciona una imagen en formato JPEG o PNG. Puedes usar las radiografías de ejemplo que están en la carpeta `muestras_busqueda/`.

### Paso 5: Interpreta los resultados
Verás:
- **Columna izquierda**: la radiografía original.
- **Columna derecha**: el mapa GradCAM (zonas en rojo/amarillo = áreas que más influyeron en la predicción).
- **Gráfico de barras**: probabilidad de cada patología. Verde = detectada, gris = no detectada.
- **Mensaje de resultado**: lista de patologías detectadas con el umbral aplicado.
- **Botón "Descargar resultados (CSV)"**: exporta todas las probabilidades como archivo de texto separado por comas.

---

## 5. Cómo usar la API REST

La API permite integrar el clasificador en otros sistemas enviando imágenes y recibiendo los resultados en formato JSON.

### Verificar que la API está funcionando

```bash
curl http://localhost:8000/health
# Respuesta esperada: {"status": "ok"}
```

### Analizar una radiografía

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@ruta/a/tu/imagen.jpg"
```

Respuesta de ejemplo:
```json
{
  "threshold": 0.5,
  "probabilities": {
    "No Finding": 0.0312,
    "Cardiomegaly": 0.1205,
    "Edema": 0.8743,
    "Pleural Effusion": 0.6102
  },
  "detected_pathologies": ["Edema", "Pleural Effusion"]
}
```

También puedes usar el umbral personalizado añadiendo `?threshold=0.7` a la URL.

La documentación interactiva completa (con formulario de prueba) está disponible en [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 6. Entrenamiento del modelo

Para entrenar el modelo necesitas el dataset CheXpert descargado localmente. Actualiza las rutas en `config/config.yml` antes de continuar.

### Entrenamiento completo

```bash
python -m src.main
```

El modelo entrenado se guarda automáticamente en `models/mejor_modelo_densenet121.pth`.

### Entrenamiento rápido (para validar que todo funciona)

```bash
# Usa solo 500 imágenes y 2 épocas — tarda pocos minutos
python -m src.main --epochs 2 --subset 500
```

### Probar una arquitectura diferente

```bash
# Entrena un ResNet-50 en lugar del DenseNet-121 predeterminado
python -m src.main --model resnet50

# El resultado se guarda en models/mejor_modelo_resnet50.pth
# y aparecerá automáticamente en el selector de la interfaz web
```

Arquitecturas disponibles: `densenet121`, `resnet50`, `efficientnet_b0`, `efficientnet_b4`.

---

## 7. Estructura del proyecto

```
MDL_TFM/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # Pruebas automáticas en GitHub al hacer push
│
├── chexpert_csv/
│   └── train_cheXbert.csv          # Metadatos del dataset (índice de imágenes y etiquetas)
│
├── config/
│   └── config.yml                  # Todos los parámetros ajustables del proyecto
│
├── docs/                           # Documentación técnica adicional
│
├── models/                         # Modelos entrenados — NO incluidos en el repositorio
│   └── mejor_modelo_densenet121.pth
│
├── muestras_busqueda/              # Radiografías de ejemplo para probar la aplicación
│
├── notebook/                       # Cuadernos Jupyter del análisis y prototipo original
│
├── src/                            # Código fuente principal
│   ├── __init__.py
│   ├── api.py                      # Servidor de inferencia (FastAPI): GET /health, POST /predict
│   ├── app.py                      # Interfaz web interactiva (Streamlit)
│   ├── logging_config.py           # Sistema centralizado de registro de eventos
│   ├── main.py                     # Programa de entrenamiento ejecutable desde terminal
│   ├── models.py                   # Definición de arquitecturas y carga de checkpoints
│   ├── train.py                    # Bucle de entrenamiento con AMP, EarlyStopping y checkpoint
│   ├── utils.py                    # ETL del dataset, métricas y clases auxiliares
│   └── visualization.py            # GradCAM, matrices de confusión y gráficas de entrenamiento
│
├── test/                           # Suite de pruebas automáticas (47 tests)
│   ├── conftest.py                 # Datos sintéticos compartidos entre tests
│   ├── test_models.py
│   ├── test_train.py
│   └── test_utils.py
│
├── .dockerignore                   # Archivos excluidos al construir la imagen Docker
├── .gitignore                      # Archivos excluidos del control de versiones
├── Dockerfile                      # Instrucciones para construir la imagen Docker
├── docker-compose.yml              # Orquestación: lanza la API y la webapp a la vez
├── README.md                       # Este archivo
└── requirements.txt                # Versiones exactas de todas las librerías necesarias
```

---

## 8. Configuración avanzada

Todos los parámetros del proyecto se encuentran en `config/config.yml`. Puedes editarlo con cualquier editor de texto.

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `data.csv_path` | Ruta al archivo CSV con las etiquetas del dataset | `chexpert_csv/train_cheXbert.csv` |
| `data.images_root` | Carpeta raíz donde están las imágenes del dataset | `C:/CheXpertDataset/...` |
| `data.img_size` | Tamaño al que se redimensionan las imágenes antes de analizarlas | `224` (píxeles) |
| `data.train_split` | Proporción de pacientes usados para entrenar (el resto para validar) | `0.9` (90%) |
| `model.name` | Arquitectura de red neuronal a usar | `densenet121` |
| `model.num_classes` | Número de patologías que clasifica el modelo | `13` |
| `model.dropout` | Tasa de "olvido" durante el entrenamiento (reduce el sobreajuste) | `0.5` (50%) |
| `model.checkpoint_path` | Ruta del modelo que carga la API por defecto | `models/mejor_modelo_densenet121.pth` |
| `training.epochs` | Número máximo de rondas de entrenamiento | `50` |
| `training.batch_size` | Cuántas imágenes procesa a la vez durante el entrenamiento | `64` |
| `training.learning_rate` | Velocidad de aprendizaje de la red | `0.0001` |
| `training.threshold` | Umbral de confianza predeterminado para clasificar una patología como presente | `0.5` (50%) |
| `training.seed` | Número para hacer el entrenamiento reproducible | `42` |

---

## 9. Ejecutar los tests

Los tests verifican automáticamente que el código funciona correctamente. No requieren el dataset real.

```bash
# Activar el entorno virtual primero (Opción B)
.venv\Scripts\Activate.ps1

# Ejecutar todos los tests
.venv\Scripts\pytest.exe test/ -v
```

Resultado esperado: `47 passed` en pocos segundos.
