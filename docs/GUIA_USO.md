# Guía de uso del sistema

Manual orientado al usuario y al Trabajo de Fin de Máster. Explica cómo utilizar la interfaz web, la API y la
línea de comandos, y cómo interpretar los resultados. Para la instalación, ver el [README](../README.md);
para el detalle técnico, [ARQUITECTURA.md](ARQUITECTURA.md).

Aviso: sistema de demostración e investigación. Las probabilidades reflejan la confianza del modelo, no
certezas clínicas, y no sustituyen el criterio de un profesional sanitario.

---

## 1. Qué hace el sistema

A partir de una radiografía de tórax frontal, el modelo estima la probabilidad de 13 patologías y genera
explicaciones visuales (Grad-CAM) que señalan las regiones de la imagen que más han influido en cada
predicción. El resultado puede descargarse como informe PDF.

---

## 2. Interfaz web

Arranque: `streamlit run src/app.py` (o mediante Docker; ver README). Se abre en `http://localhost:8501`.

### 2.1. Pasos

1. **Arquitectura y clases**: en la barra lateral, elige primero la **arquitectura** (DenseNet-121, ResNet-50,
   ConvNeXt-Tiny, Swin-Tiny) y después las **clases entrenadas** (13, 12 o 9; el segundo desplegable solo
   ofrece las configuraciones disponibles para la arquitectura elegida). La combinación determina el checkpoint
   `models/mejor_modelo_*.pth` que se carga.
2. **Umbral de clasificación**: nivel de confianza a partir del cual una patología se considera "detectada".
   Un umbral bajo detecta más patologías (más sensibilidad, más falsos positivos); uno alto es más conservador.
3. **Máximo de paneles (Grad-CAM)**: cuántas explicaciones visuales generar como máximo. Solo se muestran las
   patologías cuya probabilidad **supera el umbral** (las más probables primero); este valor limita cuántas.
4. **Cargar radiografía**: sube un archivo JPEG o PNG. Puedes usar los ejemplos de `muestras_busqueda/`.
5. Revisa los resultados (sección 2.2) y, si quieres, **descarga** el informe PDF, el ZIP de imágenes o el CSV
   del historial.

**Comparar dos modelos (opcional):** activa *Comparar con un segundo modelo* en la barra lateral y elige un
segundo (arquitectura + clases). Con la comparación activa, la sección de probabilidades muestra **tres
gráficas** (modelo A, modelo B y la comparativa sobre las patologías comunes), y la explicabilidad muestra
**tres imágenes por patología**: la original y el mapa de calor Grad-CAM de cada modelo. El **informe PDF**
descargado en este modo incluye ambos modelos: probabilidades de A y de B, la gráfica comparativa y el
Grad-CAM de los dos por patología.

El tema claro u oscuro se cambia desde el menú de la esquina superior derecha, en Settings, Theme.

### 2.2. Qué se muestra

- **Resumen**: tarjetas con el número de patologías detectadas, la probabilidad máxima, la patología principal
  y el umbral aplicado, más la lista de patologías detectadas.
- **Explicabilidad visual**: solo por cada patología **detectada** (probabilidad ≥ umbral), la radiografía
  original a la izquierda y su mapa de calor Grad-CAM a la derecha. Si ninguna supera el umbral no se muestran
  mapas (un mensaje invita a bajarlo). Las zonas cálidas (rojas) son las que más han influido en la predicción
  de esa patología; las frías (azules), las que menos.
- **Probabilidades por patología**: gráfico de barras con el porcentaje de cada clase (verde = detectada,
  gris = no detectada) y tabla detallada. El informe PDF incluye este mismo gráfico con cuadrícula y el valor
  numérico de cada barra. La gráfica aparece **antes** de los mapas Grad-CAM para verla sin bajar hasta el final.
- **Comparación de dos modelos** (si está activada): tres gráficas de barras (modelo A, modelo B y comparativa
  agrupada sobre las patologías comunes), resumen de decisiones coincidentes al umbral y tabla con la diferencia
  |A−B|; en la explicabilidad, cada patología muestra el original y el Grad-CAM de ambos modelos.

---

## 3. API REST

Arranque: `uvicorn src.api:app --reload` (`http://localhost:8000`).

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@imagen.jpg"
curl -X POST "http://localhost:8000/predict?threshold=0.7" -F "file=@imagen.jpg"
```

La respuesta incluye el umbral, las probabilidades por patología y la lista de patologías detectadas.
Documentación interactiva en `http://localhost:8000/docs`.

---

## 4. Entrenamiento desde la línea de comandos

Requiere el dataset local y las rutas en `config/config.yml`.

```bash
python -m src.main                                              # completo (config por defecto)
python -m src.main --epochs 1 --subset 500 --val-subset 200    # validación rápida del pipeline
python -m src.main --model swin_t --class-config nofracture12 --batch-size 16   # arquitectura y clases concretas
python -m src.main --resume                                    # reanudar un run interrumpido
```

Opciones principales:

| Opción | Significado |
|---|---|
| `--model` | `densenet121`, `vgg16_bn`, `resnet50`, `convnext_tiny`, `swin_t` |
| `--class-config` | `full13` (13), `nofracture12` (12), `min5pct9` (9) |
| `--epochs` | Número máximo de épocas |
| `--batch-size` / `--inference-batch-size` | Tamaño de lote train / inferencia (bajar en 8 GB para VGG16-BN y Swin) |
| `--subset` / `--val-subset` | Limitar train / validación (pruebas rápidas) |
| `--seed` | Semilla (reproducibilidad / multi-semilla) |
| `--tag` | Etiqueta para identificar el experimento |
| `--resume` | Continuar desde el checkpoint por época si existe |

Dónde quedan los resultados:

- Carpeta autocontenida por run en `experiments/<run_id>/` (configuración, métricas, curvas, predicciones,
  análisis de error e informe `report.md`).
- Fila resumen en `experiments/leaderboard.csv`.
- Mejor modelo en `models/mejor_modelo_<backbone>_<class_config>.pth` (si supera al campeón anterior).

Para acelerar el entrenamiento, ejecuta primero el pre-redimensionado una sola vez y apunta las rutas a las
carpetas `_224`:

```bash
python -m src.preprocess_resize
```

---

## 5. Interpretación de las métricas

- **AUROC** (área bajo la curva ROC): capacidad de ordenar correctamente positivos frente a negativos.
  0.5 = azar; 1.0 = perfecto. Es independiente del umbral.
- **AUROC CheXpert-5**: AUROC media de las 5 patologías oficiales de CheXpert (Atelectasis, Cardiomegaly,
  Consolidation, Edema, Pleural Effusion). Es la **métrica principal** del proyecto, por estar bien
  representadas en el test y ser el estándar de comparación con la literatura.
- **PR-AUC** (precision-recall): más informativa que la AUROC bajo fuerte desbalanceo (clases poco frecuentes).
- **F1-macro**: media no ponderada del F1 por clase (a umbral 0.5); penaliza el mal rendimiento en clases
  minoritarias.
- **Soporte**: número de positivos reales de la clase en el conjunto. Con soporte 0, la clase es **no
  evaluable** (no se puede calcular su AUROC) y se omite de los promedios. En el test silver ocurre con
  `Fracture` (0 positivos) y casi con `Lung Lesion`.

Lectura recomendada: comparar modelos por **AUROC CheXpert-5** sobre el test, y revisar las métricas por clase
(con su soporte) para entender el comportamiento en patologías concretas.

---

## 6. Comparación de arquitecturas

Para comparar arquitecturas de forma justa, se fija la misma configuración de clases y el mismo protocolo, y se
varía solo el backbone:

```bash
python -m src.main --class-config nofracture12 --model densenet121
python -m src.main --class-config nofracture12 --model vgg16_bn        --batch-size 16
python -m src.main --class-config nofracture12 --model resnet50
python -m src.main --class-config nofracture12 --model convnext_tiny   --batch-size 16
python -m src.main --class-config nofracture12 --model swin_t          --batch-size 16
```

Los resultados se agregan en `experiments/leaderboard.csv`. La incertidumbre se estima por bootstrap sobre el
test (resampleo de las imágenes), ya que se entrena con una sola semilla.

---

## 7. Problemas comunes

- **No aparece ningún modelo en la web**: no hay `models/mejor_modelo_*.pth`. Entrena un modelo o coloca un
  checkpoint en `models/`.
- **La imagen subida no es válida**: la web valida tamaño y resolución, y avisa si la imagen parece estar en
  color (las radiografías son en escala de grises); los resultados sobre una imagen no radiográfica no son fiables.
- **Entrenamiento muy lento**: ejecuta el pre-redimensionado (sección 4) y apunta las rutas a `_224`.
- **Memoria de GPU insuficiente** con VGG16 o ConvNeXt-Tiny: reduce `training.batch_size` en `config/config.yml`.
