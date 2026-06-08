# Comparativa de arquitecturas (CheXpert) — informe de resultados

> Informe de la Fase 6 del plan de comparación. Resume los resultados experimentales con
> intervalos de confianza bootstrap y deriva las conclusiones para la memoria del TFM.
> **Estado:** `nofracture12` completa con las **5 arquitecturas** (incl. VGG16-BN); `min5pct9` con 4
> (VGG16-BN en curso) + ablación DenseNet-121 en full13. 10 runs reales agregados. Fuentes:
> `experiments/leaderboard.csv` y `experiments/leaderboard_ci.csv`.

---

## 1. Objetivo y protocolo experimental

Comparar cinco arquitecturas de visión sobre la clasificación multietiqueta de patologías
torácicas (CheXpert), bajo un protocolo idéntico, y medir el efecto de reducir el conjunto de
clases. Las arquitecturas cubren los paradigmas principales:

| Arquitectura | Paradigma | Parámetros (aprox.) |
|--------------|-----------|:---:|
| DenseNet-121 | CNN densa (referencia en CXR) | ~7 M |
| ResNet-50 | CNN residual | ~25 M |
| ConvNeXt-Tiny | CNN moderna (SOTA) | ~28 M |
| Swin-Tiny | Transformer jerárquico (SOTA) | ~28 M |
| VGG16-BN | CNN secuencial (referencia histórica) | ~138 M |

**Protocolo común:**
- **Datos:** CheXpert, etiquetas de entrenamiento CheXbert; split por paciente 90/10 (sin leakage).
- **Test:** conjunto "silver-standard" (valid oficial de Stanford, anotado por radiólogos),
  filtrado Frontal+AP → **169 imágenes**. Mismo test para todos los runs.
- **Entrenamiento:** `BCEWithLogitsLoss` con `pos_weight=neg/pos`, AdamW (lr 1e-4, wd 0.01),
  `ReduceLROnPlateau`, AMP, selección de época por AUROC de validación, early stopping (paciencia 8).
- **Reproducibilidad:** **una sola semilla** (42) por experimento + dataset pre-redimensionado a 224×224.
- **Incertidumbre:** intervalos de confianza al 95 % por **bootstrap** sobre el test (2000 réplicas,
  resampleo de las 169 imágenes con reemplazo), no entre semillas.
- **Métrica principal:** **AUROC media de las 5 patologías oficiales de CheXpert** (Atelectasis,
  Cardiomegaly, Consolidation, Edema, Pleural Effusion).
- **Hardware:** NVIDIA RTX 4060 (8 GB), CUDA.

**Configuraciones de clases comparadas:**

| Config | Clases | Descripción |
|--------|:---:|-------------|
| `nofracture12` | 12 | Sin Fracture (0 positivos en el test silver). Config base de comparación. |
| `min5pct9` | 9 | Sin las clases de prevalencia <5 % (Enlarged Cardiomediastinum, Lung Lesion, Pneumonia, Fracture). |
| `full13` | 13 | Referencia completa. Solo se entrena con DenseNet-121 (ablación). |

---

## 2. Resultados — configuración `nofracture12` (12 clases)

AUROC, PR-AUC con intervalo de confianza bootstrap 95 % sobre el test silver (169 imágenes).

| Arquitectura | AUROC CheXpert-5 [IC 95%] | AUROC-macro [IC 95%] | PR-AUC-macro [IC 95%] | F1-macro | Épocas | Duración (min) |
|--------------|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXt-Tiny | **0.8536** [0.826, 0.880] | 0.8129 [0.793, 0.852] | 0.6181 [0.584, 0.722] | 0.4809 | 12 | 147 |
| ResNet-50 | 0.8517 [0.824, 0.879] | 0.7978 [0.768, 0.858] | 0.5992 [0.568, 0.705] | **0.5062** | 13 | 138 |
| DenseNet-121 | 0.8497 [0.821, 0.877] | 0.8229 [0.803, 0.865] | 0.6136 [0.583, 0.724] | 0.4968 | 13 | 131 |
| Swin-Tiny | 0.8438 [0.815, 0.871] | 0.7924 [0.773, 0.861] | 0.6162 [0.584, 0.725] | 0.5040 | 17 | 186 |
| VGG16-BN | 0.8403 [0.812, 0.868] | 0.7830 [0.760, 0.845] | 0.5864 [0.554, 0.689] | 0.4867 | 12 | 817 |

> VGG16-BN se entrenó a **batch 64** (comparable al resto), pero a un coste muy superior: **817 min
> (~13,6 h), ≈6× DenseNet-121**, por desbordamiento de VRAM a RAM en 8 GB (138 M parámetros). Es la
> arquitectura con **el peor punto de AUROC CheXpert-5 y el mayor coste**, aunque su IC sigue solapando
> con el de las demás (equivalencia estadística).

---

## 3. Resultados — configuración `min5pct9` (9 clases)

| Arquitectura | AUROC CheXpert-5 [IC 95%] | AUROC-macro [IC 95%] | PR-AUC-macro [IC 95%] | F1-macro | Épocas | Duración (min) |
|--------------|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXt-Tiny | **0.8505** [0.823, 0.877] | **0.8788** [0.860, 0.897] | **0.7331** [0.681, 0.798] | 0.6022 | 12 | 85 |
| ResNet-50 | 0.8422 [0.813, 0.869] | 0.8751 [0.853, 0.895] | 0.7091 [0.662, 0.783] | **0.6051** | 13 | 85 |
| Swin-Tiny | 0.8404 [0.811, 0.868] | 0.8673 [0.836, 0.893] | 0.7313 [0.675, 0.801] | 0.5941 | 18 | 235 |
| DenseNet-121 | 0.8256 [0.793, 0.855] | 0.8578 [0.829, 0.883] | 0.7038 [0.653, 0.775] | 0.5768 | 14 | 96 |

---

## 4. Ablación de configuración — DenseNet-121 (full13 / nofracture12 / min5pct9)

| Config | Clases | AUROC CheXpert-5 [IC 95%] | AUROC-macro [IC 95%] | PR-AUC-macro [IC 95%] | F1-macro |
|--------|:---:|:---:|:---:|:---:|:---:|
| full13 | 13 | 0.8451 [0.814, 0.873] | 0.7988 [0.773, 0.857] | 0.6035 [0.574, 0.705] | 0.4555 |
| nofracture12 | 12 | 0.8497 [0.821, 0.877] | 0.8229 [0.803, 0.865] | 0.6136 [0.583, 0.724] | 0.4968 |
| min5pct9 | 9 | 0.8256 [0.793, 0.855] | 0.8578 [0.829, 0.883] | 0.7038 [0.653, 0.775] | 0.5768 |

**Tendencia de la ablación (DenseNet-121):** al reducir el número de clases (13 → 12 → 9), las
métricas macro y el F1-macro crecen de forma monótona (AUROC-macro 0.799 → 0.823 → 0.858; F1-macro
0.456 → 0.497 → 0.577), confirmando que las clases raras y mal soportadas penalizan los promedios. La
AUROC CheXpert-5 se mantiene estable (0.845 / 0.850 / 0.826; IC solapados), porque las 5 patologías
oficiales están presentes en las tres configuraciones.

---

## 5. Análisis

### 5.1. Entre arquitecturas: equivalencia estadística

En la métrica principal (**AUROC CheXpert-5**), los intervalos de confianza **se solapan por completo**.
En `nofracture12` (las **5 arquitecturas**) el rango de puntos estimados va de 0.8403 (VGG16-BN) a 0.8536
(ConvNeXt-Tiny) —una diferencia de 0.0133— muy inferior a la anchura de los IC (~±0.027). Por tanto,
**ninguna arquitectura es significativamente superior** a las demás en las cinco patologías oficiales
sobre este test. ConvNeXt-Tiny obtiene el punto más alto y VGG16-BN el más bajo, pero no de forma
estadísticamente distinguible. (La comparación de las 5 en `min5pct9` se cerrará al terminar VGG16-BN.)

### 5.2. Entre configuraciones: efecto significativo en métricas macro

El cambio de configuración sí produce un efecto medible. Para ConvNeXt-Tiny, la AUROC-macro pasa de
0.8129 [0.793, 0.852] en `nofracture12` a 0.8788 [0.860, 0.897] en `min5pct9`: **los intervalos no se
solapan**. El mismo patrón se observa en la PR-AUC-macro (0.618 → 0.733). Reducir el conjunto a las 9
clases de prevalencia ≥5 % elimina las patologías raras y mal soportadas (cuyas métricas son ruidosas)
y mejora de forma estadísticamente sólida las medias macro. La AUROC CheXpert-5, en cambio, se mantiene
estable entre configuraciones (las 5 oficiales sobreviven en todas), confirmando que la mejora macro
proviene de excluir clases difíciles, no de un mejor aprendizaje de las 5 principales.

**Conclusión transversal:** el conjunto de clases evaluado influye más en las métricas agregadas que
la elección de arquitectura.

### 5.3. Coste computacional

DenseNet-121 es la arquitectura más eficiente (menos parámetros y menor duración por run, ~95–130 min)
con rendimiento estadísticamente equivalente al resto. Swin-Tiny es lenta (186–235 min) sin ventaja de
rendimiento. **VGG16-BN es la más cara con diferencia: ~817 min (~13,6 h), ≈6× DenseNet-121**, por
desbordamiento de VRAM a RAM en 8 GB (138 M parámetros) — y además el peor punto de AUROC. Es el caso
de manual de "más coste, ningún beneficio".

---

## 6. Recomendación

- **Arquitectura:** dado que el rendimiento es estadísticamente equivalente, se recomienda
  **DenseNet-121** por su mejor relación rendimiento/coste (menos parámetros, entrenamiento e
  inferencia más rápidos), coherente con su uso como referencia en radiografía de tórax. Si se
  prioriza el punto estimado más alto sin atender al coste, ConvNeXt-Tiny lo encabeza.
- **Configuración de clases:** `min5pct9` ofrece las mejores métricas macro y es preferible cuando el
  objetivo es el rendimiento agregado sobre clases bien soportadas; `nofracture12` conserva más
  patologías a costa de medias macro más bajas.

---

## 7. Limitaciones

- **Test pequeño (169 imágenes):** los intervalos de confianza son anchos; diferencias de ~0.01 en
  AUROC no son detectables. Clases como Fracture (0 positivos) o Lung Lesion (~1) no son evaluables.
- **Una sola semilla:** no se cuantifica la varianza por inicialización/orden de datos; la
  incertidumbre reportada es solo la del muestreo del test (bootstrap).
- **Sesgo de selección:** el test participa en el gate de promoción del mejor modelo, por lo que su
  métrica deja de ser estrictamente insesgada; se mitiga con el margen `promotion_min_delta` y el
  historial completo.
- **Etiquetas de entrenamiento automáticas** (CheXbert): contienen ruido frente a las de radiólogo del test.

---

## 8. Reproducibilidad

```bash
# Entrenar un run (ejemplo)
python -m src.main --class-config nofracture12 --model convnext_tiny

# Regenerar los intervalos de confianza del leaderboard
python -m src.bootstrap_ci --n-boot 2000 --seed 42
```

Resultados crudos por run en `experiments/<run_id>/` (métricas, predicciones, curvas, `report.md`);
índices agregados en `experiments/leaderboard.csv` y `experiments/leaderboard_ci.csv`.