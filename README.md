# SignBridge Model — ASL Sign Language Recognition

Implementación del modelo de aprendizaje profundo utilizado en SignBridge 
para el reconocimiento de señas aisladas del Lenguaje de Señas Americano (ASL).
El modelo recibe secuencias de keypoints extraídos con MediaPipe Holistic y 
las clasifica mediante una arquitectura Transformer Encoder.

---

## Estructura del repositorio
```
ASL_Model/
├── data/
│   ├── dataset/
│   ├── features/
├── outputs/
│   ├── metrics_report/
│   ├── config.yaml            # Configuración de entrenamiento e inferencia
├── transformer/
│   ├── core/
│   │   ├── config.py          # Configuración global (dataclasses)
│   │   └── exceptions.py      # Excepciones
│   ├── data_/
│   │   ├── dataset.py         # SignLanguageDataset + collate_fn
│   │   ├── augmentation.py    # Aumentación temporal y de pose
│   │   ├── feature_extractors.py  # Extracción de keypoints con MediaPipe
│   │   ├── sampler.py         # Muestreo balanceado por clase
│   │   └── preparation.py     # Preparación y partición del dataset
│   ├── evaluation/
│   │   ├── evaluator.py       # validacion con split de prueba
│   │   └── metrics_report.py          # Generación de gráficas y reporte de evaluación
│   ├── model/
│   │   ├── transformer.py     # SignLanguageTransformer (modelo principal)
│   │   └── components.py      # FeatureProjection, PositionalEncoding,
│   │                          # CrossModalAttention, AttentionPooling, etc.
│   ├── training/
│   │   ├── trainer.py         # Bucle de entrenamiento + callbacks
│   │   ├── metrics.py         # MetricsTracker, TrainingHistory
│   │   └── callbacks.py       # EarlyStopping, ModelCheckpoint, LRLogger
│   
└── main.py              # Script principal de entrenamiento
```
---

## Descripción del modelo

El reconocimiento de señas se modela como un problema de **clasificación de 
secuencias**, donde cada muestra es una seña aislada representada como una 
serie temporal de keypoints corporales.

### Arquitectura: Transformer Encoder
```
Entrada: pose_features (B, T, D) + attention_mask (B, T)
┌─────────────────────────────┐
│   FeatureProjection         │  858 → 512 (Linear + LayerNorm + GELU)
├─────────────────────────────┤
│   Positional Encoding       │  Aprendible, max_len=128
├─────────────────────────────┤
│   Transformer Encoder x4    │  8 cabezas · ff_dim=2048 · pre-norm
├─────────────────────────────┤
│   Attention Pooling         │  T pasos → 1 vector de 512 dim
├─────────────────────────────┤
│   Classification Head       │  512 → 1024 → 2286 logits
└─────────────────────────────┘
Salida: logits (B, 2286)
```

| Parámetro | Valor |
|---|---|
| Dimensión interna (d_model) | 512 |
| Capas del encoder | 4 |
| Cabezas de atención | 8 |
| Dimensión feed-forward | 2048 |
| Tipo de pooling | Attention pooling |
| Clases de salida | 2.286 |
| Longitud máxima de secuencia | 64 fotogramas |

---

## Pipeline de datos

### 1. Extracción de características
Las características de pose son extraídas con **MediaPipe Holistic** 
(manos, rostro y cuerpo) y almacenadas en archivos **HDF5 con compresión gzip** 
para acceso eficiente durante el entrenamiento.

Cada fotograma produce un vector base de **429 dimensiones**:
- Mano izquierda: 21 puntos × 3 coordenadas = 63
- Mano derecha: 21 puntos × 3 coordenadas = 63
- Rostro (subconjunto): 68 puntos × 3 = 204
- Cuerpo (pose): 33 puntos × 3 = 99

### 2. Feature engineering — velocidad temporal
Se calcula la velocidad como la diferencia entre fotogramas consecutivos 
(primera derivada temporal), duplicando la dimensionalidad:
velocity[t] = features[t] - features[t-1]
features_final = concat(features, velocity)  →  858 dim/frame

Esto permite al modelo capturar no solo la postura estática, 
sino también la dinámica del movimiento de la seña.

### 3. Normalización de secuencias
Todas las secuencias se estandarizan a `max_seq_length = 64` fotogramas:
- **Truncado**: secuencias más largas se recortan desde el final
- **Padding**: secuencias más cortas se rellenan con vectores de ceros

### 4. Máscara de atención
Se genera un tensor booleano de 64 posiciones que indica al Transformer 
cuáles fotogramas son reales y cuáles son relleno:
- `True` → fotograma válido
- `False` → padding (ignorado en los cálculos de auto-atención)

### 5. Construcción de batches
Una función `collate_fn` personalizada agrupa las muestras en batches 
homogéneos, apilando tensores de pose, máscaras y etiquetas en un 
objeto `BatchData` listo para el modelo.

---

## Entrenamiento

| Parámetro | Valor |
|---|---|
| Optimizador | AdamW (β₁=0.9, β₂=0.999) |
| Tasa de aprendizaje | 5×10⁻⁴ (cosine warmup, 8 épocas) |
| Batch size | 128 |
| Épocas máximas | 150 |
| Label smoothing | 0.1 |
| Precisión mixta (AMP) | float16 |
| Early stopping | paciencia=15, delta=0.001 |

El entrenamiento utiliza **pesos de clase** con frecuencia inversa 
suavizada (`power=0.5`) para compensar el desbalance entre las 2.286 clases.

---

## Resultados

| Partición | Top-1 | Top-5 | Top-10 |
|---|---|---|---|
| Entrenamiento | 99.80% | 99.94% | 99.96% |
| Validación | 75.76% | 90.54% | 93.11% |
| Prueba | 70.21% | — | — |

El modelo final (`best_model.pt`) corresponde a la época 144, 
punto de mayor exactitud de validación registrado por `ModelCheckpoint`.

---

## Script de inicio del entrenamiento

```python
py main.py
```
