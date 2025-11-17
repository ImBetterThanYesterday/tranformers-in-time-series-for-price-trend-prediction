# Guía de Inferencia - TLOB Model

> **Documentación completa de entrada de datos y scripts de inferencia**  
> Fecha: 14 Noviembre 2025

---

## 📋 Tabla de Contenidos

1. [Estructura de Datos de Entrada](#estructura-de-datos-de-entrada)
2. [Arquitectura del Modelo TLOB](#arquitectura-del-modelo-tlob)
3. [Scripts de Inferencia](#scripts-de-inferencia)
4. [Resultados de Ejemplo](#resultados-de-ejemplo)
5. [Análisis Detallado](#análisis-detallado)

---

## 1. Estructura de Datos de Entrada

### 📊 Archivos `.npy` del Dataset BTC

El dataset BTC se encuentra en `data/BTC/` con tres archivos principales:

```
data/BTC/
├── train.npy      (933.6 MB) - 2,780,963 timesteps × 44 features
├── val.npy        (115.6 MB) -   344,454 timesteps × 44 features
└── test.npy       (203.2 MB) -   605,453 timesteps × 44 features
```

### 🔍 Forma de los Datos

Cada archivo `.npy` contiene:

| Archivo | Shape | Interpretación |
|---------|-------|----------------|
| `train.npy` | `(2780963, 44)` | 2.7M snapshots del LOB × 44 features |
| `val.npy` | `(344454, 44)` | 344K snapshots del LOB × 44 features |
| `test.npy` | `(605453, 44)` | 605K snapshots del LOB × 44 features |

**Nota importante:** Los datos tienen **44 features** en total, pero el modelo TLOB fue entrenado con **solo 40 features** (las del Limit Order Book puro, sin metadata adicional).

### 📦 Composición de los 44 Features

```
Features 0-39:  LOB Features (Limit Order Book)
  ├─ 0-9:   ASK Prices (10 niveles)
  ├─ 10-19: ASK Volumes (10 niveles)
  ├─ 20-29: BID Prices (10 niveles)
  └─ 30-39: BID Volumes (10 niveles)

Features 40-43: Metadata adicional (4 features)
  └─ Información auxiliar (no usada en el entrenamiento)
```

### 🪟 Ventanas de Entrada al Modelo

TLOB procesa **secuencias temporales** de snapshots del LOB:

```python
seq_size = 128  # Longitud de la ventana temporal

# Cada entrada al modelo tiene forma:
input_shape = (batch_size, seq_size, num_features)
            = (batch_size, 128, 40)

# Ejemplo concreto:
# - 5 ejemplos de entrada
# - 128 timesteps consecutivos cada uno
# - 40 features LOB por timestep
X = np.array([...])  # Shape: (5, 128, 40)
```

### 📈 Ejemplo Visual de un Snapshot

```python
import numpy as np

# Un snapshot en el tiempo t
snapshot = train[t, :40]  # Solo primeras 40 features

# Estructura del snapshot:
ASK_prices  = snapshot[0:10]   # [p_ask_1, p_ask_2, ..., p_ask_10]
ASK_volumes = snapshot[10:20]  # [v_ask_1, v_ask_2, ..., v_ask_10]
BID_prices  = snapshot[20:30]  # [p_bid_1, p_bid_2, ..., p_bid_10]
BID_volumes = snapshot[30:40]  # [v_bid_1, v_bid_2, ..., v_bid_10]

# Ejemplo de valores (normalizados con Z-score):
print("ASK prices (5 primeros niveles):", snapshot[:5])
# Output: [-1.457, 0.465, -1.457, 2.147, -1.457]

print("BID volumes (5 primeros niveles):", snapshot[30:35])
# Output: [0.234, -0.567, 1.234, -0.890, 0.456]
```

### 🔢 Estadísticas de los Datos

```python
# Rango de valores (después de normalización)
Min: -1.4981
Max: inf (algunos outliers extremos)

# Distribución típica
Mean: ~-0.5 a -0.7
Std:  ~0.8 a 1.2

# Los datos están normalizados con Z-score durante el preprocesamiento
```

---

## 2. Arquitectura del Modelo TLOB

### 🏗️ Configuración del Modelo

```python
MODEL_CONFIG = {
    "hidden_dim": 40,        # Dimensión oculta del transformer
    "num_layers": 4,         # Número de capas transformer (doble rama)
    "seq_size": 128,         # Longitud de secuencia de entrada
    "num_features": 40,      # Features del LOB
    "num_heads": 1,          # Número de attention heads
    "is_sin_emb": True,      # Usar positional encoding sinusoidal
    "dataset_type": "BTC",   # Tipo de dataset
}

# Parámetros totales: 1,135,974 (~1.1M parámetros)
```

### 🧠 Arquitectura Dual-Branch Transformer

```
                         INPUT
                   (batch, 128, 40)
                          ↓
                    ┌─────────┐
                    │   BiN   │  ← Batch-Instance Normalization
                    │Normalize│
                    └────┬────┘
                         ↓
                    ┌─────────┐
                    │ Embed   │  ← Linear: 40 → 40 (hidden_dim)
                    │ Layer   │
                    └────┬────┘
                         ↓
                    ┌─────────┐
                    │   Add   │  ← Positional Encoding (sinusoidal)
                    │Pos Emb  │
                    └────┬────┘
                         ↓
            ┌────────────┴────────────┐
            ↓                         ↓
      ┌──────────┐            ┌──────────┐
      │ Branch 1 │            │ Branch 2 │
      │(Spatial) │            │(Temporal)│
      └────┬─────┘            └─────┬────┘
           │                        │
           │  4 Transformer Layers  │
           │  cada uno con:         │
           │  • Multi-Head Attn     │
           │  • Layer Norm          │
           │  • MLP (feedforward)   │
           │                        │
           └────────────┬───────────┘
                        ↓
                   ┌─────────┐
                   │Concatena│
                   │  te     │
                   └────┬────┘
                        ↓
                   ┌─────────┐
                   │  MLP    │  ← Final layers: Linear → ReLU → Linear
                   │ Final   │     (hidden*2 → hidden → 3)
                   └────┬────┘
                        ↓
                      OUTPUT
                   (batch, 3)
                 [DOWN, STAT, UP]
```

### 📊 Flujo de Datos en Detalle

#### Paso 1: Normalización (BiN)
```python
# Batch-Instance Normalization
# Normaliza tanto a nivel de batch como de instancia
x = BiN(x)  # (batch, 128, 40) → (batch, 128, 40)
```

#### Paso 2: Embedding
```python
# Proyección lineal a dimensión oculta
x = Linear(num_features=40, hidden_dim=40)(x)
# (batch, 128, 40) → (batch, 128, 40)
```

#### Paso 3: Positional Encoding
```python
# Añadir información posicional (sinusoidal)
x = x + pos_encoding  # (batch, 128, 40)
```

#### Paso 4: Dual Attention (Key Innovation)
```python
# BRANCH 1: Spatial Attention (entre features en cada timestep)
# Captura relaciones entre precios y volúmenes

# BRANCH 2: Temporal Attention (entre timesteps)
# Captura evolución temporal del mercado

# Cada branch tiene 4 capas transformer:
for layer in range(4):
    # Multi-Head Self-Attention
    q, k, v = compute_qkv(x)
    attn_out = MultiHeadAttention(q, k, v)
    x = LayerNorm(x + attn_out)
    
    # Feedforward MLP
    mlp_out = MLP(x)
    x = x + mlp_out
```

#### Paso 5: Agregación y Clasificación
```python
# Concatenar salidas de ambas ramas
x = concat([branch1_out, branch2_out])  # (batch, hidden*2)

# Capas finales
x = Linear(hidden*2, hidden)(x)
x = ReLU(x)
x = Linear(hidden, 3)(x)  # 3 clases: DOWN, STATIONARY, UP

# Output: logits (batch, 3)
```

### 🎯 Clases de Predicción

| Clase | Valor | Significado | Interpretación |
|-------|-------|-------------|----------------|
| **DOWN** | 0 | Precio bajará | Tendencia bajista en próximos k timesteps |
| **STATIONARY** | 1 | Precio estable | Sin cambio significativo |
| **UP** | 2 | Precio subirá | Tendencia alcista en próximos k timesteps |

**Horizonte de predicción:** `horizon = 10` timesteps hacia el futuro.

---

## 3. Scripts de Inferencia

### 📝 Archivos Creados

```
TLOB-main/
├── inference_pytorch.py      # Inferencia con PyTorch
├── inference_onnx.py          # Inferencia con ONNX Runtime (más rápido)
├── inspect_data.py            # Inspección y visualización de datos
└── data/BTC/
    └── inference_examples.npy # 5 ejemplos extraídos para inferencia
```

### 🚀 Script 1: `inference_pytorch.py`

**Propósito:** Cargar el modelo entrenado (`.pt`) y hacer predicciones con PyTorch.

**Uso:**
```bash
cd /path/to/TLOB-main
python3 inference_pytorch.py
```

**Flujo del script:**

1. **Cargar modelo TLOB**
   ```python
   from models.tlob import TLOB
   
   model = TLOB(**MODEL_CONFIG)
   checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
   
   # Remover prefijo "model." del state_dict
   state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items() 
                 if k.startswith("model.")}
   model.load_state_dict(state_dict)
   model.eval()
   ```

2. **Cargar datos de entrada**
   ```python
   examples = np.load("data/BTC/inference_examples.npy")
   examples = examples[:, :, :40]  # Solo 40 features LOB
   X = torch.from_numpy(examples).float()
   ```

3. **Realizar inferencia**
   ```python
   with torch.no_grad():
       logits = model(X)
       probs = F.softmax(logits, dim=1)
       preds = torch.argmax(probs, dim=1)
   ```

4. **Guardar resultados**
   ```python
   np.save("inference_results/predictions_pytorch.npy", preds.numpy())
   np.save("inference_results/probabilities_pytorch.npy", probs.numpy())
   ```

**Salida esperada:**
```
================================================================================
INFERENCIA CON PYTORCH - MODELO TLOB
================================================================================

📌 Device: cpu
📌 Checkpoint: data/checkpoints/TLOB/.../val_loss=0.623_epoch=2.pt

================================================================================
1. CARGANDO MODELO
================================================================================
✓ Modelo cargado exitosamente
  → Parámetros totales: 1,135,974
  → Parámetros entrenables: 1,135,974

================================================================================
3. REALIZANDO INFERENCIA
================================================================================
✓ Logits obtenidos: torch.Size([5, 3])

--- Ejemplo 1 ---
  Probabilidades:
    • DOWN:       6.91%
    • STATIONARY: 92.30%
    • UP:         0.79%
  Predicción: STATIONARY (clase 1)
  Confianza: 92.30%
```

---

### ⚡ Script 2: `inference_onnx.py`

**Propósito:** Inferencia optimizada con ONNX Runtime (más rápido para producción).

**Ventajas de ONNX:**
- ✅ **Más rápido:** ~3ms por batch de 5 ejemplos
- ✅ **Portátil:** No requiere PyTorch en producción
- ✅ **Optimizado:** Graph optimizations automáticas
- ✅ **Multi-plataforma:** CPU, GPU, Edge devices

**Uso:**
```bash
cd /path/to/TLOB-main
python3 inference_onnx.py
```

**Benchmark de velocidad:**
```
✓ Benchmark completado (100 iteraciones):
  → Tiempo promedio: 2.94 ± 0.14 ms
  → Throughput: 1,699.7 ejemplos/segundo
  → Latencia por ejemplo: 0.59 ms
```

**Flujo del script:**

1. **Cargar sesión ONNX**
   ```python
   import onnxruntime as ort
   
   session = ort.InferenceSession(
       ONNX_PATH,
       providers=['CPUExecutionProvider']
   )
   ```

2. **Inspeccionar input/output**
   ```python
   input_info = session.get_inputs()[0]
   # Nombre: input
   # Shape: ['batch_size', 128, 40]
   # Type: tensor(float)
   
   output_info = session.get_outputs()[0]
   # Nombre: output
   # Shape: ['batch_size', 3]
   # Type: tensor(float)
   ```

3. **Ejecutar inferencia**
   ```python
   onnx_input = {"input": examples}
   outputs = session.run(None, onnx_input)
   logits = outputs[0]
   ```

---

### 🔍 Script 3: `inspect_data.py`

**Propósito:** Explorar y visualizar la estructura de los datos BTC.

**Uso:**
```bash
cd /path/to/TLOB-main
python3 inspect_data.py
```

**Funcionalidades:**

1. **Estadísticas generales**
   - Shape, dtype, tamaño en MB
   - Min/Max, Mean/Std
   - Conteo de NaN e Inf

2. **Análisis por feature**
   - Estadísticas de cada columna
   - Distribución de valores

3. **Visualizaciones generadas:**
   ```
   inspection_results/
   ├── feature_distributions.png    # Histogramas de 10 features
   ├── temporal_evolution.png       # Series temporales
   └── window_heatmap.png           # Heatmap de ventana de entrada
   ```

---

## 4. Resultados de Ejemplo

### 🎯 Predicciones Reales (5 Ejemplos)

Ejecutamos inferencia sobre 5 ventanas diferentes del dataset BTC:

#### Ejemplo 1
```python
Entrada:
  Índices: 0 → 127 (primeros 128 timesteps)
  Mean: -0.6705, Std: 1.0642
  
Predicción:
  Logits: [-0.1626, 2.4288, -2.3314]
  Probabilidades:
    • DOWN:       6.91%
    • STATIONARY: 92.30% ← PREDICCIÓN
    • UP:         0.79%
  Confianza: 92.30%
```

#### Ejemplo 2
```python
Entrada:
  Índices: 500 → 627
  Mean: -0.7152, Std: 0.8461
  
Predicción:
  Logits: [-1.8530, 3.4993, -1.6616]
  Probabilidades:
    • DOWN:       0.47%
    • STATIONARY: 98.96% ← PREDICCIÓN
    • UP:         0.57%
  Confianza: 98.96% (muy alta!)
```

#### Ejemplo 3
```python
Entrada:
  Índices: 1000 → 1127
  Mean: -0.7241, Std: 0.8501
  
Predicción:
  Logits: [-1.0201, 3.6611, -2.6291]
  Probabilidades:
    • DOWN:       0.92%
    • STATIONARY: 98.90% ← PREDICCIÓN
    • UP:         0.18%
  Confianza: 98.90%
```

#### Ejemplo 4
```python
Entrada:
  Índices: 1500 → 1627
  Mean: -0.7175, Std: 0.8984
  
Predicción:
  Logits: [-2.5591, 3.0360, -0.4507]
  Probabilidades:
    • DOWN:       0.36%
    • STATIONARY: 96.68% ← PREDICCIÓN
    • UP:         2.96%
  Confianza: 96.68%
```

#### Ejemplo 5
```python
Entrada:
  Índices: 2000 → 2127
  Mean: -0.6570, Std: 1.1354
  
Predicción:
  Logits: [-2.4749, 3.6659, -1.1501]
  Probabilidades:
    • DOWN:       0.21%
    • STATIONARY: 98.99% ← PREDICCIÓN
    • UP:         0.80%
  Confianza: 98.99% (confianza máxima!)
```

### 📊 Resumen de Resultados

| Ejemplo | Predicción | Confianza | Logit DOWN | Logit STAT | Logit UP |
|---------|------------|-----------|------------|------------|----------|
| 1 | STATIONARY | 92.30% | -0.163 | **2.429** | -2.331 |
| 2 | STATIONARY | 98.96% | -1.853 | **3.499** | -1.662 |
| 3 | STATIONARY | 98.90% | -1.020 | **3.661** | -2.629 |
| 4 | STATIONARY | 96.68% | -2.559 | **3.036** | -0.451 |
| 5 | STATIONARY | 98.99% | -2.475 | **3.666** | -1.150 |

**Observaciones:**
- ✅ Todos los ejemplos predicen **STATIONARY** (precio estable)
- ✅ Confianza muy alta en todos los casos (>92%)
- ✅ Los logits para la clase STATIONARY dominan claramente (+2.4 a +3.7)
- ⚠️ Clase UP tiene logits muy negativos en todos los casos
- ⚠️ Clase DOWN también suprimida (logits negativos)

**Interpretación:** El modelo está muy seguro de que el precio de Bitcoin se mantendrá estable en los próximos 10 timesteps para estos ejemplos específicos.

---

## 5. Análisis Detallado

### 🔬 Formato de Entrada en Producción

Para usar el modelo en un entorno real:

```python
import numpy as np
import torch
from models.tlob import TLOB

# 1. Preparar los datos
def prepare_lob_data(lob_snapshot_history):
    """
    Args:
        lob_snapshot_history: Lista de diccionarios con snapshots LOB
        Ejemplo de un snapshot:
        {
            'ask_prices': [100.5, 100.6, ...],   # 10 niveles
            'ask_volumes': [5, 3, ...],          # 10 niveles
            'bid_prices': [100.4, 100.3, ...],   # 10 niveles
            'bid_volumes': [7, 4, ...]           # 10 niveles
        }
    
    Returns:
        np.ndarray con shape (1, 128, 40) listo para inferencia
    """
    # Asegurar que tenemos 128 snapshots consecutivos
    assert len(lob_snapshot_history) == 128
    
    # Construir matriz de features
    features = []
    for snapshot in lob_snapshot_history:
        row = (
            snapshot['ask_prices'] +     # 0-9
            snapshot['ask_volumes'] +    # 10-19
            snapshot['bid_prices'] +     # 20-29
            snapshot['bid_volumes']      # 30-39
        )
        features.append(row)
    
    X = np.array(features).astype(np.float32)
    
    # Normalizar (importante!)
    # En producción, guardar media/std del train set
    X = (X - train_mean) / train_std
    
    # Añadir dimensión de batch
    X = np.expand_dims(X, axis=0)  # (128, 40) → (1, 128, 40)
    
    return X


# 2. Cargar modelo
model = TLOB(**MODEL_CONFIG)
checkpoint = torch.load(checkpoint_path, weights_only=False)
state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items() 
              if k.startswith("model.")}
model.load_state_dict(state_dict)
model.eval()


# 3. Hacer predicción
with torch.no_grad():
    X_tensor = torch.from_numpy(X)
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()

print(f"Predicción: {['DOWN', 'STATIONARY', 'UP'][pred_class]}")
print(f"Confianza: {confidence:.2%}")
```

---

### 🎛️ Ajustes y Configuración

#### Cambiar horizonte de predicción

```python
# El horizonte está definido en el checkpoint usado
# Para cambiar el horizonte, usar un checkpoint diferente:

CHECKPOINT_HORIZON_10  = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/..."
CHECKPOINT_HORIZON_20  = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_20_seed_42/..."
CHECKPOINT_HORIZON_50  = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_50_seed_42/..."
CHECKPOINT_HORIZON_100 = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_100_seed_42/..."
```

#### Usar GPU en lugar de CPU

```python
# En inference_pytorch.py, línea 22:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# En inference_onnx.py, añadir provider de GPU:
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(ONNX_PATH, providers=providers)
```

#### Batch de múltiples predicciones

```python
# Cargar más ejemplos
examples = []
for i in range(100):  # 100 ventanas
    start = i * 128
    window = train_data[start:start+128, :40]
    examples.append(window)

X = np.array(examples)  # Shape: (100, 128, 40)

# Inferencia en batch (más eficiente)
with torch.no_grad():
    logits = model(torch.from_numpy(X))
    # Output: (100, 3) - predicciones para las 100 ventanas
```

---

### 📐 Métricas del Modelo

```python
# Del checkpoint entrenado
Validation Loss: 0.623  (mejor época)
Epoch: 2

# Del paper (F1-score en BTC):
TLOB BTC F1-score: ~67.8 (promedio sobre 4 horizontes)
Mejora vs SoTA: +1.1 F1-score

# Velocidad de inferencia (ONNX):
Latencia: 0.59 ms por ejemplo
Throughput: 1,699 ejemplos/segundo
```

---

### ⚠️ Limitaciones y Consideraciones

1. **Normalización crítica:**
   - Los datos **deben** estar normalizados con la misma media/std del train set
   - Sin normalización, las predicciones serán erróneas

2. **Ventana completa requerida:**
   - Se necesitan exactamente 128 timesteps consecutivos
   - No se puede hacer predicción con menos timesteps

3. **Features exactas:**
   - Debe haber exactamente 40 features LOB en el orden correcto
   - El orden importa: ASK prices, ASK vols, BID prices, BID vols

4. **Horizonte fijo:**
   - Cada checkpoint predice a un horizonte específico (10, 20, 50 o 100)
   - No se puede cambiar el horizonte sin re-entrenar

5. **Clase STATIONARY dominante:**
   - En los ejemplos, STATIONARY siempre gana
   - Puede indicar desbalance de clases en el train set
   - En mercados reales, considerar umbrales de confianza

---

### 🔗 Integración con Trading Systems

Ejemplo de cómo integrar en un sistema de trading:

```python
class TLOBPredictor:
    def __init__(self, checkpoint_path, device='cpu'):
        self.model = TLOB(**MODEL_CONFIG)
        self._load_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
        # Buffer para mantener últimos 128 snapshots
        self.snapshot_buffer = deque(maxlen=128)
        
    def add_snapshot(self, lob_snapshot):
        """Añade un nuevo snapshot al buffer"""
        features = self._extract_features(lob_snapshot)
        self.snapshot_buffer.append(features)
        
    def predict(self):
        """Predice tendencia basado en últimos 128 snapshots"""
        if len(self.snapshot_buffer) < 128:
            raise ValueError("Se necesitan al menos 128 snapshots")
        
        # Preparar entrada
        X = np.array(list(self.snapshot_buffer))
        X = self._normalize(X)
        X = np.expand_dims(X, 0)  # Añadir batch dim
        
        # Inferencia
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return {
            'class': ['DOWN', 'STATIONARY', 'UP'][probs.argmax().item()],
            'probabilities': {
                'DOWN': probs[0, 0].item(),
                'STATIONARY': probs[0, 1].item(),
                'UP': probs[0, 2].item()
            },
            'confidence': probs.max().item()
        }

# Uso en loop de trading
predictor = TLOBPredictor('path/to/checkpoint.pt')

while True:
    # Obtener snapshot actual del exchange
    snapshot = get_current_lob_snapshot()
    
    # Actualizar predictor
    predictor.add_snapshot(snapshot)
    
    # Hacer predicción cada N segundos
    if should_predict():
        prediction = predictor.predict()
        
        # Ejecutar estrategia basada en predicción
        if prediction['confidence'] > 0.90:
            if prediction['class'] == 'UP':
                place_buy_order()
            elif prediction['class'] == 'DOWN':
                place_sell_order()
    
    time.sleep(1)  # Esperar próximo snapshot
```

---

### 📚 Referencias

- **Paper:** "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data"
- **Autores:** Leonardo Berti (Sapienza), Gjergji Kasneci (TUM)
- **Dataset BTC:** Kaggle Bitcoin LOB dataset (2023-01-09 to 2023-01-20)
- **Repositorio:** TLOB-main/

---

### 📞 Archivos de Soporte

```
TLOB-main/
├── docs/
│   ├── knowledge.md              # Documentación general del proyecto
│   └── inference_guide.md        # Este documento
├── inference_pytorch.py          # Script de inferencia PyTorch
├── inference_onnx.py             # Script de inferencia ONNX
├── inspect_data.py               # Exploración de datos
├── data/BTC/
│   ├── train.npy
│   ├── val.npy
│   ├── test.npy
│   └── inference_examples.npy    # 5 ejemplos extraídos
├── inference_results/            # Resultados guardados
│   ├── predictions_pytorch.npy
│   ├── predictions_onnx.npy
│   ├── probabilities_pytorch.npy
│   └── probabilities_onnx.npy
└── inspection_results/           # Visualizaciones
    ├── feature_distributions.png
    ├── temporal_evolution.png
    └── window_heatmap.png
```

---

**Última actualización:** 14 Noviembre 2025  
**Autor:** Documentación generada para el proyecto TLOB

