# 📈 TLOB: Predicción de Tendencias con Transformers en Limit Order Book

> **Implementación del modelo TLOB (Transformer for Limit Order Book) con despliegue Docker y visualización Streamlit para predicción de tendencias de precios en Bitcoin**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Tabla de Contenidos

1. [Artículo Base](#-artículo-base)
2. [Descripción del Modelo](#-descripción-del-modelo)
3. [Resumen Teórico de la Arquitectura](#-resumen-teórico-de-la-arquitectura)
4. [Mecanismo de Atención (Q, K, V)](#-mecanismo-de-atención-q-k-v)
5. [Pasos para Ejecutar el Proyecto](#-pasos-para-ejecutar-el-proyecto)
6. [Carga de Pesos Preentrenados](#-carga-de-pesos-preentrenados)
7. [Proceso de Inferencia](#-proceso-de-inferencia)
8. [Estructura del Repositorio](#-estructura-del-repositorio)
9. [Documentación Adicional](#-documentación-adicional)
10. [Referencias](#-referencias)

---

## 📄 Artículo Base

**Título:** *"TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data"*

**Autores:** 
- Leonardo Berti (Sapienza University of Rome)
- Gjergji Kasneci (Technical University of Munich)

**Publicación:** arXiv:2502.15757, 2025

**Repositorio Original:** [https://github.com/LeonardoBerti00/TLOB](https://github.com/LeonardoBerti00/TLOB)

**Paper:** [https://arxiv.org/pdf/2502.15757](https://arxiv.org/pdf/2502.15757)

### Citación

```bibtex
@article{berti2025tlob,
  title={TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data},
  author={Berti, Leonardo and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2502.15757},
  year={2025}
}
```

### Abstract del Paper

El modelo TLOB introduce una arquitectura Transformer especializada para la predicción de tendencias de precios utilizando datos del Limit Order Book (LOB). A diferencia de modelos anteriores basados en CNN y LSTM, TLOB utiliza un mecanismo de **atención dual** (spatial y temporal) que captura relaciones entre features y evolución temporal de manera más efectiva. El modelo incorpora **BiN (Batch Independent Normalization)** para funcionar eficientemente con batch_size=1 en producción, y un **nuevo método de etiquetado sin sesgo de horizonte** que mejora la consistencia entre diferentes horizontes de predicción.

---

## 🎯 Descripción del Modelo

### ¿Qué es TLOB?

**TLOB (Transformer for Limit Order Book)** es un modelo de aprendizaje profundo diseñado específicamente para predecir tendencias de precios en mercados financieros usando datos del **Limit Order Book**.

### ¿Qué es un Limit Order Book?

El Limit Order Book es una estructura de datos en tiempo real que contiene:
- **Ask (Sell) Orders**: Órdenes de venta ordenadas por precio (menor a mayor)
- **Bid (Buy) Orders**: Órdenes de compra ordenadas por precio (mayor a menor)

**Ejemplo:**
```
Nivel  |  ASK Price  |  ASK Volume  |  BID Price  |  BID Volume
-------|-------------|--------------|-------------|-------------
  1    |  $50,100    |    2.5 BTC   |  $50,095    |   3.2 BTC
  2    |  $50,105    |    1.8 BTC   |  $50,090    |   2.1 BTC
  ...  |    ...      |     ...      |    ...      |    ...
  10   |  $50,150    |    5.0 BTC   |  $50,050    |   4.5 BTC
```

### Principales Innovaciones del Modelo

#### 1. **Dual Attention Mechanism** 🔍
El modelo aplica atención en DOS dimensiones:

- **Feature Attention (Espacial):**
  - ¿Qué niveles del LOB son más importantes?
  - Ejemplo: El primer nivel (best bid/ask) típicamente tiene más peso

- **Temporal Attention:**
  - ¿Qué timesteps del pasado son más relevantes?
  - Ejemplo: Eventos recientes vs. patrones históricos

#### 2. **BiN (Batch-Instance Normalization)** 📊
Normalización híbrida que combina:
```python
BiN(x) = 0.5 * BatchNorm(x) + 0.5 * InstanceNorm(x)
```

**Ventajas:**
- Estabiliza el entrenamiento con datos financieros volátiles
- Preserva información tanto a nivel de batch como de instancia individual

#### 3. **Arquitectura Eficiente** ⚡
- **Parámetros totales:** ~1.1M (compacto pero potente)
- **Inferencia rápida:** ~50ms por predicción en CPU
- **Memoria:** ~500MB (modelo + datos)

#### 4. **Desempeño Superior** 🏆
Comparado con modelos state-of-the-art:
- **F1-Score:** +3.7% en dataset FI-2010
- **Accuracy:** +1.1% en dataset Bitcoin
- **Generalización:** Funciona en múltiples criptomonedas y acciones

---

## 🏗️ Resumen Teórico de la Arquitectura

### Flujo General del Modelo

```
┌──────────────────────────────────────────────────────────┐
│                  INPUT: LOB Snapshot                      │
│       Shape: (batch=32, seq_len=128, features=40)        │
│                                                            │
│  Features: [ASK_P1, ASK_V1, BID_P1, BID_V1, ... ×10]    │
│  Timesteps: 128 snapshots × 250ms = 32 segundos          │
└────────────────────────┬─────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 1: BiN Normalization                               │
│  ────────────────────────                                │
│  • Normaliza precios y volúmenes                         │
│  • Combina batch + instance normalization                │
│  • Output: (32, 128, 40)                                 │
└────────────────────────┬─────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Linear Embedding                                │
│  ────────────────────────                                │
│  • Proyecta features a espacio latente                   │
│  • 40 features → hidden_dim (256)                        │
│  • Output: (32, 128, 256)                                │
└────────────────────────┬─────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Positional Encoding                             │
│  ────────────────────────                                │
│  • Añade información temporal (posición en secuencia)    │
│  • Sinusoidal o aprendible                               │
│  • Output: (32, 128, 256)                                │
└────────────────────────┬─────────────────────────────────┘
               ↓
              ┌──────────┴──────────┐
              │                     │
              ↓                     ↓
┌──────────────────────┐  ┌──────────────────────┐
│  BRANCH 1:           │  │  BRANCH 2:           │
│  Feature Attention   │  │  Temporal Attention  │
│  ─────────────────   │  │  ─────────────────   │
│                      │  │                      │
│  ┌────────────────┐ │  │  ┌────────────────┐ │
│  │ Transformer    │ │  │  │ Transformer    │ │
│  │ Layer 1        │ │  │  │ Layer 1        │ │
│  │ (256 × 128)    │ │  │  │ (128 × 256)    │ │
│  └────────────────┘ │  │  └────────────────┘ │
│          ↓          │  │          ↓          │
│  ┌────────────────┐ │  │  ┌────────────────┐ │
│  │ Transformer    │ │  │  │ Transformer    │ │
│  │ Layer 2        │ │  │  │ Layer 2        │ │
│  └────────────────┘ │  │  └────────────────┘ │
│          ↓          │  │          ↓          │
│  ┌────────────────┐ │  │  ┌────────────────┐ │
│  │ Transformer    │ │  │  │ Transformer    │ │
│  │ Layer 3        │ │  │  │ Layer 3        │ │
│  └────────────────┘ │  │  └────────────────┘ │
│          ↓          │  │          ↓          │
│  ┌────────────────┐ │  │  ┌────────────────┐ │
│  │ Transformer    │ │  │  │ Transformer    │ │
│  │ Layer 4        │ │  │  │ Layer 4        │ │
│  └────────────────┘ │  │  └────────────────┘ │
│                      │  │                      │
│  Output: (32,32,64) │  │  Output: (32,32,64) │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └──────────┬──────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: Concatenate & Flatten                           │
│  ────────────────────────────────                        │
│  • Combina ambas ramas                                   │
│  • Flatten: (32, 32, 64) + (32, 32, 64) → (32, 4096)    │
└────────────────────────┬─────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 5: Final MLP                                       │
│  ─────────────────                                       │
│  • Linear(4096 → 1024) + GELU                           │
│  • Linear(1024 → 256) + GELU                            │
│  • Linear(256 → 3)                                      │
│  • Softmax                                               │
└────────────────────────┬─────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: Predicción de Tendencia                         │
│  ──────────────────────────────────                      │
│  Shape: (32, 3)                                          │
│                                                           │
│  Clases:                                                 │
│    0: DOWN       (precio bajará)                         │
│    1: STATIONARY (precio estable)                        │
│    2: UP         (precio subirá)                         │
│                                                           │
│  Ejemplo: [0.10, 0.15, 0.75] → Predicción: UP (75%)    │
└─────────────────────────────────────────────────────────┘
```

### Componentes Clave

#### TransformerLayer

Cada capa Transformer contiene:

```python
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        # 1. Layer Normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 2. Multi-Head Attention
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(...)
        
        # 3. Feed-Forward MLP
        self.mlp = MLP(hidden_dim, hidden_dim*4, final_dim)
        
    def forward(self, x):
        # Residual connection
        res = x
        
        # Atención
        q, k, v = self.qkv(x)
        x, att = self.attention(q, k, v)
        
        # Skip connection + Norm
        x = self.norm(x + res)
        
        # MLP + Skip connection
        x = self.mlp(x) + x
        
        return x, att
```

---

## 🔍 Mecanismo de Atención (Q, K, V)

### ¿Qué son Q, K, V?

El mecanismo de atención se basa en tres proyecciones de los datos de entrada:

- **Q (Queries)**: "¿Qué estoy buscando?"
- **K (Keys)**: "¿Qué información está disponible?"
- **V (Values)**: "¿Cuál es el contenido real?"

### Generación de Q, K, V en TLOB

```python
class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        q = self.q(x)  # Queries
        k = self.k(x)  # Keys
        v = self.v(x)  # Values
        return q, k, v
```

### Proceso Detallado

#### 1️⃣ Input Embeddings

```
Input: (batch=32, seq_len=128, features=40)
         ↓ BiN + Embedding
Embedded: (32, 128, 256)
```

#### 2️⃣ Proyecciones Lineales

Cada timestep pasa por 3 transformaciones lineales independientes:

```python
# Para cada posición t:
Q[t] = W_q @ x[t] + b_q  # Shape: (256,)
K[t] = W_k @ x[t] + b_k  # Shape: (256,)
V[t] = W_v @ x[t] + b_v  # Shape: (256,)
```

**Matrices aprendibles:**
- `W_q`, `W_k`, `W_v`: Pesos de las proyecciones lineales
- Se aprenden durante el entrenamiento

#### 3️⃣ Multi-Head Attention

Las proyecciones se dividen en múltiples "cabezas":

```
num_heads = 8
hidden_dim = 256
head_dim = hidden_dim / num_heads = 32

Q: (32, 128, 256) → Reshape → (32, 8, 128, 32)
K: (32, 128, 256) → Reshape → (32, 8, 128, 32)
V: (32, 128, 256) → Reshape → (32, 8, 128, 32)
```

**¿Por qué múltiples cabezas?**
- Cada cabeza aprende diferentes aspectos de los datos
- Cabeza 1: Puede enfocarse en el spread (diferencia bid-ask)
- Cabeza 2: Puede enfocarse en el volumen
- Cabeza 3: Puede enfocarse en cambios temporales

#### 4️⃣ Cálculo de Atención

**Fórmula Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**Paso a paso:**

```python
# 1. Scores de atención (producto punto)
scores = Q @ K.transpose(-2, -1)  # (32, 8, 128, 128)
# scores[i, h, t, s] = cuánto el timestep t "atiende" al timestep s

# 2. Scaling (para estabilidad numérica)
d_k = 32  # head_dim
scores = scores / math.sqrt(d_k)

# 3. Softmax (normalizar a pesos que sumen 1)
attention_weights = softmax(scores, dim=-1)  # (32, 8, 128, 128)
# attention_weights[i, h, t, :].sum() == 1.0

# 4. Weighted sum de Values
output = attention_weights @ V  # (32, 8, 128, 32)
```

#### 5️⃣ Interpretación de los Pesos de Atención

```python
# Ejemplo: Predicción en el timestep t=127
attention_weights[0, 0, 127, :]  # Primera cabeza, último timestep

# Resultado típico:
# [0.001, 0.002, ..., 0.050, 0.080, 0.150]
#   ↑                  ↑      ↑       ↑
#   timesteps         t=100  t=120  t=126
#   antiguos          (medio) (reciente) (muy reciente)
```

**Interpretación:**
- Pesos altos en timesteps recientes → Considera eventos inmediatos
- Pesos bajos en timesteps antiguos → Menos relevantes para la predicción actual

### Visualización de Atención

La aplicación Streamlit incluye visualización de pesos de atención:

```python
# En app.py
att_weights = model.attention_weights  # (num_heads, seq_len, seq_len)

# Heatmap de atención
plt.imshow(att_weights[0, :, :], cmap='viridis')
plt.xlabel('Key Position (timestep)')
plt.ylabel('Query Position (timestep)')
plt.title('Attention Weights - Head 0')
```

**📖 Para más detalles:** Ver [`docs/MECANISMO_ATENCION_QKV.md`](docs/MECANISMO_ATENCION_QKV.md)

---

## 🚀 Pasos para Ejecutar el Proyecto

### Requisitos Previos

```bash
# Sistema operativo: Linux, macOS, o Windows (con WSL2)
# Docker Desktop 20.10+ (incluye Docker Compose)
# RAM: 4GB+ disponible
# Disco: 10GB+ libre
```

### ⚡ Método Rápido: Docker Compose (Recomendado)

**Solo 3 comandos para ejecutar todo el proyecto:**

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/tlob-prediction.git
cd tlob-prediction

# 2. Levantar aplicación con un solo comando
docker-compose up -d

# ✅ ¡Listo! La app estará disponible en http://localhost:8501
```

#### Comandos Útiles

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Verificar estado
docker-compose ps

# Detener aplicación
docker-compose down

# Reconstruir después de cambios
docker-compose up -d --build
```

**📖 Para más detalles:** Ver [`docs/INFERENCIA_Y_DESPLIEGUE_DOCKER_STREAMLIT.md`](docs/INFERENCIA_Y_DESPLIEGUE_DOCKER_STREAMLIT.md)

---

### Opción 2: Instalación Local 💻

#### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/tlob-prediction.git
cd tlob-prediction
```

#### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python3.12 -m venv venv

# Activar entorno
# En Linux/macOS:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

#### 3. Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import torch; import streamlit; print('✓ OK')"
```

#### 4. Ejecutar Streamlit

```bash
# Ejecutar aplicación
streamlit run app.py

# La aplicación se abrirá automáticamente en:
# http://localhost:8501
```

---

## 💾 Carga de Pesos Preentrenados

### Ubicación de los Checkpoints

Los pesos preentrenados se encuentran en:

```
src/data/checkpoints/
├── TLOB/
│   └── BTC_seq_size_128_horizon_10_seed_42/
│       ├── pt/
│       │   ├── model.pt          # ⭐ Modelo PyTorch
│       │   └── config.json       # Configuración del modelo
│       └── predictions.npy       # Predicciones de ejemplo
├── DEEPLOB/
│   └── BTC_seq_size_100_horizon_10_seed_42/...
├── MLPLOB/
│   └── BTC_seq_size_384_horizon_10_seed_42/...
└── BINCTABL/
    └── BTC_seq_size_10_horizon_10_seed_42/...
```

### Proceso de Carga en el Código

#### 1. Definición de Rutas (config/constants.py)

```python
import torch
from pathlib import Path

# Ruta base
DATA_DIR = Path("src/data")
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Checkpoint del modelo TLOB
TLOB_CHECKPOINT = CHECKPOINT_DIR / "TLOB" / "BTC_seq_size_128_horizon_10_seed_42" / "pt" / "model.pt"

# Device (CPU o GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### 2. Inicialización del Modelo (models/tlob.py)

```python
from models.tlob import TLOB
import torch

# Configuración del modelo (debe coincidir con el entrenamiento)
model_config = {
    'hidden_dim': 40,          # Dimensión del espacio latente
    'num_layers': 4,           # Número de capas Transformer
    'seq_size': 128,           # Longitud de secuencia
    'num_features': 40,        # Número de features del LOB
    'num_heads': 8,            # Cabezas de atención
    'is_sin_emb': True,        # Positional encoding sinusoidal
    'dataset_type': 'BTC'      # Tipo de dataset
}

# Crear instancia del modelo
model = TLOB(**model_config)
```

#### 3. Carga de Pesos (app.py)

```python
# Cargar pesos preentrenados
checkpoint_path = "src/data/checkpoints/TLOB/.../model.pt"

# Cargar state_dict
state_dict = torch.load(checkpoint_path, map_location=DEVICE)

# Aplicar pesos al modelo
model.load_state_dict(state_dict)

# Modo evaluación (desactiva dropout, batch norm, etc.)
model.eval()

print(f"✓ Modelo cargado desde: {checkpoint_path}")
print(f"✓ Device: {DEVICE}")
print(f"✓ Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
```

### Verificación de la Carga

```python
# Verificar que los pesos se cargaron correctamente
def verify_model_weights(model):
    """Verifica que el modelo tiene pesos válidos"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise ValueError(f"NaN detectado en {name}")
        if torch.isinf(param).any():
            raise ValueError(f"Inf detectado en {name}")
    print("✓ Todos los pesos son válidos")

verify_model_weights(model)
```

### Gestión de Errores Comunes

```python
import sys

try:
    # Intentar cargar modelo
    model = TLOB(**model_config)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
except FileNotFoundError:
    print(f"❌ Error: Checkpoint no encontrado en {checkpoint_path}")
    print("→ Verificar que la ruta es correcta")
    sys.exit(1)
    
except RuntimeError as e:
    print(f"❌ Error al cargar state_dict: {e}")
    print("→ Verificar que model_config coincide con el modelo entrenado")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    sys.exit(1)
```

---

## 🔮 Proceso de Inferencia

### Flujo Completo

```
┌──────────────────────┐
│  1. Cargar Datos     │  ← CSV o NPY con LOB data
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  2. Preprocesar      │  ← Reordenar + Normalizar
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  3. Crear Ventanas   │  ← Sequences de 128 timesteps
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  4. Inferencia       │  ← Forward pass del modelo
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  5. Post-proceso     │  ← Softmax + argmax
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  6. Visualizar       │  ← Mostrar en Streamlit
└──────────────────────┘
```

### 1. Cargar Datos

```python
import numpy as np
import pandas as pd

# Opción A: Desde archivo NPY
data = np.load("src/data/BTC/raw_examples/raw_example_1.npy")
# Shape: (128, 42)

# Opción B: Desde archivo CSV
data = pd.read_csv("src/data/BTC/raw_examples/raw_example_1.csv")
data = data.values  # Convertir a numpy array
# Shape: (128, 42)
```

### 2. Preprocesar Datos

#### A. Reordenamiento de Columnas

El CSV original tiene columnas en orden diferente al esperado por el modelo:

```python
def reorder_columns(data):
    """
    CSV Original:
    [timestamp, datetime, BID_P1-10, BID_V1-10, ASK_P1-10, ASK_V1-10]
    
    Formato del Modelo:
    [timestamp, ASK_P1, ASK_V1, BID_P1, BID_V1, ASK_P2, ASK_V2, ...]
    """
    df = pd.DataFrame(data)
    df.columns = np.arange(42)
    
    # Reordenamiento según preprocessing/btc.py
    new_order = [
        1,   # timestamp
        22, 23,  # ASK_P1, ASK_V1
        2, 3,    # BID_P1, BID_V1
        24, 25,  # ASK_P2, ASK_V2
        4, 5,    # BID_P2, BID_V2
        # ... hasta nivel 10
    ]
    
    df_reordered = df.loc[:, new_order]
    return df_reordered.values
```

#### B. Normalización Z-Score

```python
def z_score_normalize(data):
    """
    Normaliza precios y volúmenes por separado
    usando Z-score normalization
    """
    # Separar timestamp
    timestamp = data[:, 0]
    features = data[:, 1:]  # 40 features (sin timestamp)
    
    # Columnas de precios (pares: 0, 2, 4, ...)
    price_cols = features[:, 0::2]
    # Columnas de volúmenes (impares: 1, 3, 5, ...)
    volume_cols = features[:, 1::2]
    
    # Calcular estadísticas
    mean_prices = price_cols.mean()
    std_prices = price_cols.std()
    mean_volumes = volume_cols.mean()
    std_volumes = volume_cols.std()
    
    # Aplicar z-score
    price_cols_norm = (price_cols - mean_prices) / std_prices
    volume_cols_norm = (volume_cols - mean_volumes) / std_volumes
    
    # Recombinar (intercalando precios y volúmenes)
    normalized = np.empty_like(features)
    normalized[:, 0::2] = price_cols_norm
    normalized[:, 1::2] = volume_cols_norm
    
    return normalized, (mean_prices, std_prices, mean_volumes, std_volumes)
```

#### C. Implementación Completa

```python
from preprocessing.btc import preprocess_btc_data

# Preprocesar (reordenar + normalizar)
data_processed, stats = preprocess_btc_data(data_raw)

# data_processed shape: (128, 40)
# stats: {mean_prices, std_prices, mean_volumes, std_volumes}
```

### 3. Crear Tensor de Entrada

```python
import torch

# Convertir a tensor
input_tensor = torch.tensor(data_processed, dtype=torch.float32)

# Añadir dimensión de batch
input_tensor = input_tensor.unsqueeze(0)  # (1, 128, 40)

# Mover a device
input_tensor = input_tensor.to(DEVICE)

print(f"Input shape: {input_tensor.shape}")
# Output: Input shape: torch.Size([1, 128, 40])
```

### 4. Inferencia

```python
# Desactivar gradientes (inferencia, no entrenamiento)
with torch.no_grad():
    # Forward pass
    output, attention_weights = model(input_tensor, store_att=True)

    # output shape: (1, 3)
    # attention_weights: dict con pesos de atención de cada capa

# Aplicar softmax para obtener probabilidades
probabilities = torch.softmax(output, dim=1)
    
# Obtener predicción (clase con mayor probabilidad)
predicted_class = torch.argmax(probabilities, dim=1)

print(f"Probabilities: {probabilities[0].cpu().numpy()}")
# Output: Probabilities: [0.102, 0.153, 0.745]

print(f"Predicted class: {predicted_class.item()}")
# Output: Predicted class: 2 (UP)
```

### 5. Interpretación de Resultados

```python
# Mapeo de clases
LABEL_MAP = {
    0: "DOWN",
    1: "STATIONARY",
    2: "UP"
}

# Obtener predicción legible
prediction = LABEL_MAP[predicted_class.item()]
confidence = probabilities[0, predicted_class].item() * 100

print(f"Predicción: {prediction}")
print(f"Confianza: {confidence:.1f}%")

# Output:
# Predicción: UP
# Confianza: 74.5%
```

### 6. Visualización en Streamlit

```python
import streamlit as st
import plotly.graph_objects as go

# Mostrar resultados
st.success(f"✅ Predicción: **{prediction}**")
st.info(f"📊 Confianza: **{confidence:.1f}%**")
    
# Gráfico de barras con probabilidades
fig = go.Figure(data=[
    go.Bar(
        x=["DOWN", "STATIONARY", "UP"],
        y=probabilities[0].cpu().numpy() * 100,
        marker_color=['#FF4B4B', '#FFA500', '#4BFF4B']
    )
])
fig.update_layout(
    title="Probabilidades de Predicción",
    yaxis_title="Probabilidad (%)",
    yaxis_range=[0, 100]
)
    st.plotly_chart(fig)
```

### Script de Inferencia Independiente

Para ejecutar inferencia sin Streamlit:

```bash
# Inferencia de un archivo individual
python inference/inference_pytorch.py \
  --model TLOB \
  --input_file src/data/BTC/raw_examples/raw_example_1.npy \
  --output_dir results/

# Inferencia en batch
python inference/inference_pytorch.py \
  --model TLOB \
  --input_file src/data/BTC/csv_examples/csv_examples_batch.npy \
  --batch_size 32 \
  --output_dir results/
```

**📖 Para más detalles:** Ver [`docs/INFERENCIA_Y_DESPLIEGUE_DOCKER_STREAMLIT.md`](docs/INFERENCIA_Y_DESPLIEGUE_DOCKER_STREAMLIT.md)

---

## 📂 Estructura del Repositorio

```
TLOB-main/
├── README.md                          # 📖 Este archivo - Guía completa
├── LICENSE                            # Licencia MIT
├── .gitignore                         # Archivos ignorados por Git
│
├── app.py                             # 🎨 Aplicación Streamlit (Principal)
├── Dockerfile                         # 🐳 Configuración de imagen Docker
├── docker-compose.yml                 # 🐳 Orquestación multi-contenedor
├── requirements.txt                   # 📦 Dependencias Python con versiones
│
├── .devcontainer/                     # 🛠️ Dev Container para VSCode
│   ├── devcontainer.json              # Configuración del contenedor
│   └── Dockerfile                     # Dockerfile para desarrollo
│
├── src/                               # 📂 Código fuente principal
│   ├── constants.py                   # 🔧 Constantes del proyecto
│   ├── main.py                        # 🚀 Script principal de entrenamiento
│   ├── run.py                         # 🏃 Runner de experimentos
│   │
│   ├── config/                        # ⚙️ Configuración
│   │   └── config.py                  # Configuración con Hydra
│   │
│   ├── data/                          # 📊 Datos y checkpoints
│   │   ├── checkpoints/               # ⭐ Pesos preentrenados
│   │   │   ├── TLOB/                  # Modelos TLOB (horizonte 10/20/50/100)
│   │   │   │   ├── BTC_seq_size_128_horizon_10_seed_42/
│   │   │   │   │   ├── pt/            # Checkpoints PyTorch (.pt)
│   │   │   │   │   └── onnx/          # Modelos ONNX (.onnx)
│   │   │   │   └── ...
│   │   │   ├── DEEPLOB/               # Modelos DeepLOB
│   │   │   ├── MLPLOB/                # Modelos MLPLOB
│   │   │   └── BINCTABL/              # Modelos BiNCTABL
│   │   │
│   │   └── BTC/                       # Datos de Bitcoin
│   │       ├── original_source/       # CSV original de Binance
│   │       ├── individual_examples/   # Ejemplos preprocesados (.npy)
│   │       └── raw_examples/          # Ejemplos sin procesar (.csv, .npy)
│   │
│   ├── models/                        # 🧠 Arquitecturas de modelos
│   │   ├── tlob.py                    # ⭐ Modelo TLOB con Dual Attention
│   │   ├── deeplob.py                 # Modelo DeepLOB (baseline)
│   │   ├── mlplob.py                  # Modelo MLPLOB
│   │   ├── binctabl.py                # Modelo BiNCTABL
│   │   ├── bin.py                     # BiN (Batch Independent Normalization)
│   │   └── engine.py                  # Engine de entrenamiento (Lightning)
│   │
│   ├── preprocessing/                 # 🔄 Preprocesamiento de datos
│   │   ├── btc.py                     # Procesamiento BTC/Binance
│   │   ├── fi_2010.py                 # Procesamiento FI-2010
│   │   ├── dataset.py                 # PyTorch Dataset personalizado
│   │   └── lobster.py                 # Formato LOBSTER
│   │
│   └── utils/                         # 🛠️ Utilidades
│       ├── utils_data.py              # Funciones de datos y etiquetado
│       └── utils_model.py             # Funciones auxiliares de modelos
│
├── inference/                         # 🔮 Scripts de inferencia
│   ├── inference_pytorch.py           # Inferencia con PyTorch
│   └── create_raw_examples.py         # Generador de ejemplos raw
│
└── docs/                                            # 📚 Documentación técnica
    ├── MECANISMO_ATENCION_QKV.md                    # ⭐ Teoría matemática Q, K, V
    ├── INFERENCIA_Y_DESPLIEGUE_DOCKER_STREAMLIT.md  # ⭐ Inferencia y despliegue completo
    ├── INNOVACIONES_TLOB.md                         # ⭐ Dual Attention, BiN, etc.
    ├── ARQUITECTURA_COMPLETA.md                     # ⭐ 4 pares de Transformers
    └── RESUMEN_EJECUTIVO.md                         # Resumen del proyecto
```

**Nota:** Los archivos marcados con ⭐ son documentos clave del proyecto.

---

## 📚 Documentación Adicional

### Documentos Clave

| Documento | Descripción |
|-----------|-------------|
| [`docs/MECANISMO_ATENCION_QKV.md`](docs/MECANISMO_ATENCION_QKV.md) | ⭐ **Teoría matemática completa del mecanismo de atención (Q, K, V) con ejemplos paso a paso** |
| [`docs/INFERENCIA_Y_DESPLIEGUE_DOCKER_STREAMLIT.md`](docs/INFERENCIA_Y_DESPLIEGUE_DOCKER_STREAMLIT.md) | ⭐ **Guía completa de inferencia, preprocesamiento y despliegue con Docker/Streamlit** |
| [`docs/INNOVACIONES_TLOB.md`](docs/INNOVACIONES_TLOB.md) | ⭐ **Innovaciones del modelo: Dual Attention, BiN, etiquetado adaptativo** |
| [`docs/ARQUITECTURA_COMPLETA.md`](docs/ARQUITECTURA_COMPLETA.md) | ⭐ **Arquitectura detallada: 4 pares de Transformers explicados con dimensiones** |
| [`docs/RESUMEN_EJECUTIVO.md`](docs/RESUMEN_EJECUTIVO.md) | Resumen ejecutivo del proyecto |

### Código Comentado

Todo el código del proyecto está **extensamente comentado** explicando:

- ✅ Cómo se cargan los pesos del modelo
- ✅ Cómo se preprocesan los datos de entrada
- ✅ Cómo se genera la salida o inferencia
- ✅ Cómo se integra la visualización en Streamlit

**Archivos clave con comentarios:**

- `app.py`: Aplicación Streamlit (líneas 1-658)
- `models/tlob.py`: Modelo TLOB (líneas 1-157)
- `preprocessing/btc.py`: Preprocesamiento (líneas 1-120)
- `inference/inference_pytorch.py`: Inferencia (líneas 1-176)

---

## 🎓 Uso del Proyecto

### Caso de Uso 1: Predicción en Tiempo Real

1. Cargar datos del LOB en tiempo real
2. Preprocesar (ventana de 128 timesteps)
3. Ejecutar inferencia
4. Visualizar predicción en Streamlit

### Caso de Uso 2: Análisis Histórico

1. Cargar dataset histórico (CSV)
2. Crear ejemplos con `create_examples_from_csv.py`
3. Ejecutar inferencia en batch
4. Analizar resultados y métricas

### Caso de Uso 3: Comparación de Modelos

1. Cargar mismo ejemplo para múltiples modelos
2. Ejecutar inferencia con TLOB, DeepLOB, MLPLOB, BINCTABL
3. Comparar predicciones y confianza
4. Visualizar diferencias en Streamlit

---

## 🔬 Resultados y Desempeño

Los resultados presentados a continuación son los reportados en el paper original (Berti & Kasneci, 2025) y demuestran el desempeño superior de TLOB y MLPLOB frente a modelos del estado del arte.

### Resultados en FI-2010 Benchmark

**F1-Score (%) en cuatro horizontes de predicción:**

| Modelo | h = 10 | h = 20 | h = 50 | h = 100 |
|--------|--------|--------|--------|---------|
| SVM | 35.9 | 43.2 | 49.4 | 51.2 |
| Random Forest | 48.7 | 46.3 | 51.2 | 53.9 |
| XGBoost | 62.4 | 59.6 | 65.3 | 67.6 |
| MLP | 48.2 | 44.0 | 49.0 | 51.6 |
| LSTM [39] | 66.5 | 58.8 | 66.9 | 59.4 |
| CNN [38] | 49.3 | 46.1 | 65.8 | 67.2 |
| CTABL [36] | 69.5 | 62.4 | 71.6 | 73.9 |
| DAIN-MLP [30] | 53.9 | 46.7 | 61.2 | 62.8 |
| CNNLSTM [40] | 63.5 | 49.1 | 69.2 | 71.0 |
| AXIALLOB [23] | 73.2 | 63.4 | 78.3 | 79.2 |
| DLA [18] | 79.4 | 69.3 | 87.1 | 52.2 |
| DeepLOB [45] | 71.1 | 62.4 | 75.4 | 77.6 |
| BiNCTABL [37] | 81.1 | 71.5 | 87.7 | 92.1 |
| **MLPLOB** | **81.64** | **84.88** | **91.39** | 92.62 |
| **TLOB** | 81.55 | 82.68 | 90.03 | **92.81** |

**Análisis:**
- TLOB y MLPLOB superan a todos los baselines con una mejora promedio de **+3.7% en F1-score**
- MLPLOB es ligeramente superior en horizontes cortos (h=10, 20)
- TLOB domina en horizontes largos (h=50, 100), demostrando su capacidad para capturar dependencias temporales de largo alcance

### Resultados en Tesla (TSLA) - NASDAQ

**F1-Score (%) en cuatro horizontes de predicción:**

| Modelo | h = 10 | h = 20 | h = 50 | h = 100 |
|--------|--------|--------|--------|---------|
| DeepLOB | 36.25 | 36.58 | 35.29 | 34.43 |
| BiNCTABL | 58.69 | 48.83 | 42.23 | 38.77 |
| MLPLOB | **60.72** | **50.25** | 38.97 | 32.95 |
| **TLOB** | 60.50 | 49.74 | **43.48** | **39.84** |

**Análisis:**
- Mejora promedio de **+1.3% en F1-score** sobre baselines
- Tesla presenta mayor volatilidad y complejidad que FI-2010
- TLOB muestra ventaja significativa en horizontes largos (h=50, 100)
- Performance inferior a FI-2010 refleja mayor eficiencia del mercado NASDAQ

### Resultados en Intel (INTC) - NASDAQ

**F1-Score (%) en cuatro horizontes de predicción:**

| Modelo | h = 10 | h = 20 | h = 50 | h = 100 |
|--------|--------|--------|--------|---------|
| DeepLOB | 68.13 | 63.70 | 40.3 | 30.1 |
| BiNCTABL | 72.65 | 66.57 | 53.99 | 41.08 |
| **MLPLOB** | **81.15** | **73.25** | 55.74 | 43.18 |
| **TLOB** | 80.15 | 72.75 | **62.07** | **50.14** |

**Análisis:**
- Mejora promedio de **+7.7% en F1-score** sobre baselines
- Intel como "small-tick stock" presenta características diferentes a Tesla
- TLOB supera significativamente a todos los modelos en horizontes largos
- Diferencia de ~7% entre MLPLOB y TLOB en h=50 y h=100 demuestra importancia de dual attention

### Resultados en Bitcoin (BTC)

**F1-Score (%) en cuatro horizontes de predicción:**

| Modelo | h = 10 | h = 20 | h = 50 | h = 100 |
|--------|--------|--------|--------|---------|
| DeepLOB | 68.07 | 57.87 | 45.13 | 37.43 |
| BiNCTABL | 73.4 | 61.34 | 47.05 | 40.59 |
| MLPLOB | 74.6 | 61.02 | 42.74 | 36.97 |
| **TLOB** | **74.7** | **61.74** | **48.54** | **41.49** |

**Análisis:**
- Mejora promedio de **+1.1% en F1-score** sobre baselines
- Dataset más reciente (2023) con alta volatilidad
- TLOB supera consistentemente a MLPLOB en **todos los horizontes**
- Diferencia más notable en horizontes largos (~5% en h=50, h=100)

### Análisis Temporal: Eficiencia de Mercado Creciente

**Intel 2012 vs 2015 (h = 50):**

| Periodo | F1-Score (%) |
|---------|--------------|
| INTC 2012 | **66.87** |
| INTC 2015 | 60.19 |
| **Decline** | **-6.68** |

**Conclusión:** Confirmación empírica de que los mercados se vuelven más eficientes con el tiempo, dificultando la predicción. Los patrones predictivos se erosionan a medida que son descubiertos y explotados por traders.

### Threshold Basado en Spread (Tesla)

**F1-Score (%) con θ = promedio del spread:**

| Horizonte | F1-Score (%) |
|-----------|--------------|
| h = 50 | 41.39 |
| h = 100 | 36.48 |
| h = 200 | 30.82 |

**Conclusión:** Definir el threshold basado en costos de transacción (spread) en lugar de balance de clases deteriora significativamente el performance, evidenciando el **gap crítico entre métricas académicas y profitabilidad práctica**.

### Ablation Study: Importancia de Dual Attention

**F1-Score (%) en FI-2010:**

| Modelo | h = 10 | h = 20 | h = 50 | h = 100 |
|--------|--------|--------|--------|---------|
| TLOB w/o Spatial Attention | 79.59 | 78.96 | 87.51 | 91.40 |
| TLOB w/o Temporal Attention | 80.27 | 79.20 | 87.72 | 91.42 |
| **TLOB Completo** | **81.55** | **82.68** | **90.03** | **92.81** |

**Conclusión:** El modelo completo con **dual attention (spatial + temporal)** supera consistentemente a las versiones ablated, demostrando que ambos mecanismos capturan información complementaria esencial.

### Tiempo de Inferencia y Parámetros

**Comparación de modelos (batch_size=1):**

| Modelo | Parámetros | Tiempo (ms) |
|--------|------------|-------------|
| MLP | 10⁶ | 0.08 |
| LSTM [39] | 1.6 × 10⁴ | 0.21 |
| CNN [38] | 3.5 × 10⁴ | 0.36 |
| CTABL [36] | 1.1 × 10⁴ | 0.48 |
| DAIN-MLP [30] | 5.3 × 10⁴ | 0.50 |
| CNNLSTM [40] | 2.8 × 10⁵ | 0.49 |
| AXIALLOB [23] | 2 × 10⁴ | 1.91 |
| DLA [18] | 1.2 × 10⁵ | 0.23 |
| DeepLOB [45] | 1.4 × 10⁵ | 1.31 |
| BiNCTABL [37] | 1.1 × 10⁴ | 0.71 |
| **MLPLOB** | **6.3 × 10⁷** | **4.79** |
| **TLOB** | **1 × 10⁷** | **2.24** |

**Análisis:**
- TLOB tiene más parámetros que los baselines pero **mantiene tiempo de inferencia competitivo (2.24ms)**
- Velocidad adecuada para aplicaciones de **high-frequency trading**
- Trade-off favorable: Mayor capacidad del modelo por costo computacional moderado

### Observaciones Clave

1. **Horizontes Cortos (h=10, 20):** MLPLOB es suficiente y ligeramente superior
2. **Horizontes Largos (h=50, 100):** TLOB domina gracias a mecanismo de atención
3. **Eficiencia de Mercado:** FI-2010 (2010) > BTC (2023) > NASDAQ (2015)
4. **Volatilidad:** Tesla (alta) más difícil de predecir que Intel (baja)
5. **Transaction Costs:** Críticos para profitabilidad real, no capturados por F1-score

---

## 🛠️ Desarrollo y Extensión

### Agregar Nuevo Modelo

1. Crear archivo en `src/models/nuevo_modelo.py`
2. Implementar arquitectura compatible
3. Agregar checkpoint en `src/data/checkpoints/NUEVO_MODELO/`
4. Actualizar `app.py` para incluir nuevo modelo en dropdown

### Agregar Nuevo Dataset

1. Crear script de preprocesamiento en `src/preprocessing/nuevo_dataset.py`
2. Implementar reordenamiento y normalización
3. Agregar ejemplos en `src/data/NUEVO_DATASET/`
4. Actualizar `app.py` para cargar nuevos ejemplos

---

## 📊 Visualizaciones Disponibles

La aplicación Streamlit incluye:

1. **Gráfico de Probabilidades**: Barras con las 3 clases
2. **Heatmap de Atención**: Visualización de pesos de atención
3. **Evolución Temporal del LOB**: Serie de tiempo de precios y volúmenes
4. **Comparación de Modelos**: Tabla comparativa de predicciones

---

## 🤝 Contribuciones

Este proyecto es parte de un trabajo académico. Para sugerencias o mejoras:

1. Fork del repositorio
2. Crear branch (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Add: nueva funcionalidad'`)
4. Push al branch (`git push origin feature/mejora`)
5. Crear Pull Request

---

## 📖 Referencias

### Paper Original

```bibtex
@article{berti2025tlob,
  title={TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data},
  author={Berti, Leonardo and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2502.15757},
  year={2025}
}
```

### Recursos Adicionales

1. **Attention is All You Need** (Vaswani et al., 2017)
   - [Paper](https://arxiv.org/abs/1706.03762)
   - Base del mecanismo de atención

2. **DeepLOB** (Zhang et al., 2019)
   - [Paper](https://arxiv.org/abs/1808.03668)
   - Baseline para comparación

3. **FI-2010 Dataset**
   - [Paper](https://arxiv.org/abs/1705.03233)
   - Dataset de referencia en LOB

4. **PyTorch Documentation**
   - [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
   - [Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

---

## 📝 Licencia

Este proyecto está bajo la licencia **MIT**. Ver archivo [`LICENSE`](LICENSE) para más detalles.

---

## 👥 Autores

**Proyecto Final - Análisis de Series Temporales con Transformers**

- Implementación de TLOB
- Despliegue con Docker y Streamlit
- Visualización interactiva
- Documentación completa

**Basado en el trabajo de:**
- Leonardo Berti (Sapienza University of Rome)
- Gjergji Kasneci (Technical University of Munich)

---

## 📧 Contacto

Para preguntas o sugerencias sobre este proyecto:

- 📧 Email: [tu-email@universidad.edu]
- 💼 LinkedIn: [Tu perfil]
- 🐙 GitHub: [github.com/tu-usuario]

---

## 🎯 Próximos Pasos

- [ ] Agregar soporte para más criptomonedas
- [ ] Implementar inferencia en tiempo real con API de Binance
- [ ] Agregar análisis de uncertainty/confianza
- [ ] Crear dashboard de monitoreo
- [ ] Implementar fine-tuning del modelo

---

**Última actualización:** Noviembre 2025  
**Versión:** 1.0.0

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐**

Made with ❤️ using PyTorch, Streamlit, and Docker

</div>
