# Mecanismo de Atención del Modelo TLOB
## Queries (Q), Keys (K) y Values (V)

---

## 📚 **Índice**

1. [Introducción](#introducción)
2. [Arquitectura del Modelo TLOB](#arquitectura-del-modelo-tlob)
3. [Mecanismo de Atención](#mecanismo-de-atención)
4. [Generación de Q, K, V](#generación-de-q-k-v)
5. [Cálculo de Atención](#cálculo-de-atención)
6. [Implementación en Código](#implementación-en-código)
7. [Ejemplo Práctico](#ejemplo-práctico)
8. [Referencias](#referencias)

---

## 🎯 **Introducción**

El modelo **TLOB (Transformer for Limit Order Book)** utiliza una arquitectura Transformer para predecir movimientos de precios en mercados financieros basándose en datos de Limit Order Book (LOB).

El componente clave del Transformer es el **mecanismo de atención multi-cabeza (Multi-Head Attention)**, que permite al modelo identificar relaciones importantes entre diferentes timesteps y features del LOB.

---

## 🏗️ **Arquitectura del Modelo TLOB**

El modelo TLOB implementa una arquitectura Transformer dual que procesa los datos del LOB en dos dimensiones:

```
Input: (batch, seq_length=128, features=40)
         ↓
    BiN Normalization
         ↓
    Linear Embedding → hidden_dim
         ↓
    Positional Encoding
         ↓
┌────────────────────────────────┐
│   Transformer Layers (×N)      │
│                                 │
│  ┌───────────────────────────┐ │
│  │ Feature Attention         │ │  ← Atención entre features
│  │ (temporal × hidden_dim)   │ │
│  └───────────────────────────┘ │
│             ↓                   │
│  ┌───────────────────────────┐ │
│  │ Temporal Attention        │ │  ← Atención entre timesteps
│  │ (seq_length × features)   │ │
│  └───────────────────────────┘ │
└────────────────────────────────┘
         ↓
    Final MLP Layers
         ↓
Output: (batch, 3)  → [DOWN, STATIONARY, UP]
```

---

## 🔍 **Mecanismo de Atención**

### ¿Qué es la Atención?

La atención permite al modelo **enfocarse en diferentes partes de la entrada** al hacer predicciones. En el contexto del LOB:

- **Atención Temporal**: ¿Qué timesteps del pasado son más relevantes?
- **Atención Feature**: ¿Qué niveles de precios/volúmenes son más importantes?

### Componentes Clave

El mecanismo de atención se basa en tres matrices:

1. **Q (Queries)**: "¿Qué estoy buscando?"
2. **K (Keys)**: "¿Qué información tengo disponible?"
3. **V (Values)**: "¿Cuál es el contenido real de esa información?"

---

## ⚙️ **Generación de Q, K, V**

### Implementación en TLOB

```python
class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Tres proyecciones lineales independientes
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        q = self.q(x)  # (batch, seq_len, hidden_dim * num_heads)
        k = self.k(x)  # (batch, seq_len, hidden_dim * num_heads)
        v = self.v(x)  # (batch, seq_len, hidden_dim * num_heads)
        return q, k, v
```

### Proceso Paso a Paso

#### 1️⃣ **Input Embeddings**

```
Input LOB: (batch=32, seq_len=128, features=40)
    ↓ BiN Normalization
    ↓ Linear Embedding
Embedded: (batch=32, seq_len=128, hidden_dim=256)
```

#### 2️⃣ **Proyecciones Lineales**

Cada embedding pasa por **tres transformaciones lineales independientes**:

```python
# Para cada posición temporal t en la secuencia:
Q[t] = W_q @ x[t] + b_q  # Proyección Query
K[t] = W_k @ x[t] + b_k  # Proyección Key
V[t] = W_v @ x[t] + b_v  # Proyección Value
```

Donde:
- `W_q`, `W_k`, `W_v` son matrices de pesos aprendibles
- `x[t]` es el embedding en el timestep t

#### 3️⃣ **Multi-Head Attention**

Las proyecciones se dividen en múltiples "cabezas" (heads=8):

```
Q: (batch=32, seq_len=128, hidden_dim*num_heads=256*8)
    ↓ Reshape
Q: (batch=32, num_heads=8, seq_len=128, head_dim=256)

K: (batch=32, num_heads=8, seq_len=128, head_dim=256)
V: (batch=32, num_heads=8, seq_len=128, head_dim=256)
```

---

## 🧮 **Cálculo de Atención**

### Fórmula de Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

### Paso a Paso

#### 1️⃣ **Scores de Atención**

```python
# Producto punto entre Queries y Keys
scores = Q @ K.transpose(-2, -1)  # (batch, heads, seq_len, seq_len)

# Ejemplo: Para el timestep t=100
# scores[100, :] indica qué tanto debe "atender" t=100 a todos los otros timesteps
```

#### 2️⃣ **Scaling**

```python
d_k = hidden_dim  # 256
scores = scores / math.sqrt(d_k)  # Normalización para estabilidad numérica
```

¿Por qué dividir por √d_k?
- Evita que los valores sean demasiado grandes
- Previene que el softmax sature

#### 3️⃣ **Softmax (Pesos de Atención)**

```python
attention_weights = softmax(scores, dim=-1)  # (batch, heads, seq_len, seq_len)
```

Los pesos suman 1 para cada timestep:
```
∑ attention_weights[t, :] = 1.0
```

Interpretación:
- `attention_weights[100, 50] = 0.3` → El timestep 100 presta 30% de atención al timestep 50

#### 4️⃣ **Weighted Sum de Values**

```python
output = attention_weights @ V  # (batch, heads, seq_len, head_dim)
```

Cada timestep obtiene una **combinación ponderada** de todos los Values:

```
output[t] = Σ (attention_weights[t, s] * V[s])
            s=0...seq_len
```

---

## 💻 **Implementación en Código**

### Clase TransformerLayer Completa

```python
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Layer Normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Módulo de Q, K, V
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        
        # Multi-Head Attention de PyTorch
        self.attention = nn.MultiheadAttention(
            hidden_dim * num_heads, 
            num_heads, 
            batch_first=True
        )
        
        # MLP Feed-Forward
        self.mlp = MLP(hidden_dim, hidden_dim*4, final_dim)
        
        # Proyección final
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
    def forward(self, x):
        # Residual connection
        res = x
        
        # 1. Generar Q, K, V
        q, k, v = self.qkv(x)
        
        # 2. Aplicar atención multi-cabeza
        x, att = self.attention(q, k, v, 
                                average_attn_weights=False, 
                                need_weights=True)
        
        # 3. Proyección lineal
        x = self.w0(x)
        
        # 4. Residual connection
        x = x + res
        
        # 5. Layer Normalization
        x = self.norm(x)
        
        # 6. Feed-Forward MLP
        x = self.mlp(x)
        
        # 7. Segunda residual connection (si dimensiones coinciden)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
            
        return x, att  # Retorna output y pesos de atención
```

---

## 🎓 **Ejemplo Práctico**

### Datos de Entrada: Limit Order Book

```python
import numpy as np
import torch

# Datos del LOB (simplificado)
batch_size = 1
seq_len = 128  # 128 timesteps (32 segundos @ 250ms)
features = 40  # 10 niveles × 4 (ASK_P, ASK_V, BID_P, BID_V)

lob_data = torch.randn(batch_size, seq_len, features)
```

### Dimensiones en Cada Paso

```python
# 1. Input
input_shape = (1, 128, 40)

# 2. Después de BiN Normalization
normalized_shape = (1, 128, 40)

# 3. Después de Embedding
hidden_dim = 256
embedded_shape = (1, 128, 256)

# 4. Q, K, V
num_heads = 8
qkv_shape = (1, 128, 256*8)  # (1, 128, 2048)

# 5. Reshape para Multi-Head
# (batch, seq_len, hidden*heads) → (batch, heads, seq_len, head_dim)
q_multihead = (1, 8, 128, 256)
k_multihead = (1, 8, 128, 256)
v_multihead = (1, 8, 128, 256)

# 6. Attention Scores
scores = (1, 8, 128, 128)  # Cada timestep atiende a todos los timesteps

# 7. Attention Weights (después de softmax)
attention_weights = (1, 8, 128, 128)

# 8. Output (weighted sum de Values)
attention_output = (1, 8, 128, 256)

# 9. Concatenar cabezas
output = (1, 128, 2048)

# 10. Proyección final
final_output = (1, 128, 256)
```

### Visualización de Pesos de Atención

```python
import matplotlib.pyplot as plt
import seaborn as sns

# att shape: (1, num_heads, seq_len, seq_len)
att_weights = att[0, 0, :, :].detach().cpu().numpy()  # Primera cabeza

plt.figure(figsize=(10, 8))
sns.heatmap(att_weights, cmap='viridis', cbar=True)
plt.title('Attention Weights - Head 0')
plt.xlabel('Key Position (timestep)')
plt.ylabel('Query Position (timestep)')
plt.show()
```

**Interpretación del heatmap:**
- **Filas (Query)**: Timestep que está "preguntando"
- **Columnas (Key)**: Timesteps disponibles para "responder"
- **Color brillante**: Alta atención → Ese timestep es importante
- **Color oscuro**: Baja atención → Ese timestep es menos relevante

---

## 🔬 **Innovaciones del Modelo TLOB**

### 1. **Atención Dual (Temporal + Feature)**

A diferencia de Transformers tradicionales, TLOB aplica atención en **dos dimensiones**:

```python
# Capa 1: Atención sobre FEATURES (qué niveles del LOB son importantes)
feature_att, att1 = transformer_layer1(x)  # (batch, seq_len, hidden_dim)

# Capa 2: Atención sobre TIEMPO (qué timesteps son importantes)
temporal_att, att2 = transformer_layer2(x.transpose(1, 2))  # (batch, hidden_dim, seq_len)
```

### 2. **BiN Normalization**

Normalización especializada para datos de series temporales financieras:

```python
class BiN(nn.Module):
    """Batch-Instance Normalization para LOB"""
    def forward(self, x):
        # Normaliza por batch Y por instancia
        batch_norm = (x - x.mean(dim=0)) / x.std(dim=0)
        instance_norm = (x - x.mean(dim=(1,2))) / x.std(dim=(1,2))
        return 0.5 * batch_norm + 0.5 * instance_norm
```

### 3. **Positional Encoding**

Codifica la posición temporal en el LOB:

```python
if is_sin_emb:
    # Sinusoidal (como en "Attention is All You Need")
    pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim)
else:
    # Aprendible
    pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))
```

---

## 📊 **Interpretación de los Pesos de Atención**

### ¿Qué nos dicen los pesos de atención?

Los pesos de atención revelan **qué información del pasado usa el modelo** para hacer predicciones:

#### Ejemplo 1: Atención Temporal

```
Predicción en t=128:
- attention_weights[127, 126] = 0.15  ← Presta 15% atención al timestep inmediato anterior
- attention_weights[127, 100] = 0.08  ← 8% a timestep 28 segundos atrás
- attention_weights[127, 50]  = 0.03  ← 3% a timestep lejano
```

**Interpretación**: El modelo considera principalmente los timesteps recientes, pero también mira eventos importantes del pasado.

#### Ejemplo 2: Atención Feature

```
Para predecir el movimiento:
- Alta atención a sell1/buy1 (primer nivel del LOB) → Spread es importante
- Media atención a sell2-sell5 → Profundidad del mercado relevante
- Baja atención a sell8-sell10 → Niveles lejanos menos importantes
```

---

## 🎯 **Ventajas del Mecanismo de Atención**

### 1. **Dependencias de Largo Alcance**

```python
# RNN/LSTM: Información se "olvida" con el tiempo
# Transformer: Puede atender a CUALQUIER timestep del pasado
attention_weights[127, 0] = 0.02  # Puede mirar el primer timestep directamente
```

### 2. **Paralelización**

```python
# RNN: Procesa secuencialmente (t=1 → t=2 → t=3 → ...)
# Transformer: Procesa TODOS los timesteps en paralelo
Q, K, V = compute_qkv(all_timesteps)  # Una sola operación matricial
```

### 3. **Interpretabilidad**

```python
# Los pesos de atención son interpretables
# Podemos visualizar QUÉ mira el modelo para hacer predicciones
plot_attention_heatmap(attention_weights)
```

---

## 📚 **Referencias**

1. **Artículo Original**: [TLOB: Transformer for Limit Order Books](https://arxiv.org/abs/2110.00551)
2. **Attention is All You Need**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
3. **Repositorio Original**: [GitHub - TLOB](https://github.com/SiddharthKarnam/TLOB)
4. **Multi-Head Attention**: [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

---

## 🔗 **Resumen Visual del Flujo Completo**

```
LOB Data (128 timesteps × 40 features)
         ↓
    BiN Normalization
         ↓
    Embedding (hidden_dim=256)
         ↓
    Add Positional Encoding
         ↓
    ┌─────────────────────────────┐
    │  ComputeQKV                 │
    │  ┌─────────────────────┐   │
    │  │ Q = Linear(x)       │   │  ← Queries
    │  │ K = Linear(x)       │   │  ← Keys
    │  │ V = Linear(x)       │   │  ← Values
    │  └─────────────────────┘   │
    └─────────────────────────────┘
         ↓
    ┌─────────────────────────────┐
    │  Multi-Head Attention       │
    │  ┌─────────────────────┐   │
    │  │ Scores = Q @ K^T    │   │
    │  │ Weights = Softmax   │   │
    │  │ Output = Weights@V  │   │
    │  └─────────────────────┘   │
    └─────────────────────────────┘
         ↓
    Residual Connection + LayerNorm
         ↓
    Feed-Forward MLP
         ↓
    Predicción: [DOWN, STATIONARY, UP]
```

---

**Última actualización**: Noviembre 2025  
**Autor**: Proyecto Final - Análisis de Series Temporales con Transformers


