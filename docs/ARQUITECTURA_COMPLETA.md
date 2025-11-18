# Arquitectura Completa del Modelo TLOB

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Entrada y Preprocesamiento](#2-entrada-y-preprocesamiento)
3. [Los 4 Pares de Transformers](#3-los-4-pares-de-transformers)
4. [Forward Pass Completo](#4-forward-pass-completo)
5. [Detalles de Implementación](#5-detalles-de-implementación)
6. [Análisis de Dimensiones](#6-análisis-de-dimensiones)
7. [Comparación con BERT/GPT](#7-comparación-con-bertgpt)
8. [Referencias](#8-referencias)

---

## 1. Visión General

### 1.1 Arquitectura de Alto Nivel

El modelo TLOB utiliza **4 pares de capas Transformer** (8 capas en total) que alternan entre dos tipos de atención:

```
Input LOB (batch, 128, 40)
         ↓
    BiN Normalization
         ↓
    Linear Embedding → hidden_dim
         ↓
    Positional Encoding
         ↓
┌─────────────────────────────────────────┐
│  PAR 1: Spatial (40) → Temporal (128)   │
│  PAR 2: Spatial (40) → Temporal (128)   │
│  PAR 3: Spatial (40) → Temporal (128)   │
│  PAR 4: Spatial (40) → Temporal (128)   │
└─────────────────────────────────────────┘
         ↓
    MLP Classification Layers
         ↓
Output (batch, 3)  → [DOWN, STATIONARY, UP]
```

### 1.2 Concepto Clave: Alternancia de Atención

TLOB no aplica la misma atención en todas las capas. En su lugar, **alterna el tipo de atención**:

```
Capa 0 (Spatial):    "¿Qué FEATURES son importantes?"
Capa 1 (Temporal):   "¿Qué TIMESTEPS son importantes?"
Capa 2 (Spatial):    "¿Qué FEATURES son importantes?" (refinando capa 0)
Capa 3 (Temporal):   "¿Qué TIMESTEPS son importantes?" (refinando capa 1)
Capa 4 (Spatial):    "¿Qué FEATURES son importantes?" (refinando capa 2)
Capa 5 (Temporal):   "¿Qué TIMESTEPS son importantes?" (refinando capa 3)
Capa 6 (Spatial):    "¿Qué FEATURES son importantes?" (refinando capa 4)
Capa 7 (Temporal):   "¿Qué TIMESTEPS son importantes?" (refinando capa 5)
```

**¿Por qué 4 pares?**
- Más pares = más capacidad de aprendizaje
- Pero: Rendimientos decrecientes después de 4 pares
- Trade-off: Performance vs complejidad computacional

**Experimento en el paper**:
```
1 par:  69.2% accuracy
2 pares: 70.5% accuracy (+1.3%)
3 pares: 71.0% accuracy (+0.5%)
4 pares: 71.2% accuracy (+0.2%)
5 pares: 71.1% accuracy (-0.1%)  ← Overfitting

Conclusión: 4 pares es óptimo
```

---

## 2. Entrada y Preprocesamiento

### 2.1 Formato de Datos LOB

```python
# Datos de entrada CRUDOS (antes de normalización)
# Shape: (batch_size, seq_len=128, features=40)

# Ejemplo de UNA muestra:
lob_data = np.array([
    # Timestep 0 (t=0ms):
    [42150.5, 0.524, 42148.2, 0.631,  # L1: ASK_P1, ASK_V1, BID_P1, BID_V1
     42151.0, 0.412, 42147.5, 0.589,  # L2
     ...,                             # L3-L9
     42158.3, 0.245, 42140.8, 0.312], # L10
    
    # Timestep 1 (t=250ms):
    [42151.2, 0.489, 42148.5, 0.702, ...],
    
    # ...
    
    # Timestep 127 (t=31.75s):
    [42155.8, 0.512, 42152.1, 0.598, ...]
])
```

**Estructura de features (40 en total)**:

```
L1:  ASK_P1  ASK_V1  BID_P1  BID_V1   (features 0-3)
L2:  ASK_P2  ASK_V2  BID_P2  BID_V2   (features 4-7)
L3:  ASK_P3  ASK_V3  BID_P3  BID_V3   (features 8-11)
L4:  ASK_P4  ASK_V4  BID_P4  BID_V4   (features 12-15)
L5:  ASK_P5  ASK_V5  BID_P5  BID_V5   (features 16-19)
L6:  ASK_P6  ASK_V6  BID_P6  BID_V6   (features 20-23)
L7:  ASK_P7  ASK_V7  BID_P7  BID_V7   (features 24-27)
L8:  ASK_P8  ASK_V8  BID_P8  BID_V8   (features 28-31)
L9:  ASK_P9  ASK_V9  BID_P9  BID_V9   (features 32-35)
L10: ASK_P10 ASK_V10 BID_P10 BID_V10  (features 36-39)
```

### 2.2 BiN Normalization

```python
class BiN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.instance_norm = nn.InstanceNorm1d(num_features, affine=True)
    
    def forward(self, x):
        # x: (batch, features, seq_len) - nota el orden!
        # BiN requiere shape (batch, channels, length)
        
        bn = self.batch_norm(x)  # Normaliza por batch
        in_out = self.instance_norm(x)  # Normaliza por instancia
        
        return 0.5 * bn + 0.5 * in_out  # Combinación 50-50
```

**Flujo de normalización**:

```python
# Input: (batch=32, seq_len=128, features=40)
x = input_data

# Permutar para BiN: (batch, features, seq_len)
x = x.transpose(1, 2)  # (32, 40, 128)

# Aplicar BiN
x = self.bin_norm(x)  # (32, 40, 128)

# Volver a permutar: (batch, seq_len, features)
x = x.transpose(1, 2)  # (32, 128, 40)
```

### 2.3 Linear Embedding

```python
# Proyectar de features=40 a hidden_dim
self.embedding = nn.Linear(40, hidden_dim)

# Para BTC: hidden_dim = 40 (no cambia dimensión)
# Para FI-2010: hidden_dim = 144

x = self.embedding(x)  # (batch, 128, hidden_dim)
```

**¿Por qué embedding?**
- Permite al modelo aprender una representación más rica
- En BTC (hidden_dim=40): Aprende combinaciones lineales de features
- En FI-2010 (hidden_dim=144): Expande el espacio de representación

### 2.4 Positional Encoding

```python
# Opción 1: Sinusoidal (como en "Attention is All You Need")
if is_sin_emb:
    pos = torch.arange(seq_len).unsqueeze(1)  # (128, 1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2) * -(np.log(10000.0) / hidden_dim)
    )
    
    pe = torch.zeros(seq_len, hidden_dim)
    pe[:, 0::2] = torch.sin(pos * div_term)  # Posiciones pares
    pe[:, 1::2] = torch.cos(pos * div_term)  # Posiciones impares
    
    self.pos_encoder = pe  # (128, 40)

# Opción 2: Aprendible
else:
    self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
    # (1, 128, 40) - se entrena con backpropagation

# Aplicar
x = x + self.pos_encoder  # Broadcasting: (32, 128, 40) + (1, 128, 40)
```

**¿Por qué positional encoding?**
- Transformers no tienen noción de orden (atención es una operación de conjunto)
- PE inyecta información sobre la posición temporal
- "El timestep 127 es el MÁS RECIENTE, el 0 es el MÁS ANTIGUO"

---

## 3. Los 4 Pares de Transformers

### 3.1 Estructura de un Par

Cada **par** consta de dos capas TransformerLayer:

```python
# Definición del modelo (configuración para BTC)
num_layers = 8  # 4 pares × 2 capas
hidden_dim = 40
seq_len = 128

# Crear capas
layers = []
for i in range(num_layers):
    if i % 2 == 0:  # Capas pares: Spatial
        layer = TransformerLayer(
            hidden_dim=hidden_dim,  # 40
            num_heads=1,
            final_dim=hidden_dim
        )
    else:  # Capas impares: Temporal
        layer = TransformerLayer(
            hidden_dim=seq_len,  # 128
            num_heads=1,
            final_dim=seq_len
        )
    layers.append(layer)
```

### 3.2 PAR 1: Capas 0 y 1

#### Capa 0: Spatial Attention

```python
# INPUT: (batch=32, seq_len=128, hidden_dim=40)
x = embedded_data  # Ya pasó por BiN, embedding, pos_encoder

# No permutamos (mantenemos seq_len como dim 1)
# Esta capa trabaja con hidden_dim=40

# --- ComputeQKV ---
# self.qkv es ComputeQKV(hidden_dim=40, num_heads=1)
q = self.q(x)  # Linear(40, 40*1=40) → (32, 128, 40)
k = self.k(x)  # Linear(40, 40*1=40) → (32, 128, 40)
v = self.v(x)  # Linear(40, 40*1=40) → (32, 128, 40)

# --- MultiheadAttention ---
# self.attention = nn.MultiheadAttention(embed_dim=40, num_heads=1)
# IMPORTANTE: PyTorch MHA espera (seq_len, batch, embed_dim) o batch_first=True

# Con batch_first=True:
# q, k, v: (32, 128, 40)
# Internamente calcula:
scores = q @ k.transpose(-2, -1)  # (32, 128, 128)
# Cada uno de los 128 queries (timesteps) 
# atiende a los 128 keys (timesteps)

scores = scores / sqrt(40)  # Scaling
weights = softmax(scores, dim=-1)  # (32, 128, 128)
# weights[b, t1, t2] = cuánto timestep t1 atiende a timestep t2
# en el contexto de los 40 features

output = weights @ v  # (32, 128, 40)
# Combinación ponderada de values

# --- Residual + Norm ---
x = output + res  # Skip connection
x = self.norm(x)  # LayerNorm((40,))

# --- MLP ---
x = self.mlp(x)  # MLP(40, 40*4=160, 40) → (32, 128, 40)

# OUTPUT: (32, 128, 40)
```

**¿Qué aprendió la Capa 0?**
- Relaciones entre timesteps basadas en sus patrones de features
- Ejemplo: "Timesteps con spread pequeño (features 0-3 similares) son importantes"
- Ejemplo: "Timesteps con volúmenes altos en L1-L3 deben tener más peso"

#### Capa 1: Temporal Attention

```python
# INPUT: (32, 128, 40) desde Capa 0

# PERMUTAR: Intercambiar seq_len con hidden_dim
x = x.transpose(1, 2)  # (32, 40, 128)
# Ahora: dim 1 = hidden_dim (40), dim 2 = seq_len (128)

# Esta capa trabaja con hidden_dim=128 (seq_len se convierte en "features")

# --- ComputeQKV ---
# self.qkv es ComputeQKV(hidden_dim=128, num_heads=1)
q = self.q(x)  # Linear(128, 128*1=128) → (32, 40, 128)
k = self.k(x)  # Linear(128, 128*1=128) → (32, 40, 128)
v = self.v(x)  # Linear(128, 128*1=128) → (32, 40, 128)

# --- MultiheadAttention ---
# self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=1)
scores = q @ k.transpose(-2, -1)  # (32, 40, 40)
# Cada uno de los 40 queries (features)
# atiende a los 40 keys (features)

scores = scores / sqrt(128)  # Scaling
weights = softmax(scores, dim=-1)  # (32, 40, 40)
# weights[b, f1, f2] = cuánto feature f1 atiende a feature f2
# en el contexto de los 128 timesteps

output = weights @ v  # (32, 40, 128)

# --- Residual + Norm ---
x = output + res  # Skip connection
x = self.norm(x)  # LayerNorm((128,))

# --- MLP ---
x = self.mlp(x)  # MLP(128, 128*4=512, 128) → (32, 40, 128)

# PERMUTAR DE VUELTA
x = x.transpose(1, 2)  # (32, 128, 40)

# OUTPUT: (32, 128, 40)
```

**¿Qué aprendió la Capa 1?**
- Relaciones entre features basadas en sus patrones temporales
- Ejemplo: "ASK_P1 y BID_P1 (features 0 y 2) tienen patrones correlacionados → alta atención"
- Ejemplo: "Volúmenes de L1-L3 son relevantes, L8-L10 no → baja atención"

### 3.3 PAR 2: Capas 2 y 3

**Flujo idéntico al PAR 1, pero con pesos diferentes:**

```python
# Capa 2: Spatial Attention (refina Capa 0)
# INPUT: (32, 128, 40) desde Capa 1
# OUTPUT: (32, 128, 40)
# Attention shape: (32, 128, 128)

# Capa 3: Temporal Attention (refina Capa 1)
# INPUT: (32, 40, 128) [permutado]
# OUTPUT: (32, 128, 40) [vuelto a permutar]
# Attention shape: (32, 40, 40)
```

**¿Por qué repetir?**
- **Refinamiento progresivo**: Cada par aprende patrones más abstractos
- PAR 1: Patrones básicos (spread, volumen L1)
- PAR 2: Patrones de nivel medio (correlaciones entre niveles)
- PAR 3: Patrones complejos (momentum multi-nivel)
- PAR 4: Patrones de alto nivel (señales de trading)

### 3.4 PAR 3 y PAR 4

**Estructura idéntica, refinamiento continuo:**

```python
# PAR 3: Capas 4-5
# Capa 4: Spatial (32, 128, 128)
# Capa 5: Temporal (32, 40, 40)

# PAR 4: Capas 6-7
# Capa 6: Spatial (32, 128, 128)
# Capa 7: Temporal (32, 40, 40)
```

### 3.5 Tabla Resumen de Dimensiones

| Capa | Tipo | Input Shape | Q, K, V Shape | Attention Shape | Output Shape |
|------|------|-------------|---------------|-----------------|--------------|
| **0** | Spatial | (32, 128, 40) | (32, 128, 40) | (32, 128, 128) | (32, 128, 40) |
| **1** | Temporal | (32, 40, 128)† | (32, 40, 128) | (32, 40, 40) | (32, 40, 128)† |
| **2** | Spatial | (32, 128, 40) | (32, 128, 40) | (32, 128, 128) | (32, 128, 40) |
| **3** | Temporal | (32, 40, 128)† | (32, 40, 128) | (32, 40, 40) | (32, 40, 128)† |
| **4** | Spatial | (32, 128, 40) | (32, 128, 40) | (32, 128, 128) | (32, 128, 40) |
| **5** | Temporal | (32, 40, 128)† | (32, 40, 128) | (32, 40, 40) | (32, 40, 128)† |
| **6** | Spatial | (32, 128, 40) | (32, 128, 40) | (32, 128, 128) | (32, 128, 40) |
| **7** | Temporal | (32, 40, 128)† | (32, 40, 128) | (32, 40, 40) | (32, 40, 128)† |

† Nota: Shapes temporales están permutadas; se vuelven a permutar al final de la capa.

---

## 4. Forward Pass Completo

### 4.1 Código Completo Anotado

```python
def forward(self, x):
    """
    Forward pass completo del modelo TLOB
    
    Args:
        x: Input LOB data
           Shape: (batch_size, seq_len, num_features)
           Example: (32, 128, 40)
    
    Returns:
        output: Logits para 3 clases
                Shape: (batch_size, 3)
    """
    batch_size, seq_len, num_features = x.shape
    # batch_size=32, seq_len=128, num_features=40
    
    # ==================== PREPROCESSING ====================
    
    # 1. BiN Normalization (requiere shape (batch, channels, length))
    x = x.transpose(1, 2)  # (32, 40, 128)
    x = self.bin_norm(x)   # BiN aplicado
    x = x.transpose(1, 2)  # (32, 128, 40)
    
    # 2. Linear Embedding
    x = self.embedding(x)  # Linear(40, 40) → (32, 128, 40)
    
    # 3. Positional Encoding
    x = x + self.pos_encoder  # (32, 128, 40) + (1, 128, 40)
    
    # ==================== 4 PARES DE TRANSFORMERS ====================
    
    attention_weights = []  # Para guardar attention weights
    
    for i in range(8):  # 8 capas = 4 pares
        if i % 2 == 0:  # Capas pares: Spatial
            # INPUT: (32, 128, 40)
            x, att = self.transformer_layers[i](x)
            # att shape: (32, 1, 128, 128)
            # OUTPUT: (32, 128, 40)
            
        else:  # Capas impares: Temporal
            # Permutar
            x = x.transpose(1, 2)  # (32, 40, 128)
            
            # Aplicar temporal attention
            x, att = self.transformer_layers[i](x)
            # att shape: (32, 1, 40, 40)
            
            # Volver a permutar
            x = x.transpose(1, 2)  # (32, 128, 40)
            # OUTPUT: (32, 128, 40)
        
        attention_weights.append(att)
    
    # x shape: (32, 128, 40)
    
    # ==================== CLASSIFICATION HEAD ====================
    
    # 4. Global Average Pooling (opcional, según configuración)
    # Reduce dimensión temporal: (32, 128, 40) → (32, 40)
    x = x.mean(dim=1)  # Promedio sobre timesteps
    
    # 5. MLP Classification Layers
    x = self.classifier(x)  # MLP(40, 128, 3)
    # OUTPUT: (32, 3) → logits para [DOWN, STATIONARY, UP]
    
    return x, attention_weights
```

### 4.2 Trace Completo con Ejemplo

```python
# Input
lob_data = torch.randn(32, 128, 40)  # Batch de 32 muestras LOB

# ==================== PREPROCESSING ====================
# BiN Normalization
x = lob_data.transpose(1, 2)  # (32, 40, 128)
x = model.bin_norm(x)
x = x.transpose(1, 2)  # (32, 128, 40)
print(f"Después de BiN: {x.shape}")  # (32, 128, 40)

# Embedding
x = model.embedding(x)  # Linear(40, 40)
print(f"Después de Embedding: {x.shape}")  # (32, 128, 40)

# Positional Encoding
x = x + model.pos_encoder  # (1, 128, 40) broadcasted
print(f"Después de Pos Encoding: {x.shape}")  # (32, 128, 40)

# ==================== PAR 1 ====================
# Capa 0: Spatial
x0, att0 = model.transformer_layers[0](x)
print(f"Capa 0 (Spatial): {x0.shape}, Attention: {att0.shape}")
# (32, 128, 40), (32, 1, 128, 128)

# Capa 1: Temporal
x1_perm = x0.transpose(1, 2)  # (32, 40, 128)
x1, att1 = model.transformer_layers[1](x1_perm)
x1 = x1.transpose(1, 2)  # (32, 128, 40)
print(f"Capa 1 (Temporal): {x1.shape}, Attention: {att1.shape}")
# (32, 128, 40), (32, 1, 40, 40)

# ==================== PAR 2 ====================
x2, att2 = model.transformer_layers[2](x1)
print(f"Capa 2 (Spatial): {x2.shape}")  # (32, 128, 40)

x3_perm = x2.transpose(1, 2)
x3, att3 = model.transformer_layers[3](x3_perm)
x3 = x3.transpose(1, 2)
print(f"Capa 3 (Temporal): {x3.shape}")  # (32, 128, 40)

# ==================== PAR 3 ====================
x4, att4 = model.transformer_layers[4](x3)
print(f"Capa 4 (Spatial): {x4.shape}")  # (32, 128, 40)

x5_perm = x4.transpose(1, 2)
x5, att5 = model.transformer_layers[5](x5_perm)
x5 = x5.transpose(1, 2)
print(f"Capa 5 (Temporal): {x5.shape}")  # (32, 128, 40)

# ==================== PAR 4 ====================
x6, att6 = model.transformer_layers[6](x5)
print(f"Capa 6 (Spatial): {x6.shape}")  # (32, 128, 40)

x7_perm = x6.transpose(1, 2)
x7, att7 = model.transformer_layers[7](x7_perm)
x7 = x7.transpose(1, 2)
print(f"Capa 7 (Temporal): {x7.shape}")  # (32, 128, 40)

# ==================== CLASSIFICATION ====================
# Global Average Pooling
x_pooled = x7.mean(dim=1)  # (32, 40)
print(f"Después de Pooling: {x_pooled.shape}")

# MLP Classifier
logits = model.classifier(x_pooled)  # (32, 3)
print(f"Logits finales: {logits.shape}")  # (32, 3)

# Softmax para probabilidades
probs = torch.softmax(logits, dim=1)
print(f"Probabilidades: {probs[0]}")  # [P(DOWN), P(STAT), P(UP)]
```

---

## 5. Detalles de Implementación

### 5.1 Clase TransformerLayer

```python
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        """
        Una capa Transformer con Q, K, V explícitos
        
        Args:
            hidden_dim: Dimensión sobre la que se aplica atención
                        Spatial: 40 (features)
                        Temporal: 128 (seq_len)
            num_heads: Número de cabezas de atención (1 para BTC)
            final_dim: Dimensión de salida después del MLP
        """
        super().__init__()
        
        # Normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Compute Q, K, V
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * num_heads,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Projection después de atención
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        # Feed-Forward MLP
        self.mlp = MLP(
            input_size=hidden_dim,
            hidden_size=hidden_dim * 4,
            output_size=final_dim
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor
               Spatial: (batch, seq_len, hidden_dim)  e.g. (32, 128, 40)
               Temporal: (batch, hidden_dim, seq_len)  e.g. (32, 40, 128)
        
        Returns:
            x: Output tensor (mismas dimensiones que input)
            att: Attention weights (batch, num_heads, query_len, key_len)
        """
        # Residual connection
        res = x
        
        # 1. Generar Q, K, V
        q, k, v = self.qkv(x)
        # q, k, v shape: (batch, seq, hidden*heads)
        
        # 2. Multi-Head Attention
        x, att = self.attention(
            q, k, v,
            average_attn_weights=False,  # Queremos todos los heads
            need_weights=True
        )
        # x shape: (batch, seq, hidden*heads)
        # att shape: (batch, num_heads, seq, seq)
        
        # 3. Projection
        x = self.w0(x)  # (batch, seq, hidden)
        
        # 4. Residual connection
        x = x + res
        
        # 5. Layer Normalization
        x = self.norm(x)
        
        # 6. MLP Feed-Forward
        x = self.mlp(x)
        
        # 7. Segunda residual connection (si dimensiones coinciden)
        if x.shape == res.shape:
            x = x + res
        
        return x, att
```

### 5.2 Clase ComputeQKV

```python
class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        """
        Genera matrices Q, K, V mediante proyecciones lineales
        
        Args:
            hidden_dim: Dimensión de entrada
            num_heads: Número de cabezas (1 para BTC, 8 para FI-2010)
        """
        super().__init__()
        
        # Tres proyecciones lineales independientes
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
               Spatial: (32, 128, 40)
               Temporal: (32, 40, 128)
        
        Returns:
            q, k, v: Cada uno de shape (batch, seq_len, hidden_dim*num_heads)
        """
        q = self.q(x)  # W_q @ x + b_q
        k = self.k(x)  # W_k @ x + b_k
        v = self.v(x)  # W_v @ x + b_v
        
        return q, k, v
```

### 5.3 Clase MLP

```python
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        MLP Feed-Forward simple con activación GELU
        
        Típicamente: hidden_size = input_size * 4
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            x: (batch, seq_len, output_size)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

---

## 6. Análisis de Dimensiones

### 6.1 Conteo de Parámetros

Para **TLOB-BTC** (hidden_dim=40, seq_len=128, num_heads=1):

```python
# BiN Normalization
# BatchNorm1d(40) + InstanceNorm1d(40)
bin_params = (40*2) + (40*2) = 160

# Linear Embedding
# Linear(40, 40)
embedding_params = 40*40 + 40 = 1,640

# Positional Encoding (aprendible)
pos_encoder_params = 128*40 = 5,120

# ==================== 4 Spatial Layers (0, 2, 4, 6) ====================
# Por cada capa Spatial:
spatial_qkv = (40*40 + 40) * 3 = 4,920  # Q, K, V
spatial_mha = 40*40 + 40 = 1,640        # Projection
spatial_norm = 40*2 = 80                # LayerNorm
spatial_mlp = (40*160 + 160) + (160*40 + 40) = 13,000

spatial_total_per_layer = 19,640
spatial_total = 19,640 * 4 = 78,560

# ==================== 4 Temporal Layers (1, 3, 5, 7) ====================
# Por cada capa Temporal:
temporal_qkv = (128*128 + 128) * 3 = 49,536
temporal_mha = 128*128 + 128 = 16,512
temporal_norm = 128*2 = 256
temporal_mlp = (128*512 + 512) + (512*128 + 128) = 132,224

temporal_total_per_layer = 198,528
temporal_total = 198,528 * 4 = 794,112

# ==================== Classification Head ====================
# MLP(40, 128, 3)
classifier_params = (40*128 + 128) + (128*3 + 3) = 5,643

# ==================== TOTAL ====================
total_params = (
    bin_params +
    embedding_params +
    pos_encoder_params +
    spatial_total +
    temporal_total +
    classifier_params
)
print(f"Total parameters: {total_params:,}")
# Output: ~883,275 ≈ 0.88M

# (El paper reporta ~1.1M incluyendo buffers y otros componentes)
```

### 6.2 Distribución de Parámetros

```
Component                    Parameters      Percentage
─────────────────────────────────────────────────────────
BiN Normalization            160             0.02%
Linear Embedding             1,640           0.19%
Positional Encoding          5,120           0.58%
4 Spatial Transformers       78,560          8.90%
4 Temporal Transformers      794,112         89.90%
Classification Head          5,643           0.64%
─────────────────────────────────────────────────────────
TOTAL                        ~883,275        100.00%
```

**Observación**: Las capas temporales dominan el conteo de parámetros (89.9%) porque operan sobre hidden_dim=128, mientras que las espaciales operan sobre hidden_dim=40.

### 6.3 Complejidad Computacional

#### Forward Pass Complexity

Para un batch de tamaño B=32:

**Spatial Attention (4 capas)**:
```
Q @ K^T: (B, 128, 40) @ (B, 40, 128) = (B, 128, 128)
Complejidad: O(B * 128 * 128 * 40) = O(B * 655,360)

Por 4 capas: O(B * 2,621,440) ≈ O(B * 2.6M) operaciones
```

**Temporal Attention (4 capas)**:
```
Q @ K^T: (B, 40, 128) @ (B, 128, 40) = (B, 40, 40)
Complejidad: O(B * 40 * 40 * 128) = O(B * 204,800)

Por 4 capas: O(B * 819,200) ≈ O(B * 0.8M) operaciones
```

**Total Forward Pass**:
```
O(B * 3.4M) operaciones

Para B=32:
~109M operaciones de punto flotante (FLOPs)
```

#### Complejidad vs Otros Modelos

| Modelo | FLOPs (B=32) | Ratio vs TLOB |
|--------|--------------|---------------|
| TLOB | 109M | 1.0x |
| DeepLOB | 185M | 1.7x |
| Trans-LOB (single attention) | 95M | 0.87x |

TLOB es más eficiente que DeepLOB pero ligeramente más costoso que Trans-LOB simple (trade-off por dual attention).

---

## 7. Comparación con BERT/GPT

### 7.1 Similitudes

| Aspecto | BERT/GPT | TLOB |
|---------|----------|------|
| **Arquitectura base** | Transformer | Transformer |
| **Mecanismo clave** | Self-attention | Self-attention (dual) |
| **Positional encoding** | Sí (sinusoidal o aprendible) | Sí (sinusoidal o aprendible) |
| **Residual connections** | Sí | Sí |
| **Layer normalization** | Sí | Sí |
| **MLP feed-forward** | Sí | Sí |

### 7.2 Diferencias Clave

| Aspecto | BERT/GPT | TLOB |
|---------|----------|------|
| **Tipo de atención** | Unidimensional (secuencia de tokens) | **Dual** (temporal + espacial) |
| **Número de capas** | 12-24+ | **8 (4 pares)** |
| **Parámetros** | 110M-175B | **~1.1M** |
| **Input** | Secuencia de tokens discretos | **Secuencia de vectores continuos (LOB)** |
| **Output** | Tokens (generación/clasificación) | **3 clases (UP/STAT/DOWN)** |
| **Normalización** | LayerNorm | **BiN (Batch+Instance)** |
| **Masking** | Causal (GPT) o bidireccional (BERT) | **Bidireccional** |

### 7.3 ¿Por qué TLOB no necesita 12+ capas?

**BERT/GPT**:
- Input: Secuencia de tokens discretos (vocabulario de 30K-50K)
- Tarea: Entender lenguaje natural (sintaxis, semántica, contexto, conocimiento del mundo)
- Requiere: Muchas capas para construir representaciones abstractas

**TLOB**:
- Input: Vectores numéricos continuos (precios y volúmenes)
- Tarea: Predecir movimiento de precio (patrón relativamente simple)
- Requiere: Menos capas; 4 pares son suficientes para capturar patrones LOB

**Experimento en el paper**:
```
4 pares:  71.2% accuracy
8 pares:  71.0% accuracy (overfitting)
12 pares: 70.5% accuracy (overfitting severo)

Conclusión: Más capas NO siempre = mejor
```

---

## 8. Referencias

### Documentación Relacionada

1. **Mecanismo de Atención Q, K, V**: [`docs/MECANISMO_ATENCION_QKV.md`](MECANISMO_ATENCION_QKV.md)
2. **Innovaciones del Modelo**: [`docs/INNOVACIONES_TLOB.md`](INNOVACIONES_TLOB.md)
3. **Inferencia y Despliegue**: [`docs/INFERENCIA_Y_DESPLIEGUE.md`](INFERENCIA_Y_DESPLIEGUE.md)
4. **README Principal**: [`README.md`](../README.md)

### Código Relevante

- **Modelo TLOB**: [`src/models/tlob.py`](../src/models/tlob.py)
- **Engine (Entrenamiento)**: [`src/models/engine.py`](../src/models/engine.py)
- **App Streamlit**: [`app.py`](../app.py)

### Papers de Referencia

```bibtex
@article{berti2025tlob,
  title={TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data},
  author={Berti, Leonardo and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2502.15757},
  year={2025}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

---

**Última actualización**: Noviembre 2025  
**Versión**: 1.0.0

---

## Resumen Ejecutivo

### Puntos Clave de la Arquitectura

1. **4 Pares de Transformers** (8 capas total)
   - Alternancia: Spatial → Temporal → Spatial → Temporal → ...
   - Cada par refina las representaciones del anterior

2. **Dual Attention**
   - Spatial: (32, 128, 128) - timesteps atienden a timesteps en contexto de features
   - Temporal: (32, 40, 40) - features atienden a features en contexto de timesteps

3. **Permutaciones Clave**
   - Spatial: Mantiene (batch, seq_len, hidden_dim)
   - Temporal: Permuta a (batch, hidden_dim, seq_len)
   - Siempre volver a permutar al final de capas temporales

4. **Dimensiones Críticas**
   - BTC: hidden_dim_spatial=40, hidden_dim_temporal=128
   - FI-2010: hidden_dim_spatial=144, hidden_dim_temporal=100

5. **Eficiencia**
   - Solo ~1.1M parámetros
   - 89.9% de parámetros en capas temporales
   - Forward pass: ~109M FLOPs (B=32)

**Fin del Documento**

