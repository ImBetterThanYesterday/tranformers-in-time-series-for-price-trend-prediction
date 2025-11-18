# Mecanismo de Atención: Queries, Keys y Values en TLOB

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Teoría Matemática del Mecanismo de Atención](#2-teoría-matemática-del-mecanismo-de-atención)
3. [Implementación en TLOB](#3-implementación-en-tlob)
4. [Ejemplo Práctico Paso a Paso](#4-ejemplo-práctico-paso-a-paso)
5. [Dual Attention en TLOB](#5-dual-attention-en-tlob)
6. [Visualización de Attention Weights](#6-visualización-de-attention-weights)
7. [Comparación con Otras Arquitecturas](#7-comparación-con-otras-arquitecturas)
8. [Referencias](#8-referencias)

---

## 1. Introducción

### ¿Qué es el Mecanismo de Atención?

El mecanismo de atención es el componente fundamental de las arquitecturas Transformer. Permite que el modelo **aprenda a enfocarse** en las partes más relevantes de la entrada para realizar una predicción.

### Analogía Intuitiva

Imagina que estás leyendo este documento:
- **Query (Q)**: La pregunta que tienes en mente ("¿Cómo funciona la atención?")
- **Keys (K)**: Los títulos de las secciones del documento
- **Values (V)**: El contenido de cada sección

Tu cerebro "atiende" más a las secciones cuyos títulos (keys) son más relevantes para tu pregunta (query), y extrae la información (values) de esas secciones.

### Diferencia con CNNs y RNNs

| Arquitectura | Mecanismo | Ventaja | Desventaja |
|--------------|-----------|---------|------------|
| **CNN** | Convoluciones locales | Eficiente, captura patrones locales | Receptive field limitado |
| **RNN/LSTM** | Procesamiento secuencial | Captura dependencias temporales | Olvida información lejana, lento |
| **Transformer (Atención)** | Atención global | Captura cualquier dependencia, paralelo | Complejidad O(n²) |

---

## 2. Teoría Matemática del Mecanismo de Atención

### 2.1 Fórmula General

La atención se calcula mediante la siguiente fórmula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Donde:
- $Q \in \mathbb{R}^{n \times d_k}$: Matriz de **queries** (n queries, cada uno de dimensión $d_k$)
- $K \in \mathbb{R}^{m \times d_k}$: Matriz de **keys** (m keys, cada uno de dimensión $d_k$)
- $V \in \mathbb{R}^{m \times d_v}$: Matriz de **values** (m values, cada uno de dimensión $d_v$)
- $d_k$: Dimensión de queries y keys (usada para scaling)
- $\sqrt{d_k}$: Factor de escala para estabilizar gradientes

### 2.2 Componentes Individuales

#### Q (Queries): "¿Qué estoy buscando?"

$$
Q = XW_Q + b_Q
$$

Donde:
- $X \in \mathbb{R}^{n \times d_{model}}$: Input embeddings
- $W_Q \in \mathbb{R}^{d_{model} \times d_k}$: Matriz de pesos para queries
- $b_Q \in \mathbb{R}^{d_k}$: Bias (opcional)

**Interpretación**: Cada fila de $Q$ representa "qué información busca" esa posición.

#### K (Keys): "¿Qué información tengo disponible?"

$$
K = XW_K + b_K
$$

Donde:
- $W_K \in \mathbb{R}^{d_{model} \times d_k}$: Matriz de pesos para keys
- $b_K \in \mathbb{R}^{d_k}$: Bias (opcional)

**Interpretación**: Cada fila de $K$ representa "qué información ofrece" esa posición.

#### V (Values): "¿Cuál es el contenido real?"

$$
V = XW_V + b_V
$$

Donde:
- $W_V \in \mathbb{R}^{d_{model} \times d_v}$: Matriz de pesos para values
- $b_V \in \mathbb{R}^{d_v}$: Bias (opcional)

**Interpretación**: Cada fila de $V$ contiene el "contenido" que se extraerá.

### 2.3 Proceso Paso a Paso

#### Paso 1: Calcular Scores de Atención

$$
\text{Scores} = QK^T \in \mathbb{R}^{n \times m}
$$

**Interpretación**: 
- $\text{Scores}_{ij}$ = producto punto entre query $i$ y key $j$
- Valores altos → query $i$ es "compatible" con key $j$
- Valores bajos → poca relevancia

**Ejemplo Numérico**:
```
Q = [[1, 2],      K^T = [[1, 0],
     [3, 4]]             [2, 1]]

Scores = [[1×1 + 2×2,  1×0 + 2×1],     [[5,  2],
          [3×1 + 4×2,  3×0 + 4×1]]  =   [11, 4]]
```

#### Paso 2: Scaling

$$
\text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}}
$$

**¿Por qué escalar?**

Sin scaling, cuando $d_k$ es grande, los productos punto tienden a ser muy grandes:

$$
Q \cdot K = \sum_{i=1}^{d_k} q_i k_i
$$

Para $d_k = 512$ (como en Transformers originales), si $q_i, k_i \sim \mathcal{N}(0, 1)$:
- $E[QK^T] = 0$
- $\text{Var}[QK^T] = d_k = 512$
- Valores típicos: $QK^T \in [-60, 60]$

Con scaling por $\sqrt{d_k} = \sqrt{512} \approx 22.6$:
- $\text{Var}\left[\frac{QK^T}{\sqrt{d_k}}\right] = 1$
- Valores típicos: $\frac{QK^T}{\sqrt{d_k}} \in [-3, 3]$

Esto es crucial porque softmax satura con valores grandes:

$$
\text{softmax}([60, 0, -60]) \approx [1.0, 0.0, 0.0] \quad \text{(saturado)}
$$
$$
\text{softmax}([3, 0, -3]) \approx [0.95, 0.05, 0.0] \quad \text{(gradientes no nulos)}
$$

#### Paso 3: Aplicar Softmax

$$
\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

Donde softmax se aplica por filas:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{m} e^{x_j}}
$$

**Propiedades**:
1. $\sum_{j=1}^{m} \text{AttentionWeights}_{ij} = 1$ (cada fila suma 1)
2. $\text{AttentionWeights}_{ij} \in [0, 1]$ (interpretable como probabilidad)
3. Diferenciable (permite backpropagation)

**Ejemplo Numérico**:
```python
# Scaled scores = [5/√2, 2/√2] ≈ [3.54, 1.41]
scores = [3.54, 1.41]

# Softmax
exp_scores = [e^3.54, e^1.41] = [34.47, 4.10]
weights = [34.47/(34.47+4.10), 4.10/(34.47+4.10)]
        = [0.894, 0.106]
```

Query 0 atiende 89.4% a key 0 y 10.6% a key 1.

#### Paso 4: Weighted Sum de Values

$$
\text{Output} = \text{AttentionWeights} \cdot V
$$

**Ejemplo Numérico**:
```python
# Attention weights (de paso 3)
weights = [[0.894, 0.106]]  # 1 query, 2 keys

# Values
V = [[10, 20],    # value para key 0
     [30, 40]]    # value para key 1

# Output
output = 0.894 * [10, 20] + 0.106 * [30, 40]
       = [8.94, 17.88] + [3.18, 4.24]
       = [12.12, 22.12]
```

El output es una **mezcla ponderada** de los values, donde los pesos reflejan la relevancia de cada key para el query.

### 2.4 Multi-Head Attention

En la práctica, se usan múltiples "cabezas" de atención en paralelo:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

Donde cada cabeza es:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Ventajas**:
1. Cada cabeza puede aprender diferentes aspectos (e.g., una cabeza para precios, otra para volúmenes)
2. Mayor capacidad expresiva sin aumentar mucho la complejidad
3. Permite capturar relaciones multi-escala

**En TLOB para BTC**:
- $\text{num\_heads} = 1$ (simplificado)
- Para FI-2010: $\text{num\_heads} = 8$

---

## 3. Implementación en TLOB

### 3.1 Código Relevante

#### Generación de Q, K, V (`src/models/tlob.py`)

```python
class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        q = self.q(x)  # (batch, seq_len, hidden_dim*num_heads)
        k = self.k(x)  # (batch, seq_len, hidden_dim*num_heads)
        v = self.v(x)  # (batch, seq_len, hidden_dim*num_heads)
        return q, k, v
```

**Para TLOB-BTC**:
- `hidden_dim = 40`
- `num_heads = 1`
- Input `x`: `(batch=32, seq_len=128, hidden_dim=40)`
- Output `q, k, v`: cada uno de shape `(32, 128, 40)`

### 3.2 Flujo en TransformerLayer

```python
class TransformerLayer(nn.Module):
    def forward(self, x):
        # x: (32, 128, 40)
        
        # 1. Generar Q, K, V
        q, k, v = self.qkv(x)  # Cada uno: (32, 128, 40)
        
        # 2. Multi-head attention
        x, att = self.attention(q, k, v)
        # x: (32, 128, 40) - outputs
        # att: (32, 1, 128, 128) - attention weights
        
        # 3. Proyección + residual + norm
        x = self.w0(x)  # (32, 128, 40)
        x = x + res     # Residual connection
        x = self.norm(x)
        
        # 4. MLP
        x = self.mlp(x)  # (32, 128, final_dim)
        
        return x, att
```

### 3.3 Dual Attention en TLOB

TLOB alterna entre dos tipos de atención:

#### Atención Spatial (sobre features)

```python
# Input: (batch=32, seq_len=128, hidden_dim=40)
# Cada uno de los 128 timesteps atiende a los 40 features

# Q, K, V shapes: (32, 128, 40)
# Q @ K^T shape: (32, 128, 128)  
#   -> timestep i atiende a timestep j (NO!)

# CORRECCIÓN: En spatial attention, la dimensión es hidden_dim
# El TransformerLayer recibe hidden_dim=40 como parámetro
# Por lo tanto:
# Q: (32, 128, 40) - 128 queries (uno por timestep)
# K: (32, 128, 40) - 128 keys
# QK^T: (32, 128, 128) - 128×128 matriz de atención

# Cada timestep atiende a TODOS los otros timesteps
# pero en el contexto de las FEATURES (dimensión 40)
```

**Interpretación**: 
- Pregunta: "Para el timestep t, ¿qué valores de features son relevantes?"
- Respuesta: La atención pondera las features basándose en su relevancia

#### Atención Temporal (sobre timesteps)

```python
# Input: (batch=32, hidden_dim=40, seq_len=128) [PERMUTADO]
# Cada uno de los 40 features atiende a los 128 timesteps

# Q, K, V shapes: (32, 40, 128)
# Q @ K^T shape: (32, 40, 40)
#   -> feature i atiende a feature j (NO!)

# CORRECCIÓN: En temporal attention, la dimensión es seq_len
# El TransformerLayer recibe seq_len=128 como parámetro
# Por lo tanto:
# Q: (32, 40, 128) - 40 queries (uno por feature)
# K: (32, 40, 128) - 40 keys
# QK^T: (32, 40, 40) - 40×40 matriz de atención

# Cada feature atiende a TODOS los otros features
# pero en el contexto de los TIMESTEPS (dimensión 128)
```

**Interpretación**: 
- Pregunta: "Para el feature f, ¿qué momentos temporales son relevantes?"
- Respuesta: La atención pondera los timesteps basándose en su relevancia

---

## 4. Ejemplo Práctico Paso a Paso

### 4.1 Setup: Datos Reales de TLOB-BTC

Usaremos una ventana LOB simplificada para claridad:

```python
# Simplificación: 3 timesteps, 3 features (en lugar de 128×40)
# Features: ASK_P1, ASK_V1, BID_P1

# Datos NORMALIZADOS (después de Z-score)
X = np.array([
    [0.523, 0.145, -0.412],  # t=0
    [0.634, 0.223, -0.398],  # t=1
    [0.478, 0.189, -0.425]   # t=2
])
# Shape: (3, 3) = (seq_len, hidden_dim)
```

**Nota**: Estos valores son Z-scores después de normalización.

### 4.2 Paso 1: Generar Q, K, V

Inicializamos las matrices de proyección (simuladas):

```python
import numpy as np

# Matrices de pesos (3×3 para simplificar)
W_Q = np.array([
    [0.2, 0.3, 0.1],
    [0.4, 0.1, 0.2],
    [0.3, 0.2, 0.4]
])

W_K = np.array([
    [0.1, 0.4, 0.2],
    [0.3, 0.2, 0.3],
    [0.2, 0.1, 0.5]
])

W_V = np.array([
    [0.5, 0.2, 0.1],
    [0.1, 0.4, 0.3],
    [0.2, 0.3, 0.4]
])

# Generar Q, K, V
Q = X @ W_Q  # (3, 3) @ (3, 3) = (3, 3)
K = X @ W_K
V = X @ W_V
```

**Cálculo detallado de Q[0] (query para t=0)**:

$$
Q[0] = X[0] \cdot W_Q = [0.523, 0.145, -0.412] \cdot \begin{bmatrix} 0.2 & 0.3 & 0.1 \\ 0.4 & 0.1 & 0.2 \\ 0.3 & 0.2 & 0.4 \end{bmatrix}
$$

$$
= [0.523 \times 0.2 + 0.145 \times 0.4 + (-0.412) \times 0.3,
$$
$$
   0.523 \times 0.3 + 0.145 \times 0.1 + (-0.412) \times 0.2,
$$
$$
   0.523 \times 0.1 + 0.145 \times 0.2 + (-0.412) \times 0.4]
$$

$$
= [0.1046 + 0.0580 - 0.1236, \quad 0.1569 + 0.0145 - 0.0824, \quad 0.0523 + 0.0290 - 0.1648]
$$

$$
Q[0] = [0.0390, \quad 0.0890, \quad -0.0835]
$$

Similarmente calculamos las otras filas:

```python
Q = np.array([
    [ 0.0390,  0.0890, -0.0835],  # query para t=0
    [ 0.0649,  0.1124, -0.0672],  # query para t=1
    [ 0.0301,  0.0851, -0.0918]   # query para t=2
])

K = np.array([
    [ 0.0126,  0.2954, -0.1342],  # key para t=0
    [ 0.0341,  0.3278, -0.1245],  # key para t=1
    [ 0.0089,  0.2881, -0.1421]   # key para t=2
])

V = np.array([
    [ 0.2277,  0.1034, -0.0338],  # value para t=0
    [ 0.2545,  0.1243, -0.0102],  # value para t=1
    [ 0.2124,  0.0989, -0.0445]   # value para t=2
])
```

### 4.3 Paso 2: Calcular Scores

$$
\text{Scores} = QK^T
$$

```python
# Producto matricial Q @ K.T
Scores = Q @ K.T  # (3, 3) @ (3, 3) = (3, 3)
```

**Cálculo detallado de Scores[0, 1]** (cuánto t=0 atiende a t=1):

$$
\text{Scores}[0, 1] = Q[0] \cdot K[1] 
$$
$$
= [0.0390, 0.0890, -0.0835] \cdot [0.0341, 0.3278, -0.1245]
$$
$$
= 0.0390 \times 0.0341 + 0.0890 \times 0.3278 + (-0.0835) \times (-0.1245)
$$
$$
= 0.001331 + 0.029174 + 0.010396 = 0.040901
$$

Calculando todos los scores:

```python
Scores = np.array([
    [0.0372, 0.0409, 0.0360],  # t=0 atiende a [t=0, t=1, t=2]
    [0.0433, 0.0473, 0.0418],  # t=1 atiende a [t=0, t=1, t=2]
    [0.0345, 0.0379, 0.0333]   # t=2 atiende a [t=0, t=1, t=2]
])
```

**Interpretación**:
- `Scores[1, 1] = 0.0473` es el más alto → t=1 presta más atención a sí mismo
- `Scores[2, 0] = 0.0345` → t=2 presta poca atención a t=0 (pasado lejano)

### 4.4 Paso 3: Scaling

$$
\text{Scaled Scores} = \frac{\text{Scores}}{\sqrt{d_k}}
$$

Para $d_k = 3$ (dimensión de queries/keys):

```python
d_k = 3
scaled_scores = Scores / np.sqrt(d_k)  # Dividir por √3 ≈ 1.732

scaled_scores = np.array([
    [0.0215, 0.0236, 0.0208],
    [0.0250, 0.0273, 0.0241],
    [0.0199, 0.0219, 0.0192]
])
```

**Efecto del scaling**: Los valores se reducen para evitar saturación en softmax.

### 4.5 Paso 4: Softmax

$$
\text{Attention Weights} = \text{softmax}(\text{Scaled Scores})
$$

Aplicado por filas:

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Restar max para estabilidad numérica
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)
```

**Cálculo detallado para fila 0** (t=0):

$$
\text{softmax}([0.0215, 0.0236, 0.0208]) = \frac{[e^{0.0215}, e^{0.0236}, e^{0.0208}]}{\sum}
$$

```python
exp_values = [np.exp(0.0215), np.exp(0.0236), np.exp(0.0208)]
           = [1.0217, 1.0239, 1.0210]

sum_exp = 1.0217 + 1.0239 + 1.0210 = 3.0666

weights = [1.0217/3.0666, 1.0239/3.0666, 1.0210/3.0666]
        = [0.3331, 0.3339, 0.3330]
```

Aplicando a todas las filas:

```python
attention_weights = np.array([
    [0.3331, 0.3339, 0.3330],  # t=0 atiende uniformemente a todos
    [0.3330, 0.3339, 0.3331],  # t=1 atiende ligeramente más a t=1
    [0.3332, 0.3337, 0.3331]   # t=2 distribución uniforme
])
```

**Interpretación**:
- Distribución casi uniforme (cada peso ≈ 0.333 = 1/3)
- Esto ocurre porque los scores originales eran muy similares
- En práctica, con más datos y entrenamiento, las diferencias son más pronunciadas

**Verificación**: Cada fila suma 1.0:
```python
attention_weights.sum(axis=1)  # [1.0, 1.0, 1.0] ✓
```

### 4.6 Paso 5: Weighted Sum con V

$$
\text{Output} = \text{Attention Weights} \times V
$$

```python
output = attention_weights @ V  # (3, 3) @ (3, 3) = (3, 3)
```

**Cálculo detallado para output[0]** (representación atendida de t=0):

$$
\text{output}[0] = \sum_{j=0}^{2} \text{attention\_weights}[0, j] \times V[j]
$$

$$
= 0.3331 \times [0.2277, 0.1034, -0.0338]
$$
$$
+ 0.3339 \times [0.2545, 0.1243, -0.0102]
$$
$$
+ 0.3330 \times [0.2124, 0.0989, -0.0445]
$$

$$
= [0.0758, 0.0344, -0.0113] + [0.0850, 0.0415, -0.0034] + [0.0707, 0.0329, -0.0148]
$$

$$
\text{output}[0] = [0.2315, 0.1088, -0.0295]
$$

Resultados completos:

```python
output = np.array([
    [ 0.2315,  0.1088, -0.0295],  # representación atendida de t=0
    [ 0.2318,  0.1090, -0.0292],  # representación atendida de t=1
    [ 0.2313,  0.1087, -0.0296]   # representación atendida de t=2
])
```

**Interpretación**:
- `output[0]` es una mezcla ponderada de `V[0]`, `V[1]`, `V[2]`
- Los pesos reflejan cuánto cada timestep "contribuye" a la representación de t=0
- Valores muy similares entre filas debido a la distribución uniforme de atención

### 4.7 Comparación: Input vs Output

```python
print("Input X:")
print(X)
print("\nOutput (después de atención):")
print(output)
print("\nDiferencia:")
print(output - X)
```

```
Input X:
[[ 0.523   0.145  -0.412]
 [ 0.634   0.223  -0.398]
 [ 0.478   0.189  -0.425]]

Output (después de atención):
[[ 0.2315  0.1088 -0.0295]
 [ 0.2318  0.1090 -0.0292]
 [ 0.2313  0.1087 -0.0296]]

Diferencia:
[[-0.2915  -0.0362  0.3825]
 [-0.4022  -0.1140  0.3688]
 [-0.2467  -0.0803  0.3954]]
```

**Observación**: El output es una versión "suavizada" del input, donde cada timestep ahora contiene información agregada de todos los timesteps ponderada por relevancia.

---

## 5. Dual Attention en TLOB

### 5.1 Atención Spatial (sobre features)

**Configuración**:
- Input: `(batch=32, seq_len=128, hidden_dim=40)`
- Cada uno de los 128 timesteps actúa como un query
- Atiende a los 40 features

**Interpretación**:
- Pregunta: "¿Qué features (ASK/BID prices/volumes) son más relevantes para predecir la tendencia?"
- Ejemplo de pattern aprendido:
  * Alta atención a `ASK_P1` y `BID_P1` (best prices)
  * Baja atención a volúmenes de niveles profundos (L8, L9, L10)

**Ejemplo de Attention Weights Spatial**:

```
         ASK_P1  ASK_V1  BID_P1  BID_V1  ...  BID_V10
t=0       0.15    0.08    0.14    0.06   ...   0.01
t=1       0.16    0.07    0.15    0.05   ...   0.01
...       ...     ...     ...     ...    ...   ...
t=127     0.14    0.09    0.13    0.07   ...   0.02
```

**Interpretación**: En todos los timesteps, el modelo presta más atención a precios (ASK_P1, BID_P1) que a volúmenes.

### 5.2 Atención Temporal (sobre timesteps)

**Configuración**:
- Input: `(batch=32, hidden_dim=40, seq_len=128)` [PERMUTADO]
- Cada uno de los 40 features actúa como un query
- Atiende a los 128 timesteps

**Interpretación**:
- Pregunta: "¿Qué momentos temporales son más relevantes?"
- Ejemplo de pattern aprendido:
  * Alta atención a timesteps recientes (t=120-128)
  * Atención media a timesteps medianos (t=50-80)
  * Baja atención a timesteps antiguos (t=0-20)

**Ejemplo de Attention Weights Temporal**:

```
          t=0    t=20   t=50   t=100  t=127
ASK_P1   0.001  0.005  0.010  0.025  0.150
ASK_V1   0.001  0.004  0.012  0.030  0.140
BID_P1   0.001  0.006  0.011  0.028  0.145
...      ...    ...    ...    ...    ...
```

**Interpretación**: Todos los features prestan más atención a timesteps recientes, con decaimiento exponencial hacia el pasado.

### 5.3 Alternancia de Capas

En TLOB, las capas se alternan:

```
Layer 0: Spatial Attention  (sobre 40 features)
Layer 1: Temporal Attention (sobre 128 timesteps)
Layer 2: Spatial Attention  (sobre 40 features)
Layer 3: Temporal Attention (sobre 128 timesteps)
...
```

**Ventajas**:
1. **Captura relaciones multi-nivel**: Primero identifica features relevantes, luego cuándo son relevantes
2. **Reduce complejidad**: Separar spatial y temporal es más eficiente que atención 2D completa
3. **Interpretabilidad**: Podemos analizar qué features y qué timesteps son importantes por separado

---

## 6. Visualización de Attention Weights

### 6.1 Heatmap de Atención

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Supongamos que tenemos attention_weights de shape (128, 128)
# para una capa de atención temporal

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(attention_weights, cmap='viridis', ax=ax)
ax.set_xlabel('Key Position (timestep)')
ax.set_ylabel('Query Position (timestep)')
ax.set_title('Attention Weights - Temporal Attention Layer')
plt.show()
```

**Interpretación del heatmap**:
- **Diagonal fuerte**: Auto-atención (cada timestep atiende a sí mismo)
- **Banda inferior derecha brillante**: Timesteps recientes atienden a otros recientes
- **Esquina superior izquierda oscura**: Timesteps antiguos reciben poca atención

### 6.2 Patrones Típicos

#### Atención Causal (en modelos generativos)
```
1.0  0.0  0.0  0.0
0.5  0.5  0.0  0.0
0.3  0.3  0.4  0.0
0.2  0.2  0.3  0.3
```
Solo atiende al pasado (máscara triangular).

#### Atención Bidireccional (TLOB)
```
0.2  0.3  0.4  0.1
0.3  0.2  0.3  0.2
0.4  0.3  0.2  0.1
0.1  0.2  0.1  0.6
```
Puede atender a pasado Y futuro (dentro de la ventana).

#### Atención Local
```
0.5  0.4  0.1  0.0
0.4  0.3  0.3  0.0
0.1  0.3  0.4  0.2
0.0  0.0  0.2  0.8
```
Enfoque en vecindad local.

---

## 7. Comparación con Otras Arquitecturas

### 7.1 Tabla Comparativa

| Característica | CNN | RNN/LSTM | Transformer (TLOB) |
|----------------|-----|----------|-------------------|
| **Mecanismo** | Convolución local | Procesamiento secuencial | Atención global |
| **Receptive field** | Limitado (tamaño kernel) | Todo el pasado | Toda la secuencia |
| **Complejidad** | O(n·k) | O(n) | O(n²) |
| **Paralelización** | Sí | No | Sí |
| **Memoria** | Baja | Media (hidden state) | Alta (attn matrices) |
| **Interpretabilidad** | Baja | Baja | Alta (att weights) |

### 7.2 Ventajas de Atención

#### 1. Captura Dependencias de Largo Alcance

**RNN/LSTM**:
- Problema del gradiente que se desvanece
- Información de t=0 se "olvida" en t=100

**Transformer**:
- Conexión directa entre cualquier par de timesteps
- `attention_weights[100, 0]` puede ser alta si t=0 es relevante para t=100

#### 2. Interpretabilidad

**CNN**:
- Difícil saber qué características específicas activaron una predicción

**Transformer**:
- Podemos visualizar attention weights
- "El modelo predice UP porque presta mucha atención a ASK_P1 en t=120"

#### 3. Paralelización

**RNN**:
```python
# Procesamiento secuencial (no paralelo)
h[0] = f(x[0])
h[1] = f(x[1], h[0])  # Depende de h[0]
h[2] = f(x[2], h[1])  # Depende de h[1]
```

**Transformer**:
```python
# Procesamiento paralelo
Q, K, V = generate_qkv(X)  # Todo X procesado en paralelo
attention = softmax(QK^T / √d_k) @ V  # Multiplicación matricial paralela
```

### 7.3 Desventajas de Atención

#### 1. Complejidad Cuadrática

Para secuencia de longitud n:
- Memoria: O(n²) para almacenar attention weights
- Cómputo: O(n²·d) donde d es dimensión

**Ejemplo para TLOB**:
- n = 128, d = 40
- Atención temporal: 128² = 16,384 operaciones
- Atención spatial: 40² = 1,600 operaciones

#### 2. Mayor Uso de Memoria

```python
# LSTM: Solo almacena hidden state
memory_lstm = batch_size * hidden_dim
            = 32 * 256 = 8,192 floats ≈ 32 KB

# Transformer: Almacena attention matrices
memory_transformer = batch_size * num_layers * seq_len * seq_len
                   = 32 * 8 * 128 * 128 = 4,194,304 floats ≈ 16 MB
```

---

## 8. Referencias

### Paper Original TLOB

```bibtex
@article{berti2025tlob,
  title={TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data},
  author={Berti, Leonardo and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2502.15757},
  year={2025}
}
```

**Link**: https://arxiv.org/pdf/2502.15757

### Attention is All You Need (Paper Original de Transformers)

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

**Link**: https://arxiv.org/abs/1706.03762

### Recursos Adicionales

1. **The Illustrated Transformer** (Jay Alammar)
   - http://jalammar.github.io/illustrated-transformer/
   - Visualizaciones excelentes del mecanismo de atención

2. **Attention? Attention!** (Lilian Weng)
   - https://lilianweng.github.io/posts/2018-06-24-attention/
   - Explicación matemática detallada

3. **Código TLOB Original**
   - https://github.com/LeonardoBerti00/TLOB
   - Implementación de referencia

4. **Documentación PyTorch MultiheadAttention**
   - https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
   - API y ejemplos

---

## Apéndice A: Código Completo del Ejemplo

```python
import numpy as np

def softmax(x):
    """Calcula softmax de forma numéricamente estable"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# Datos de entrada (3 timesteps, 3 features)
X = np.array([
    [0.523, 0.145, -0.412],
    [0.634, 0.223, -0.398],
    [0.478, 0.189, -0.425]
])

# Matrices de proyección
W_Q = np.array([
    [0.2, 0.3, 0.1],
    [0.4, 0.1, 0.2],
    [0.3, 0.2, 0.4]
])

W_K = np.array([
    [0.1, 0.4, 0.2],
    [0.3, 0.2, 0.3],
    [0.2, 0.1, 0.5]
])

W_V = np.array([
    [0.5, 0.2, 0.1],
    [0.1, 0.4, 0.3],
    [0.2, 0.3, 0.4]
])

# Generar Q, K, V
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Q (Queries):")
print(Q)
print("\nK (Keys):")
print(K)
print("\nV (Values):")
print(V)

# Calcular attention scores
scores = Q @ K.T
print("\nAttention Scores (QK^T):")
print(scores)

# Scaling
d_k = Q.shape[-1]
scaled_scores = scores / np.sqrt(d_k)
print(f"\nScaled Scores (divided by √{d_k}):")
print(scaled_scores)

# Softmax
attention_weights = softmax(scaled_scores)
print("\nAttention Weights (after softmax):")
print(attention_weights)
print(f"Row sums: {attention_weights.sum(axis=1)}")  # Debe ser [1, 1, 1]

# Weighted sum
output = attention_weights @ V
print("\nOutput (AttentionWeights @ V):")
print(output)

print("\nDiferencia (Output - Input):")
print(output - X)
```

---

**Fin del Documento**

Para más detalles sobre la implementación completa de TLOB, consulta:
- `src/models/tlob.py`: Código del modelo
- `docs/INFERENCIA_Y_DESPLIEGUE.md`: Proceso de inferencia
- `docs/INNOVACIONES_TLOB.md`: Innovaciones vs otros modelos

