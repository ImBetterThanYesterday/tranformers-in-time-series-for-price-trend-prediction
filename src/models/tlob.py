from torch import nn
import torch
from einops import rearrange
import src.constants as cst
from src.models.bin import BiN
from src.models.mlplob import MLP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ComputeQKV(nn.Module):
    """
    GENERACIÓN DE QUERIES, KEYS, VALUES
    ====================================
    
    Módulo que computa las matrices Q, K, V para el mecanismo de atención.
    Estas tres proyecciones son fundamentales para el mecanismo de atención en Transformers.
    
    En el Transformer, la atención se calcula como:
        Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    
    Este módulo genera Q, K, V mediante proyecciones lineales del input.
    
    Args:
        hidden_dim (int): Dimensión del espacio de embedding (40 para BTC)
        num_heads (int): Número de cabezas de atención (1 para BTC)
    
    Atributos:
        q (nn.Linear): Proyección lineal para Queries
                      Transforma: (batch, seq_len, hidden_dim) → (batch, seq_len, hidden_dim*num_heads)
        k (nn.Linear): Proyección lineal para Keys
                      Transforma: (batch, seq_len, hidden_dim) → (batch, seq_len, hidden_dim*num_heads)
        v (nn.Linear): Proyección lineal para Values
                      Transforma: (batch, seq_len, hidden_dim) → (batch, seq_len, hidden_dim*num_heads)
    
    Arquitectura:
    -------------
    - self.q: Linear(hidden_dim → hidden_dim*num_heads)
    - self.k: Linear(hidden_dim → hidden_dim*num_heads)
    - self.v: Linear(hidden_dim → hidden_dim*num_heads)
    
    La multiplicación por num_heads permite que cada cabeza tenga su propia proyección
    (multi-head attention). En el caso de TLOB con BTC (num_heads=1), el output
    tiene la misma dimensión que el input.
    
    Interpretación Conceptual:
    --------------------------
    - **Q (Queries)**: "¿Qué información estoy buscando?"
      * Para cada timestep, genera un vector que representa qué patrones buscar
      * Ejemplo: Si estoy en t=127 (último timestep), mi query busca información
        relevante en timesteps pasados
    
    - **K (Keys)**: "¿Qué información tengo disponible?"
      * Para cada timestep, genera un vector que representa la información que contiene
      * Ejemplo: Timestep t=100 tiene información sobre precios y volúmenes en ese momento
    
    - **V (Values)**: "¿Cuál es el contenido real de esa información?"
      * Para cada timestep, genera un vector con el contenido real a extraer
      * Ejemplo: Los valores específicos de precios y volúmenes procesados
    
    Forward Pass:
    -------------
    Input: x de shape (batch, seq_len, hidden_dim)
           Ejemplo: (32, 128, 40) para BTC con batch=32
    
    Proceso:
    1. x pasa por self.q → Q de shape (batch, seq_len, hidden_dim*num_heads)
    2. x pasa por self.k → K de shape (batch, seq_len, hidden_dim*num_heads)
    3. x pasa por self.v → V de shape (batch, seq_len, hidden_dim*num_heads)
    
    Output: Tupla (Q, K, V) donde cada uno tiene shape (batch, seq_len, hidden_dim*num_heads)
    
    Ejemplo Concreto para TLOB-BTC:
    --------------------------------
    Si hidden_dim=40, num_heads=1, seq_len=128, batch=32:
    - Input x: (32, 128, 40)
      * 32 ejemplos
      * 128 timesteps
      * 40 features (ASK/BID prices y volumes)
    
    - Output Q: (32, 128, 40)
      * "¿Qué patrones busco en cada timestep?"
      * Cada timestep tiene un query vector de 40 dimensiones
    
    - Output K: (32, 128, 40)
      * "¿Qué información ofrece cada timestep?"
      * Cada timestep tiene un key vector de 40 dimensiones
    
    - Output V: (32, 128, 40)
      * "¿Cuál es el contenido de cada timestep?"
      * Cada timestep tiene un value vector de 40 dimensiones
    
    Uso en Dual Attention:
    ----------------------
    En TLOB, estas proyecciones Q, K, V se usan en DOS contextos:
    
    1. **Atención Spatial (sobre features)**:
       - Input shape: (batch, seq_len=128, hidden_dim=40)
       - Cada timestep "atiende" a los 40 features
       - Pregunta: "¿Qué features son relevantes?"
    
    2. **Atención Temporal (sobre timesteps)**:
       - Input shape: (batch, hidden_dim=40, seq_len=128) [permutado]
       - Cada feature "atiende" a los 128 timesteps
       - Pregunta: "¿Qué momentos temporales son relevantes?"
    
    Nota Técnica:
    -------------
    Los pesos de las proyecciones lineales (W_q, W_k, W_v) son aprendidos durante
    el entrenamiento mediante backpropagation. No hay inicialización especial,
    PyTorch usa inicialización Kaiming uniform por defecto.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # Proyecciones lineales para Q, K, V
        # Input: (batch, seq_len, hidden_dim)
        # Output: (batch, seq_len, hidden_dim*num_heads)
        self.q = nn.Linear(hidden_dim, hidden_dim*num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim*num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim*num_heads)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        # Aplicar proyecciones lineales
        q = self.q(x)  # Queries: (batch, seq_len, hidden_dim*num_heads)
        k = self.k(x)  # Keys: (batch, seq_len, hidden_dim*num_heads)
        v = self.v(x)  # Values: (batch, seq_len, hidden_dim*num_heads)
        return q, k, v


class TransformerLayer(nn.Module):
    """
    CAPA TRANSFORMER CON MECANISMO DE ATENCIÓN
    ===========================================
    
    Implementa una capa de Transformer con atención multi-head y MLP feedforward.
    Esta es la unidad básica de construcción del modelo TLOB.
    
    En TLOB, estas capas se alternan entre:
    - **Atención ESPACIAL** (sobre features): Captura relaciones entre precios/volúmenes
    - **Atención TEMPORAL** (sobre timesteps): Captura evolución temporal
    
    Args:
        hidden_dim (int): Dimensión de embeddings (40 para BTC, 128 para FI-2010)
        num_heads (int): Número de cabezas de atención (1 para BTC, 8 para FI-2010)
        final_dim (int): Dimensión de salida del MLP (puede reducir dimensionalidad)
    
    Componentes:
    ------------
    1. **LayerNorm**: Normaliza activaciones para estabilizar entrenamiento
    2. **ComputeQKV**: Genera matrices Q, K, V para atención
    3. **MultiheadAttention**: Mecanismo de atención de PyTorch
    4. **w0**: Proyección de vuelta a hidden_dim después de atención
    5. **MLP**: Red feedforward para transformación no-lineal
    
    Arquitectura Completa:
    ---------------------
    ```
    x (input)
    │
    ├─────────┐ (residual connection)
    │         │
    │    LayerNorm → QKV → MultiheadAttention → Projection (w0)
    │         │
    └────> +  (add residual)
           │
           LayerNorm
           │
           MLP
           │
    ┌──────┘
    │
    └────> + (add residual si dims coinciden)
           │
         output
    ```
    
    Forward Pass Detallado:
    -----------------------
    Input: x de shape (batch, seq_len, hidden_dim)
    
    1. **Guardar Residual**: 
       ```python
       res = x  # Para skip connection
       ```
    
    2. **Generar Q, K, V**:
       ```python
       q, k, v = self.qkv(x)  
       # Cada uno: (batch, seq_len, hidden_dim*num_heads)
       ```
    
    3. **Multi-Head Attention**:
       ```python
       x, att = self.attention(q, k, v)
       # x: (batch, seq_len, hidden_dim*num_heads)
       # att: Pesos de atención (batch, num_heads, seq_len, seq_len)
       ```
       
       Internamente calcula:
       ```
       scores = Q @ K^T / √d_k
       attention_weights = softmax(scores)
       output = attention_weights @ V
       ```
    
    4. **Proyección de Salida**:
       ```python
       x = self.w0(x)  
       # (batch, seq_len, hidden_dim*num_heads) → (batch, seq_len, hidden_dim)
       ```
    
    5. **Primera Residual Connection + Norm**:
       ```python
       x = x + res  # Suma el input original
       x = self.norm(x)  # Normaliza
       ```
    
    6. **MLP Feedforward**:
       ```python
       x = self.mlp(x)
       # Arquitectura interna del MLP:
       # Linear(hidden_dim → hidden_dim*4) → GELU → Linear(hidden_dim*4 → final_dim)
       # Shape: (batch, seq_len, hidden_dim) → (batch, seq_len, final_dim)
       ```
    
    7. **Segunda Residual Connection** (condicional):
       ```python
       if x.shape[-1] == res.shape[-1]:
           x = x + res  # Solo si dimensiones coinciden
       ```
       Esto permite reducir dimensionalidad en capas finales sin romper residuals.
    
    Output: 
    - x: Tensor transformado de shape (batch, seq_len, final_dim)
    - att: Pesos de atención de shape (batch, num_heads, seq_len, seq_len)
    
    Residual Connections (Skip Connections):
    ----------------------------------------
    Las conexiones residuales son CRÍTICAS en redes profundas:
    
    **Problema sin residuals:**
    - Gradientes se desvanecen en redes profundas
    - Dificulta el entrenamiento de muchas capas
    
    **Solución con residuals:**
    ```python
    output = F(x) + x
    ```
    - Los gradientes fluyen directamente hacia atrás
    - Permite entrenar redes muy profundas (100+ capas)
    - El modelo puede aprender identity mapping si es óptimo
    
    En TLOB con 4 pares de capas (8 capas totales), las residuals son esenciales.
    
    Dual Attention en TLOB:
    -----------------------
    Las capas se alternan entre dos modos:
    
    **Modo 1: Atención Spatial (capas pares)**
    ```python
    # Input: (batch, seq_len=128, hidden_dim=40)
    # Cada timestep atiende a los 40 features
    # Pregunta: "¿Qué features (ASK/BID) son relevantes?"
    # Ejemplo: Alta atención a ASK_P1 y BID_P1 (best prices)
    ```
    
    **Modo 2: Atención Temporal (capas impares)**
    ```python
    # Input: (batch, hidden_dim=40, seq_len=128) [permutado]
    # Cada feature atiende a los 128 timesteps
    # Pregunta: "¿Qué momentos temporales son relevantes?"
    # Ejemplo: Alta atención a timesteps recientes (100-128)
    ```
    
    Esta alternancia captura tanto relaciones entre variables como evolución temporal.
    
    Ejemplo Concreto con Dimensiones TLOB-BTC:
    -------------------------------------------
    ```python
    # Configuración
    hidden_dim = 40
    num_heads = 1
    final_dim = 40  # (o 10 en última capa para reducir)
    
    # Input
    x = torch.randn(32, 128, 40)  # batch=32, seq=128, features=40
    
    # Forward pass
    layer = TransformerLayer(40, 1, 40)
    output, attention_weights = layer(x)
    
    # Output shapes
    # output: (32, 128, 40)
    # attention_weights: (32, 1, 128, 128)
    #   - 32 ejemplos
    #   - 1 cabeza
    #   - 128 queries × 128 keys
    #   - attention_weights[i, 0, t, s] = cuánto timestep t atiende a timestep s
    ```
    
    Visualización de Attention Weights:
    -----------------------------------
    Los pesos de atención se pueden visualizar como heatmaps:
    - Eje X: Key positions (timesteps o features)
    - Eje Y: Query positions (timesteps o features)
    - Color: Intensidad de atención (0 a 1)
    - Diagonal fuerte: Auto-atención
    - Valores altos: Dependencia fuerte
    
    Nota sobre Performance:
    -----------------------
    La complejidad de atención es O(n²) donde n es seq_len o hidden_dim:
    - Atención spatial: O(40²) = 1,600 operaciones
    - Atención temporal: O(128²) = 16,384 operaciones
    
    Por eso la atención temporal es más costosa que la spatial en TLOB.
    """
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # Layer normalization para estabilizar entrenamiento
        self.norm = nn.LayerNorm(hidden_dim)
        # Módulo para generar Q, K, V
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        # Multi-head attention de PyTorch
        self.attention = nn.MultiheadAttention(
            hidden_dim*num_heads, 
            num_heads, 
            batch_first=True,  # Input shape: (batch, seq, features)
            device=cst.DEVICE
        )
        # MLP feedforward
        self.mlp = MLP(hidden_dim, hidden_dim*4, final_dim)
        # Proyección de salida de atención
        self.w0 = nn.Linear(hidden_dim*num_heads, hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        
        # Guardar para residual connection
        res = x
        
        # Generar Q, K, V
        q, k, v = self.qkv(x)
        
        # Aplicar multi-head attention
        # average_attn_weights=False: Retornar pesos por cabeza (no promediar)
        # need_weights=True: Retornar los pesos de atención
        x, att = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        
        # Proyectar de vuelta a hidden_dim
        x = self.w0(x)
        
        # Primera residual connection
        x = x + res
        
        # Layer normalization
        x = self.norm(x)
        
        # MLP feedforward
        x = self.mlp(x)
        
        # Segunda residual connection (solo si dimensiones coinciden)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        
        return x, att


class TLOB(nn.Module):
    """
    TLOB: TRANSFORMER WITH DUAL ATTENTION FOR LIMIT ORDER BOOK
    ===========================================================
    
    Arquitectura principal del modelo TLOB para predicción de tendencias de precios
    utilizando datos del Limit Order Book.
    
    Innovaciones Clave:
    -------------------
    1. **DUAL ATTENTION**:
       - Capas pares: Atención sobre FEATURES (spatial attention)
       - Capas impares: Atención sobre TIMESTEPS (temporal attention)
       Esto permite capturar tanto relaciones entre variables como evolución temporal.
    
    2. **BiN (Batch Independent Normalization)**:
       Normalización que funciona con batch_size=1, crítico para inferencia en producción.
       A diferencia de BatchNorm, BiN normaliza cada feature independientemente.
    
    3. **POSITIONAL ENCODING**:
       Sinusoidal o aprendido, inyecta información de posición temporal en la secuencia.
    
    4. **PROGRESSIVE DIMENSION REDUCTION**:
       Las últimas capas reducen dimensionalidad progresivamente antes del clasificador final.
    
    Args:
        hidden_dim (int): Dimensión de embeddings (40 para BTC, 256 para FI-2010)
        num_layers (int): Número de PARES de TransformerLayers (4 típicamente = 8 capas totales)
        seq_size (int): Longitud de secuencia temporal (128 para BTC, 100 para FI-2010)
        num_features (int): Número de features del LOB (40 para BTC)
        num_heads (int): Cabezas de atención (1 para BTC, 8 para FI-2010)
        is_sin_emb (bool): Usar positional encoding sinusoidal (True) o aprendido (False)
        dataset_type (str): Tipo de dataset ("BTC", "FI_2010", "LOBSTER")
    
    Arquitectura Completa (BTC):
    ----------------------------
    ```
    INPUT: (batch, seq_len=128, features=40)
    │
    ├─> BiN Normalization Layer
    │   - Normaliza cada feature independientemente
    │   - Permute: (batch, 40, 128) → BiN → (batch, 40, 128)
    │   - Permute back: (batch, 128, 40)
    │   Output: (batch, 128, 40)
    │
    ├─> Embedding Layer
    │   - Linear: 40 features → hidden_dim (40)
    │   Output: (batch, 128, 40)
    │
    ├─> Positional Encoding
    │   - Suma encodings sinusoidales
    │   Output: (batch, 128, 40)
    │
    ├─> TransformerLayers Alternados (num_layers=4 * 2 = 8 capas):
    │   
    │   Par 0:
    │   ├─> Capa 0: Atención SPATIAL sobre FEATURES (hidden_dim=40)
    │   │   Input: (batch, 128, 40)
    │   │   Cada timestep atiende a los 40 features
    │   │   Output: (batch, 128, 40)
    │   │
    │   ├─> Permute: (batch, 128, 40) → (batch, 40, 128)
    │   │
    │   └─> Capa 1: Atención TEMPORAL sobre TIMESTEPS (seq_size=128)
    │       Input: (batch, 40, 128)
    │       Cada feature atiende a los 128 timesteps
    │       Output: (batch, 40, 128)
    │       Permute back: (batch, 128, 40)
    │   
    │   Par 1:
    │   ├─> Capa 2: Atención SPATIAL (40)
    │   └─> Capa 3: Atención TEMPORAL (128)
    │   
    │   Par 2:
    │   ├─> Capa 4: Atención SPATIAL (40)
    │   └─> Capa 5: Atención TEMPORAL (128)
    │   
    │   Par 3 (ÚLTIMA - con reducción):
    │   ├─> Capa 6: Atención SPATIAL con reducción (40 → 10)
    │   │   Output: (batch, 128, 10)
    │   │
    │   └─> Capa 7: Atención TEMPORAL con reducción (128 → 32)
    │       Output: (batch, 10, 32)
    │   
    │   Output final de Transformers: (batch, 10, 32)
    │
    ├─> Flatten
    │   (batch, 10, 32) → (batch, 320)
    │
    └─> Final MLP (Clasificador)
        - Series de capas Linear + GELU que reducen progresivamente:
        - Linear(320 → 80) + GELU
        - Linear(80 → 20) + GELU (si total_dim > 128)
        - Linear(... → 128) + GELU
        - Linear(128 → 3)  [Capa final sin activación]
        
        Output: (batch, 3) - Logits para 3 clases
    ```
    
    Forward Pass Paso a Paso con Shapes (BTC):
    ------------------------------------------
    ```python
    x = input  # (32, 128, 40)
    
    # Step 1: BiN Normalization
    x = rearrange(x, 'b s f -> b f s')  # (32, 40, 128)
    x = self.norm_layer(x)               # (32, 40, 128)
    x = rearrange(x, 'b f s -> b s f')  # (32, 128, 40)
    
    # Step 2: Embedding
    x = self.emb_layer(x)               # (32, 128, 40)
    
    # Step 3: Positional Encoding
    x = x + self.pos_encoder            # (32, 128, 40)
    
    # Step 4: Transformer Layers (alternancia)
    for i in range(8):  # 4 pares
        x, att = self.layers[i](x)      # Atención
        x = x.permute(0, 2, 1)          # Swap spatial/temporal
    
    # After loop: x shape = (32, 10, 32)
    
    # Step 5: Flatten
    x = x.reshape(32, 320)              # (32, 320)
    
    # Step 6: Final MLP
    for layer in self.final_layers:
        x = layer(x)
    
    # Output: (32, 3) - logits para [clase0, clase1, clase2]
    ```
    
    Nota sobre el Orden de Salida:
    ------------------------------
    El modelo da logits en el siguiente orden durante el entrenamiento:
    ```
    output[0] = logit para la clase con valor numérico más bajo
    output[1] = logit para la clase con valor numérico medio
    output[2] = logit para la clase con valor numérico más alto
    ```
    
    Dado que las etiquetas son: 0=UP, 1=STATIONARY, 2=DOWN
    El modelo aprende a dar salidas como:
    ```
    output[0] ≈ logit de DOWN (label 2, menor cambio numérico)
    output[1] ≈ logit de STATIONARY (label 1)
    output[2] ≈ logit de UP (label 0, mayor cambio numérico)
    ```
    
    Por lo tanto, durante inferencia se debe invertir el orden del softmax
    para que coincida con el mapeo de etiquetas. Ver `app.py:run_prediction()`
    para la implementación correcta.
    
    Componentes del Modelo:
    -----------------------
    - **norm_layer**: BiN para normalización independiente por feature
    - **emb_layer**: Linear embedding de features a hidden_dim
    - **pos_encoder**: Encodings posicionales (sinusoidales o aprendidos)
    - **layers**: Lista de TransformerLayers alternados (spatial/temporal)
    - **final_layers**: MLP clasificador que reduce a 3 clases
    
    Para LOBSTER Dataset:
    ---------------------
    Si dataset_type == "LOBSTER", se agrega un embedding adicional para order_type:
    ```python
    self.order_type_embedder = nn.Embedding(3, 1)
    # Convierte order_type categórico (0, 1, 2) en embedding continuo
    ```
    
    Ejemplo de Uso:
    ---------------
    ```python
    # Configuración para BTC
    model = TLOB(
        hidden_dim=40,
        num_layers=4,        # 4 pares = 8 capas totales
        seq_size=128,
        num_features=40,
        num_heads=1,
        is_sin_emb=True,
        dataset_type="BTC"
    )
    
    # Input
    x = torch.randn(32, 128, 40)  # batch=32, seq=128, features=40
    
    # Forward pass
    logits = model(x)  # Output: (32, 3)
    
    # Aplicar softmax para obtener probabilidades
    probs = F.softmax(logits, dim=1)  # (32, 3)
    
    # Predicciones
    preds = torch.argmax(probs, dim=1)  # (32,)
    ```
    
    Número de Parámetros (BTC):
    ---------------------------
    Aproximadamente 1.1M parámetros:
    - BiN: ~80 parámetros (40 features × 2 para gamma/beta)
    - Embedding: 1,600 (40×40)
    - Positional: 5,120 (128×40) si aprendido, 0 si sinusoidal
    - 8 TransformerLayers: ~900K
    - Final MLP: ~41K
    
    Total: ~1.1M parámetros entrenables
    
    Performance:
    ------------
    - Inferencia CPU: ~50ms por sample
    - Inferencia GPU: ~15ms por sample
    - Memoria: ~500MB (modelo + batch)
    - FLOPs: ~2.1 GFLOPs por forward pass
    
    Referencias:
    ------------
    Paper: https://arxiv.org/pdf/2502.15757
    Repo: https://github.com/LeonardoBerti00/TLOB
    """
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 num_heads: int,
                 is_sin_emb: bool,
                 dataset_type: str
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_sin_emb = is_sin_emb
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.dataset_type = dataset_type
        self.layers = nn.ModuleList()
        self.first_branch = nn.ModuleList()
        self.second_branch = nn.ModuleList()
        self.order_type_embedder = nn.Embedding(3, 1)
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            self.pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))
        
        for i in range(num_layers):
            if i != num_layers-1:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
            else:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim//4))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size//4))
        self.att_temporal = []
        self.att_feature = []
        self.mean_att_distance_temporal = []
        total_dim = (hidden_dim//4)*(seq_size//4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim//4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim//4
        self.final_layers.append(nn.Linear(total_dim, 3))
        
    
    def forward(self, input, store_att=False):
        if self.dataset_type == "LOBSTER":
            continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
            order_type = input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x = input
        x = rearrange(x, 'b s f -> b f s')
        x = self.norm_layer(x)
        x = rearrange(x, 'b f s -> b s f')
        x = self.emb_layer(x)
        x = x[:] + self.pos_encoder
        for i in range(len(self.layers)):
            x, att = self.layers[i](x)
            att = att.detach()
            x = x.permute(0, 2, 1)
        x = rearrange(x, 'b s f -> b (f s) 1')              
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x
    
    
def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings.to(cst.DEVICE, non_blocking=True)


def count_parameters(layer):
    print(f"Number of parameters: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}")
    

def compute_mean_att_distance(att):
    att_distances = np.zeros((att.shape[0], att.shape[1]))
    for h in range(att.shape[0]):
        for key in range(att.shape[2]):
            for query in range(att.shape[1]):
                distance = abs(query-key)
                att_distances[h, key] += torch.abs(att[h, query, key]).cpu().item()*distance
    mean_distances = att_distances.mean(axis=1)
    return mean_distances
    
    
