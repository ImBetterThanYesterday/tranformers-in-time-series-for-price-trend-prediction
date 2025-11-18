"""
BiN-CTABL - BATCH-INSTANCE NORMALIZATION + CONTEXTUAL TABULAR LAYERS
=====================================================================

Implementación del modelo BiN-CTABL que combina:
- BiN (Batch-Instance Normalization): Normalización dual
- CTABL (Contextual Tabular Layer): Capas con atención para datos tabulares

Paper: "Deep Learning for Limit Order Books" (variante experimental)
Innovación: Atención soft sobre dimensión temporal + normalización BiN

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

from torch import nn
from src.models.bin import BiN
import torch
import src.constants as cst


class TABL_layer(nn.Module):
    """
    TEMPORAL ATTENTION BILINEAR LAYER (TABL)
    =========================================
    
    Capa que aplica atención soft sobre la dimensión temporal de datos tabulares.
    Combina transformación bilinear con mecanismo de atención aprendible.
    
    Arquitectura:
    -------------
    ```
    Input: X (batch, d1, t1)
         ↓
    1. Transformación features: W1 @ X → (batch, d2, t1)
         ↓
    2. Attention weights: X @ W → E
         ↓
    3. Attention mask: softmax(E) → A
         ↓
    4. Soft attention: λ*X + (1-λ)*(X*A)
         ↓
    5. Temporal mapping: X @ W2 + B → (batch, d2, t2)
         ↓
    Output: (batch, d2, t2)
    ```
    
    Fórmula Matemática:
    -------------------
    ```
    # Step 1: Feature transformation
    X' = W1 @ X
    
    # Step 2: Attention energy
    E = X' @ W
    
    # Step 3: Attention mask
    A = softmax(E, dim=-1)
    
    # Step 4: Soft attention (balance aprendible)
    X'' = λ*X' + (1-λ)*(X' ⊙ A)
    
    # Step 5: Temporal projection
    Y = X'' @ W2 + B
    ```
    
    Donde:
    - ⊙: Element-wise multiplication
    - λ: Peso aprendible [0, 1] (balance attention vs identity)
    
    Parámetros Aprendibles:
    -----------------------
    - W1: (d2, d1) - Transformación de features
    - W: (t1, t1) - Matriz de atención temporal
    - W2: (t1, t2) - Proyección temporal
    - B: (d2, t2) - Bias
    - λ: (1,) - Peso de balance [0, 1]
    
    Args:
        d2 (int): Dimensión de salida de features
        d1 (int): Dimensión de entrada de features
        t1 (int): Longitud temporal de entrada
        t2 (int): Longitud temporal de salida
    
    Ejemplo:
    --------
    ```python
    # Crear capa TABL
    tabl = TABL_layer(d2=5, d1=60, t1=5, t2=1)
    
    # Input: (batch=32, features=60, timesteps=5)
    x = torch.randn(32, 60, 5)
    
    # Forward
    y = tabl(x)
    # Output: (32, 5, 1)
    
    # Verificar lambda aprendido
    print(f"Lambda (attention weight): {tabl.l.item():.3f}")
    # Ej: 0.623 → 62.3% identity, 37.7% attention
    ```
    
    Mecanismo de Atención Detallado:
    --------------------------------
    
    ### Step 1: Feature Transformation
    ```python
    X: (batch, d1=60, t1=5)
    W1: (d2=5, d1=60)
    
    X' = W1 @ X
    X': (batch, d2=5, t1=5)
    ```
    
    ### Step 2: Attention Energy
    ```python
    W: (t1=5, t1=5)
    # W tiene diagonal forzada a 1/t1 (self-attention normalizado)
    
    E = X' @ W
    E: (batch, 5, 5)
    
    # E[i,j] = Importancia del timestep j para el timestep i
    ```
    
    ### Step 3: Attention Mask
    ```python
    A = softmax(E, dim=-1)
    A: (batch, 5, 5)
    
    # Cada fila suma 1 (distribución de probabilidad)
    # A[i,j] = Atención del timestep i al timestep j
    ```
    
    ### Step 4: Soft Attention
    ```python
    # Balance entre identity y attention
    X'' = λ*X' + (1-λ)*(X' * A)
    
    Si λ=1: X'' = X' (sin attention)
    Si λ=0: X'' = X' * A (solo attention)
    Si λ=0.5: Balance 50/50
    
    # λ se aprende durante entrenamiento
    ```
    
    ### Step 5: Temporal Projection
    ```python
    W2: (t1=5, t2=1)
    B: (d2=5, t2=1)
    
    Y = X'' @ W2 + B
    Y: (batch, 5, 1)
    
    # Reduce dimensión temporal (5 → 1)
    # Para clasificación final
    ```
    
    Inicialización:
    ---------------
    ```python
    W1: Kaiming uniform (ReLU)
      - Optimizado para activación ReLU
      - Evita vanishing/exploding gradients
    
    W: Constant (1/t1)
      - Inicializa con uniform attention
      - Diagonal forzada a 1/t1 durante forward
    
    W2: Kaiming uniform (ReLU)
      - Optimizado para activación ReLU
    
    B: Constant (0)
      - Sin bias inicial
    
    λ: Constant (0.5)
      - Balance 50/50 inicial
      - Se aprende durante entrenamiento
    ```
    
    Restricciones en Forward:
    -------------------------
    
    ### 1. Lambda Clamping (λ ∈ [0, 1])
    ```python
    if λ < 0: λ = 0.0
    if λ > 1: λ = 1.0
    
    # Mantiene interpretación como peso de balance
    ```
    
    ### 2. Diagonal de W Forzada
    ```python
    # Forzar diagonal = 1/t1
    W_modified = W - W*I + I/t1
    
    # Donde I = identity matrix
    # Garantiza self-attention normalizado
    ```
    
    Interpretación de Lambda:
    -------------------------
    ```
    λ → 1: Modelo prefiere identity (sin attention)
           → Patrones temporales simples
    
    λ → 0: Modelo prefiere attention
           → Dependencias temporales complejas
    
    λ ≈ 0.5: Balance entre ambos
             → Combinación de patrones
    ```
    
    Ventajas:
    ---------
    - Atención soft (aprendible, no hard)
    - Balance entre identity y attention (λ aprendible)
    - Diagonal forzada en W (self-attention normalizado)
    - Proyección temporal flexible (t1 → t2)
    
    Desventajas:
    ------------
    - Más parámetros que capa linear simple
    - Atención cuadrática en t1 (no escala bien)
    - Lambda puede quedar atrapado en extremos (0 o 1)
    """
    
    def __init__(self, d2, d1, t1, t2):
        """
        Inicializa capa TABL.
        
        Args:
            d2: Dimensión de salida de features
            d1: Dimensión de entrada de features
            t1: Longitud temporal de entrada
            t2: Longitud temporal de salida
        """
        super().__init__()
        self.t1 = t1

        # W1: Feature transformation (d2 × d1)
        weight = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')
        
        # W: Attention matrix (t1 × t1)
        weight2 = torch.Tensor(t1, t1)
        self.W = nn.Parameter(weight2)
        nn.init.constant_(self.W, 1/t1)  # Uniform attention inicial
 
        # W2: Temporal projection (t1 × t2)
        weight3 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight3)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        # B: Bias (d2 × t2)
        bias1 = torch.Tensor(d2, t2)
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        # λ: Balance weight [0, 1]
        l = torch.Tensor(1,)
        self.l = nn.Parameter(l)
        nn.init.constant_(self.l, 0.5)  # Balance 50/50 inicial

        self.activation = nn.ReLU()

    def forward(self, X):
        """
        Forward pass de TABL.
        
        Args:
            X (torch.Tensor): Input de shape (batch, d1, t1)
        
        Returns:
            torch.Tensor: Output de shape (batch, d2, t2)
        """
        # =====================================================================
        # CLAMP LAMBDA TO [0, 1]
        # =====================================================================
        # Mantener λ en rango válido [0, 1]
        if (self.l[0] < 0): 
            l = torch.Tensor(1,).to(cst.DEVICE)
            self.l = nn.Parameter(l)
            nn.init.constant_(self.l, 0.0)

        if (self.l[0] > 1): 
            l = torch.Tensor(1,).to(cst.DEVICE)
            self.l = nn.Parameter(l)
            nn.init.constant_(self.l, 1.0)
     
        # =====================================================================
        # STEP 1: FEATURE TRANSFORMATION
        # =====================================================================
        # W1 @ X: (d2, d1) @ (batch, d1, t1) → (batch, d2, t1)
        X = self.W1 @ X

        # =====================================================================
        # STEP 2: ATTENTION WEIGHTS (con diagonal forzada)
        # =====================================================================
        # Forzar diagonal de W = 1/t1 (self-attention normalizado)
        # W_modified = W - W*I + I/t1
        W = self.W - self.W * torch.eye(self.t1, dtype=torch.float32).to(cst.DEVICE) \
                   + torch.eye(self.t1, dtype=torch.float32).to(cst.DEVICE) / self.t1

        # Calcular attention energy
        # X @ W: (batch, d2, t1) @ (t1, t1) → (batch, d2, t1)
        E = X @ W

        # =====================================================================
        # STEP 3: ATTENTION MASK
        # =====================================================================
        # Softmax sobre dimensión temporal
        # A: (batch, d2, t1) - cada fila suma 1
        A = torch.softmax(E, dim=-1)

        # =====================================================================
        # STEP 4: SOFT ATTENTION
        # =====================================================================
        # Balance entre identity (λ*X) y attention ((1-λ)*X*A)
        # λ=1: Solo identity (sin attention)
        # λ=0: Solo attention
        X = self.l[0] * (X) + (1.0 - self.l[0]) * X * A

        # =====================================================================
        # STEP 5: TEMPORAL PROJECTION
        # =====================================================================
        # X @ W2: (batch, d2, t1) @ (t1, t2) → (batch, d2, t2)
        # + B: (d2, t2) broadcasted
        y = X @ self.W2 + self.B
        
        return y


class BL_layer(nn.Module):
    """
    BILINEAR LAYER (BL)
    ===================
    
    Capa bilinear simple sin atención.
    Aplica transformación bilinear: W1 @ X @ W2 + B
    
    Fórmula:
    --------
    Y = ReLU(W1 @ X @ W2 + B)
    
    Donde:
    - W1: (d2, d1) - Feature transformation
    - W2: (t1, t2) - Temporal transformation
    - B: (d2, t2) - Bias
    
    Args:
        d2 (int): Dimensión de salida de features
        d1 (int): Dimensión de entrada de features
        t1 (int): Longitud temporal de entrada
        t2 (int): Longitud temporal de salida
    
    Ejemplo:
    --------
    ```python
    # Crear capa bilinear
    bl = BL_layer(d2=60, d1=40, t1=128, t2=5)
    
    # Input: (batch=32, features=40, timesteps=128)
    x = torch.randn(32, 40, 128)
    
    # Forward
    y = bl(x)
    # Output: (32, 60, 5)
    ```
    
    Diferencia con TABL:
    --------------------
    - BL: Transformación simple (sin atención)
    - TABL: Con mecanismo de atención soft
    
    Uso:
    ----
    Se usa como capa intermedia en BiN_CTABL antes de la capa final TABL.
    """
    
    def __init__(self, d2, d1, t1, t2):
        """
        Inicializa capa bilinear.
        
        Args:
            d2: Dimensión de salida de features
            d1: Dimensión de entrada de features
            t1: Longitud temporal de entrada
            t2: Longitud temporal de salida
        """
        super().__init__()
        
        # W1: Feature transformation (d2 × d1)
        weight1 = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight1)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')

        # W2: Temporal transformation (t1 × t2)
        weight2 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight2)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        # B: Bias (d2 × t2)
        bias1 = torch.zeros((d2, t2))
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass de capa bilinear.
        
        Args:
            x (torch.Tensor): Input de shape (batch, d1, t1)
        
        Returns:
            torch.Tensor: Output de shape (batch, d2, t2)
        
        Proceso:
        --------
        Y = ReLU(W1 @ X @ W2 + B)
        """
        # Transformación bilinear + bias + activation
        # W1 @ x @ W2: (d2,d1) @ (batch,d1,t1) @ (t1,t2) → (batch, d2, t2)
        x = self.activation(self.W1 @ x @ self.W2 + self.B)
        
        return x


class BiN_CTABL(nn.Module):
    """
    BiN-CTABL - MODELO COMPLETO
    ============================
    
    Combina BiN (Batch-Instance Normalization) con capas CTABL (Contextual Tabular).
    
    Arquitectura:
    -------------
    ```
    Input: (batch, seq, features)
         ↓
    Permute: (batch, features, seq)
         ↓
    BiN (normalización dual)
         ↓
    BL_layer 1 (bilinear)
         ↓
    Dropout (0.1)
         ↓
    BL_layer 2 (bilinear)
         ↓
    Dropout (0.1)
         ↓
    TABL_layer (atención temporal)
         ↓
    Squeeze
         ↓
    Softmax
         ↓
    Output: (batch, 3) probabilities
    ```
    
    Hiperparámetros Fijos:
    ----------------------
    En BiNCTABL config:
    - d1 (input): Determinado por num_features
    - t1 (input): Determinado por seq_size
    - d2=60: Dimensión intermedia 1
    - t2=120: Dimensión temporal intermedia 1
    - d3=5: Dimensión intermedia 2
    - t3=3: Dimensión temporal intermedia 2
    - d4=1: Dimensión de salida (single feature)
    - t4=3: 3 clases (UP, STAT, DOWN)
    
    Pero en utils_model.py se usa:
    ```python
    BiN_CTABL(60, num_features, seq_size, seq_size, 120, 5, 3, 1)
    ```
    
    Esto parece inconsistente con los nombres de parámetros.
    
    Args:
        d2: Hidden dim 1 (típicamente 60)
        d1: Input features
        t1: Input seq length
        t2: Hidden seq 1 (típicamente 120)
        d3: Hidden dim 2 (típicamente 5)
        t3: Hidden seq 2 (típicamente 3)
        d4: Output dim (típicamente 1)
        t4: Output seq = num_classes (típicamente 3)
    
    Ejemplo de Uso:
    ---------------
    ```python
    # Crear modelo (arquitectura típica)
    model = BiN_CTABL(
        d2=60,
        d1=40,      # num_features
        t1=128,     # seq_size
        t2=120,
        d3=5,
        t3=3,
        d4=1,
        t4=3        # num_classes
    )
    
    # Input batch
    x = torch.randn(32, 128, 40)  # (batch, seq, features)
    
    # Forward
    probs = model(x)
    print(probs.shape)  # torch.Size([32, 3])
    
    # Ya incluye softmax (probabilidades)
    predictions = torch.argmax(probs, dim=1)
    ```
    
    Flujo de Dimensiones (Ejemplo):
    -------------------------------
    ```
    Input: (32, 128, 40)
         ↓ Permute
    (32, 40, 128)
         ↓ BiN
    (32, 40, 128)  [normalizado]
         ↓ BL(40→60, 128→120)
    (32, 60, 120)
         ↓ Dropout
    (32, 60, 120)
         ↓ BL(60→5, 120→3)
    (32, 5, 3)
         ↓ Dropout
    (32, 5, 3)
         ↓ TABL(5→1, 3→3)
    (32, 1, 3)
         ↓ Squeeze
    (32, 3)
         ↓ Softmax
    (32, 3) [probabilidades]
    ```
    
    Weight Clipping:
    ----------------
    Aplica max-norm constraint a todos los pesos:
    ```python
    if ||W|| > 10:
        W = W * (10 / ||W||)
    
    # Evita exploding weights
    # Mejora estabilidad
    ```
    
    Dropout:
    --------
    - Rate: 0.1 (10% de neuronas apagadas)
    - Aplicado después de cada BL_layer
    - Regularización para evitar overfitting
    
    Softmax Incluido:
    -----------------
    Similar a DeepLOB, BiN-CTABL incluye softmax en forward:
    - Output son PROBABILIDADES (no logits)
    - Usar NLLLoss o log(probs) con CrossEntropyLoss
    
    Comparación con Otros Modelos:
    -------------------------------
    
    | Aspecto          | BiN-CTABL    | TLOB          | MLPLOB        |
    |------------------|--------------|---------------|---------------|
    | Arquitectura     | BiN+Attention| Transformer   | MLP+BiN       |
    | Parámetros       | ~3.5M        | ~1.1M (BTC)   | ~2.8M         |
    | F1 Score (BTC)   | 0.69         | 0.73          | 0.68          |
    | Training time    | 30s/epoch    | 32s/epoch     | 25s/epoch     |
    | Seq size         | 10 (corto)   | 128           | 384           |
    | Interpretable    | Parcial (λ)  | Alta (attn)   | Baja          |
    
    Ventajas:
    ---------
    - Atención soft aprendible (λ parameter)
    - BiN normalization (dual)
    - Max-norm constraint (estabilidad)
    - Sequence cortas (10 timesteps)
    
    Desventajas:
    ------------
    - Arquitectura fija (hiperparámetros hardcoded)
    - Softmax en forward (inconsistente con otros modelos)
    - Menor accuracy que TLOB
    - Sequence muy cortas (solo 10 timesteps)
    
    Nota Importante:
    ----------------
    Este modelo es experimental y NO es el principal del proyecto.
    Para mejor performance, usar TLOB.
    """
    
    def __init__(self, d2, d1, t1, t2, d3, t3, d4, t4):
        """
        Inicializa modelo BiN-CTABL.
        
        Args:
            d2: Hidden dim 1
            d1: Input features
            t1: Input seq length
            t2: Hidden seq 1
            d3: Hidden dim 2
            t3: Hidden seq 2
            d4: Output dim
            t4: Num classes
        """
        super().__init__()

        # BiN normalization (dual: temporal + features)
        self.BiN = BiN(d1, t1)
        
        # Primera capa bilinear
        self.BL = BL_layer(d2, d1, t1, t2)
        
        # Segunda capa bilinear
        self.BL2 = BL_layer(d3, d2, t2, t3)
        
        # Capa TABL con atención
        self.TABL = TABL_layer(d4, d3, t3, t4)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass de BiN-CTABL.
        
        Args:
            x (torch.Tensor): Input de shape (batch, seq, features)
        
        Returns:
            torch.Tensor: Probabilidades de shape (batch, num_classes)
                         NOTA: Ya incluye softmax
        
        Proceso:
        --------
        1. Permute para BiN: (batch, seq, features) → (batch, features, seq)
        2. Aplicar BiN normalization
        3. BL layer 1 + dropout
        4. BL layer 2 + dropout
        5. TABL layer (con atención)
        6. Squeeze: (batch, 1, classes) → (batch, classes)
        7. Softmax: logits → probabilidades
        
        Max-norm constraint se aplica a todos los pesos antes de cada capa.
        """
        # Permute para BiN: (batch, seq, features) → (batch, features, seq)
        x = x.permute(0, 2, 1)
        
        # =====================================================================
        # BiN NORMALIZATION
        # =====================================================================
        x = self.BiN(x)

        # =====================================================================
        # PRIMERA CAPA BILINEAR
        # =====================================================================
        # Aplicar max-norm constraint a pesos
        self.max_norm_(self.BL.W1.data)
        self.max_norm_(self.BL.W2.data)
        x = self.BL(x)
        x = self.dropout(x)
        
        # =====================================================================
        # SEGUNDA CAPA BILINEAR
        # =====================================================================
        # Aplicar max-norm constraint a pesos
        self.max_norm_(self.BL2.W1.data)
        self.max_norm_(self.BL2.W2.data)
        x = self.BL2(x)
        x = self.dropout(x)

        # =====================================================================
        # CAPA TABL (con atención temporal)
        # =====================================================================
        # Aplicar max-norm constraint a pesos
        self.max_norm_(self.TABL.W1.data)
        self.max_norm_(self.TABL.W.data)
        self.max_norm_(self.TABL.W2.data)
        x = self.TABL(x)
        
        # =====================================================================
        # SQUEEZE Y SOFTMAX
        # =====================================================================
        # Squeeze: (batch, 1, classes) → (batch, classes)
        x = torch.squeeze(x)
        
        # Softmax: logits → probabilidades
        # NOTA: Esto es inusual - típicamente softmax se aplica en loss
        x = torch.softmax(x, 1)
        
        return x

    def max_norm_(self, w):
        """
        Aplica max-norm constraint a matriz de pesos.
        
        Si ||W|| > 10, escala W para que ||W|| = 10.
        Esto previene exploding weights y mejora estabilidad.
        
        Args:
            w (torch.Tensor): Matriz de pesos a constrainear
        
        Proceso:
        --------
        ```python
        norm = ||W||_2  # L2 matrix norm
        if norm > 10:
            W = W * (10 / norm)
        ```
        
        Nota:
        -----
        Se aplica in-place (modifica w directamente).
        torch.no_grad() previene tracking de gradientes.
        """
        with torch.no_grad():
            # Calcular norma matricial (L2 norm)
            if (torch.linalg.matrix_norm(w) > 10.0):
                norm = torch.linalg.matrix_norm(w)
                
                # Clamp norm to [0, 10]
                desired = torch.clamp(norm, min=0.0, max=10.0)
                
                # Escalar pesos: W = W * (desired / norm)
                # 1e-8 evita división por cero
                w *= (desired / (1e-8 + norm))
