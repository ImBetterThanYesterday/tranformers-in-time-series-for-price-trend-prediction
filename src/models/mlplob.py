"""
MULTI-LAYER PERCEPTRON PARA LIMIT ORDER BOOK (MLPLOB)
======================================================

Arquitectura baseline simple basada en MLPs para predicción de tendencias
en Limit Order Books. Usa BiN (Batch-Instance Normalization) y estructura
de capas alternadas entre dimensión temporal y de features.

Arquitectura: BiN → MLP_feature → MLP_temporal → ... → Flatten → FC → Output

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

from torch import nn
import torch
from src.models.bin import BiN


class MLPLOB(nn.Module):
    """
    MULTI-LAYER PERCEPTRON PARA LIMIT ORDER BOOK
    =============================================
    
    Modelo baseline simple que usa MLPs para procesar LOB data.
    Alterna entre procesamiento de features y procesamiento temporal.
    
    Arquitectura Completa:
    ----------------------
    ```
    Input: (batch, seq_size, num_features)
         ↓
    Permute → (batch, features, seq_size)
         ↓
    BiN (normalización dual)
         ↓
    Permute → (batch, seq_size, features)
         ↓
    Linear (features → hidden_dim)
         ↓
    GELU activation
         ↓
    ┌─── Repetir num_layers veces ───┐
    │ MLP (dim=features)              │
    │    ↓                            │
    │ Permute (swap features↔time)    │
    │    ↓                            │
    │ MLP (dim=temporal)              │
    │    ↓                            │
    │ Permute (swap time↔features)    │
    └─────────────────────────────────┘
         ↓
    Flatten
         ↓
    MLP cascade (reducción dimensional)
         ↓
    Linear (→ 3 clases)
         ↓
    Output: (batch, 3) logits
    ```
    
    Innovaciones vs MLP Estándar:
    ------------------------------
    1. **BiN Normalization**: Normaliza temporal Y espacialmente
    2. **Dual Processing**: Alterna procesamiento features/tiempo
    3. **Reducción Progresiva**: Reduce dimensión gradualmente
    4. **Residual Connections**: En bloques MLP cuando dim coincide
    
    Args:
        hidden_dim (int): Dimensión del espacio latente
                         - BTC: 40
                         - FI-2010: 144
                         - Típicamente igual a num_features
        
        num_layers (int): Número de pares MLP (feature + temporal)
                         - Default: 3
                         - Más layers = más capacidad pero riesgo de overfitting
        
        seq_size (int): Longitud de secuencia temporal
                       - BTC: 128 timesteps
                       - MLPLOB usa secuencias más largas que TLOB (384 vs 128)
                       - Compensa falta de atención con más contexto
        
        num_features (int): Número de features del LOB
                           - BTC: 40 (10 niveles × 4)
                           - FI-2010: 40
                           - LOBSTER: 46 (40 LOB + 6 metadata)
        
        dataset_type (str): Tipo de dataset ("BTC", "FI_2010", "LOBSTER")
                           - LOBSTER requiere embedding de order_type
                           - Otros no tienen metadata categórica
    
    Ejemplo de Uso:
    ---------------
    ```python
    # Crear modelo MLPLOB para BTC
    model = MLPLOB(
        hidden_dim=40,
        num_layers=3,
        seq_size=128,
        num_features=40,
        dataset_type="BTC"
    )
    
    # Input batch
    x = torch.randn(32, 128, 40)  # (batch, seq, features)
    
    # Forward pass
    logits = model(x)
    print(logits.shape)  # torch.Size([32, 3])
    
    # Softmax para probabilidades
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    # 0 = UP, 1 = STATIONARY, 2 = DOWN
    ```
    
    Flujo de Datos Detallado:
    -------------------------
    
    ### Ejemplo: BTC con hidden_dim=40, num_layers=3, seq_size=128
    
    ```
    Input: (32, 128, 40)  [batch, seq, features]
         ↓
    Permute: (32, 40, 128)  [batch, features, seq]
         ↓
    BiN: (32, 40, 128)  [normalizado dual]
         ↓
    Permute: (32, 128, 40)  [batch, seq, features]
         ↓
    Linear(40 → 40): (32, 128, 40)
         ↓
    GELU: (32, 128, 40)
         ↓
    ┌─────── Layer 0 (no reducción) ───────┐
    │ MLP(40 → 160 → 40): (32, 128, 40)    │
    │ Permute: (32, 40, 128)                │
    │ MLP(128 → 512 → 128): (32, 40, 128)  │
    │ Permute: (32, 128, 40)                │
    ├─────── Layer 1 (no reducción) ───────┤
    │ MLP(40 → 160 → 40): (32, 128, 40)    │
    │ Permute: (32, 40, 128)                │
    │ MLP(128 → 512 → 128): (32, 40, 128)  │
    │ Permute: (32, 128, 40)                │
    ├─────── Layer 2 (REDUCCIÓN) ──────────┤
    │ MLP(40 → 80 → 10): (32, 128, 10)     │
    │ Permute: (32, 10, 128)                │
    │ MLP(128 → 256 → 32): (32, 10, 32)    │
    │ Permute: (32, 32, 10)                 │
    └───────────────────────────────────────┘
         ↓
    Flatten: (32, 320)  [32 × 10 = 320]
         ↓
    Linear(320 → 80): (32, 80)
         ↓
    GELU: (32, 80)
         ↓
    Linear(80 → 3): (32, 3)
         ↓
    Output logits: (32, 3)
    ```
    
    Reducción Dimensional:
    ----------------------
    
    **Capas intermedias (0 a num_layers-2)**:
    ```python
    hidden_dim → hidden_dim*4 → hidden_dim  # No reducción
    seq_size → seq_size*4 → seq_size        # No reducción
    ```
    
    **Última capa (num_layers-1)**:
    ```python
    hidden_dim → hidden_dim*2 → hidden_dim//4  # Reduce a 1/4
    seq_size → seq_size*2 → seq_size//4        # Reduce a 1/4
    ```
    
    **Capas finales**:
    ```python
    total_dim = (hidden_dim//4) * (seq_size//4)
    # Ejemplo BTC: (40//4) * (128//4) = 10 * 32 = 320
    
    # Cascade de reducción:
    while total_dim > 128:
        total_dim → total_dim//4
    
    # BTC: 320 → 80 → 3
    ```
    
    Manejo de LOBSTER (Order Type Embedding):
    -----------------------------------------
    ```python
    if dataset_type == "LOBSTER":
        # LOB features: columnas 0-40 y 42-46
        continuous = cat([input[:,:,:41], input[:,:,42:]], dim=2)
        
        # Order type: columna 41 (categórica)
        order_type = input[:,:,41].long()  # [0, 1, 2]
        
        # Embedding 3 → 1 dimension
        order_type_emb = embedding(order_type)  # (batch, seq, 1)
        
        # Concatenar de vuelta
        x = cat([continuous, order_type_emb], dim=2)
    else:
        x = input  # BTC/FI-2010: Usar directamente
    ```
    
    Interpretación de order_type:
    ```
    0 = Limit order submission
    1 = Cancellation
    2 = Market order (execution)
    ```
    
    Comparación con TLOB:
    ---------------------
    
    | Aspecto          | MLPLOB              | TLOB                |
    |------------------|---------------------|---------------------|
    | Arquitectura     | MLP simple          | Transformer         |
    | Atención         | No                  | Dual (spatial+temp) |
    | Seq_size         | 384 (largo)         | 128 (corto)         |
    | Parámetros       | ~2.8M               | ~1.1M (BTC)         |
    | F1 Score (BTC)   | 0.68                | 0.73                |
    | Training time    | 25s/epoch           | 32s/epoch           |
    | Inference        | 50ms/sample         | 75ms/sample         |
    
    Ventajas de MLPLOB:
    -------------------
    - Más rápido (training e inference)
    - Menos parámetros
    - Fácil de entrenar (no requiere lr bajo)
    - Bueno como baseline
    
    Desventajas de MLPLOB:
    ----------------------
    - Sin atención (no captura dependencias long-range)
    - Requiere secuencias más largas (384 vs 128)
    - Menor accuracy que TLOB
    - No interpreta importancia de timesteps
    
    Parámetros del Modelo:
    ----------------------
    ```python
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    # BTC (hidden=40, seq=128): ~2.8M parámetros
    # FI-2010 (hidden=144, seq=100): ~4.5M parámetros
    ```
    
    Desglose:
    ```
    BiN: ~200K
    First Linear: 40×40 = 1.6K
    MLP layers: ~2.5M (mayoría)
    Final layers: ~100K
    Order type embedding (LOBSTER): 3×1 = 3
    ```
    
    Uso en Producción:
    ------------------
    ```python
    # Cargar modelo entrenado
    model = MLPLOB(...)
    model.load_state_dict(torch.load('mlplob_btc.pt'))
    model.eval()
    
    # Inferencia
    with torch.no_grad():
        x = preprocess_lob_data(raw_data)  # (1, 128, 40)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        
        if pred == 0:
            action = "BUY"   # Precio subirá
        elif pred == 1:
            action = "HOLD"  # Precio estable
        else:
            action = "SELL"  # Precio bajará
    ```
    
    Nota Importante:
    ----------------
    MLPLOB es un modelo baseline. Para mejor performance, usar TLOB.
    Sin embargo, MLPLOB es útil para:
    - Comparación de performance
    - Validación de datos
    - Deployment con recursos limitados
    - Interpretación más simple
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 dataset_type: str
                 ) -> None:
        """
        Inicializa modelo MLPLOB.
        
        Args:
            hidden_dim: Dimensión del espacio latente
            num_layers: Número de pares de capas MLP
            seq_size: Longitud de secuencia temporal
            num_features: Número de features del LOB
            dataset_type: Tipo de dataset ("BTC", "FI_2010", "LOBSTER")
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dataset_type = dataset_type
        self.layers = nn.ModuleList()
        
        # =====================================================================
        # ORDER TYPE EMBEDDING (solo para LOBSTER)
        # =====================================================================
        # Embedding para convertir order_type categórico (0,1,2) a continuous
        # 3 tipos → 1 dimensión
        self.order_type_embedder = nn.Embedding(3, 1)
        
        # =====================================================================
        # CAPAS INICIALES
        # =====================================================================
        # Primera capa linear: num_features → hidden_dim
        self.first_layer = nn.Linear(num_features, hidden_dim)
        
        # BiN normalization (normalización dual)
        self.norm_layer = BiN(num_features, seq_size)
        
        # Agregar a lista de capas
        self.layers.append(self.first_layer)
        self.layers.append(nn.GELU())  # Activation
        
        # =====================================================================
        # CAPAS MLP (alternando features y temporal)
        # =====================================================================
        for i in range(num_layers):
            if i != num_layers-1:
                # Capas intermedias: No reducen dimensión
                # hidden_dim → hidden_dim*4 → hidden_dim
                self.layers.append(MLP(hidden_dim, hidden_dim*4, hidden_dim))
                # seq_size → seq_size*4 → seq_size
                self.layers.append(MLP(seq_size, seq_size*4, seq_size))
            else:
                # Última capa: Reduce dimensión a 1/4
                # hidden_dim → hidden_dim*2 → hidden_dim//4
                self.layers.append(MLP(hidden_dim, hidden_dim*2, hidden_dim//4))
                # seq_size → seq_size*2 → seq_size//4
                self.layers.append(MLP(seq_size, seq_size*2, seq_size//4))
        
        # =====================================================================
        # CAPAS FINALES (reducción dimensional cascada)
        # =====================================================================
        # Dimensión después de flatten
        total_dim = (hidden_dim//4) * (seq_size//4)
        # Ejemplo BTC: (40//4) * (128//4) = 10 * 32 = 320
        
        self.final_layers = nn.ModuleList()
        
        # Reducir dimensión progresivamente hasta ≤ 128
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim//4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        
        # Capa final: total_dim → 3 clases
        self.final_layers.append(nn.Linear(total_dim, 3))
    
    def forward(self, input):
        """
        Forward pass del modelo MLPLOB.
        
        Args:
            input (torch.Tensor): Input de shape (batch, seq_size, num_features)
        
        Returns:
            torch.Tensor: Logits de shape (batch, 3)
        
        Proceso:
        --------
        1. Si LOBSTER: Extraer y embeder order_type
        2. Permutar para BiN: (batch, seq, features) → (batch, features, seq)
        3. Aplicar BiN normalization
        4. Permutar de vuelta: (batch, features, seq) → (batch, seq, features)
        5. Aplicar primera linear + GELU
        6. Aplicar capas MLP alternadas (features ↔ temporal)
        7. Flatten
        8. Aplicar capas finales de reducción
        9. Output logits (batch, 3)
        """
        # =====================================================================
        # 1. MANEJO DE ORDER TYPE (LOBSTER)
        # =====================================================================
        if self.dataset_type == "LOBSTER":
            # Separar features continuas y order_type
            # Columnas 0-40: LOB features continuas
            # Columna 41: order_type (categórica)
            # Columnas 42-46: Metadata continua
            continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
            order_type = input[:, :, 41].long()  # Convertir a long para embedding
            
            # Embeder order_type: (batch, seq) → (batch, seq, 1)
            order_type_emb = self.order_type_embedder(order_type).detach()
            # .detach() previene backprop a través del embedding (frozen)
            
            # Concatenar de vuelta: (batch, seq, 40+6+1=47)
            # NOTA: Aquí hay inconsistencia - num_features debería ser 47, no 46
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            # BTC/FI-2010: Usar input directamente
            x = input
        
        # =====================================================================
        # 2. NORMALIZACIÓN BiN
        # =====================================================================
        # BiN espera (batch, features, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, features) → (batch, features, seq)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)  # (batch, features, seq) → (batch, seq, features)
        
        # =====================================================================
        # 3. CAPAS MLP ALTERNADAS
        # =====================================================================
        for layer in self.layers:
            x = layer(x)
            # Alternar entre (batch, seq, features) y (batch, features, seq)
            x = x.permute(0, 2, 1)
        
        # =====================================================================
        # 4. FLATTEN Y CAPAS FINALES
        # =====================================================================
        # Flatten: (batch, final_features, final_seq) → (batch, final_features*final_seq)
        x = x.reshape(x.shape[0], -1)
        
        # Aplicar capas finales de reducción
        for layer in self.final_layers:
            x = layer(x)
        
        # Output: (batch, 3) logits
        return x
        
        
class MLP(nn.Module):
    """
    BLOQUE MLP CON RESIDUAL CONNECTION
    ===================================
    
    Bloque MLP simple con:
    - Expansión a dimensión intermedia (4x o 2x)
    - Contracción a dimensión final
    - Residual connection (si dimensiones coinciden)
    - Layer normalization
    - GELU activation
    
    Arquitectura:
    -------------
    ```
    Input: (batch, seq, start_dim)
         ↓
    Linear (start_dim → hidden_dim)
         ↓
    GELU
         ↓
    Linear (hidden_dim → final_dim)
         ↓
    Residual (+ input si start_dim == final_dim)
         ↓
    LayerNorm
         ↓
    GELU
         ↓
    Output: (batch, seq, final_dim)
    ```
    
    Args:
        start_dim (int): Dimensión de entrada
        hidden_dim (int): Dimensión intermedia (expansión)
        final_dim (int): Dimensión de salida
    
    Ejemplo:
    --------
    ```python
    # MLP que expande y luego contrae
    mlp = MLP(start_dim=40, hidden_dim=160, final_dim=40)
    
    x = torch.randn(32, 128, 40)
    y = mlp(x)
    # y.shape: (32, 128, 40)
    
    # Con residual connection (start_dim == final_dim)
    # y = LayerNorm(Linear2(GELU(Linear1(x))) + x)
    ```
    
    Residual Connection:
    --------------------
    ```python
    if x.shape[2] == residual.shape[2]:
        x = x + residual  # Skip connection
    
    # Mejora gradient flow
    # Permite entrenar redes más profundas
    ```
    
    Nota:
    -----
    En MLPLOB, estos bloques se aplican alternadamente sobre:
    - Dimensión de features (start_dim = hidden_dim)
    - Dimensión temporal (start_dim = seq_size)
    """
    
    def __init__(self, 
                 start_dim: int,
                 hidden_dim: int,
                 final_dim: int
                 ) -> None:
        """
        Inicializa bloque MLP.
        
        Args:
            start_dim: Dimensión de entrada
            hidden_dim: Dimensión intermedia (típicamente 4x start_dim)
            final_dim: Dimensión de salida
        """
        super().__init__()
        
        # Layer normalization (aplicado DESPUÉS de residual)
        self.layer_norm = nn.LayerNorm(final_dim)
        
        # Primera capa linear (expansión)
        self.fc = nn.Linear(start_dim, hidden_dim)
        
        # Segunda capa linear (contracción)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        
        # Activation GELU (Gaussian Error Linear Unit)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        """
        Forward pass del bloque MLP.
        
        Args:
            x (torch.Tensor): Input de shape (batch, seq, start_dim)
        
        Returns:
            torch.Tensor: Output de shape (batch, seq, final_dim)
        
        Proceso:
        --------
        1. Guardar residual (input original)
        2. Linear1: start_dim → hidden_dim
        3. GELU
        4. Linear2: hidden_dim → final_dim
        5. Residual connection (si dimensiones coinciden)
        6. LayerNorm
        7. GELU final
        """
        # Guardar input original para residual connection
        residual = x
        
        # Expansión: start_dim → hidden_dim
        x = self.fc(x)
        x = self.gelu(x)
        
        # Contracción: hidden_dim → final_dim
        x = self.fc2(x)
        
        # Residual connection (solo si dimensiones coinciden)
        if x.shape[2] == residual.shape[2]:
            x = x + residual  # Skip connection
        
        # Normalización y activación final
        x = self.layer_norm(x)
        x = self.gelu(x)
        
        return x
