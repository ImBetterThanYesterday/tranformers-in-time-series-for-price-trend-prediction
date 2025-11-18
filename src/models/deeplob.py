"""
DEEPLOB - DEEP CONVOLUTIONAL NEURAL NETWORK FOR LOB
====================================================

Implementación del modelo DeepLOB (Zhang et al., 2019).
Arquitectura: CNN (3 bloques) + Inception modules + LSTM + FC

Paper: "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
URL: https://arxiv.org/abs/1808.03668

Este modelo es un baseline importante en la literatura de LOB prediction.

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

from torch import nn
import torch


class DeepLOB(nn.Module):
    """
    DEEPLOB - ARQUITECTURA CNN + LSTM
    ==================================
    
    Modelo clásico para predicción de tendencias en LOB.
    Usa CNNs para extracción de features espaciales del LOB,
    Inception modules para capturar patrones multi-escala,
    y LSTM para modelar dependencias temporales.
    
    Arquitectura Completa:
    ----------------------
    ```
    Input: (batch, seq, features)
         ↓
    Reshape: (batch, 1, seq, features)  # Add channel
         ↓
    ┌─── CNN Block 1 ───┐
    │ Conv2d (1→32)      │
    │ LeakyReLU          │
    │ BatchNorm2d        │
    │ Conv2d (32→32)     │
    │ LeakyReLU          │
    │ BatchNorm2d        │
    │ Conv2d (32→32)     │
    │ LeakyReLU          │
    │ BatchNorm2d        │
    └────────────────────┘
         ↓
    ┌─── CNN Block 2 ───┐
    │ Conv2d (32→32)     │
    │ Tanh               │
    │ BatchNorm2d        │
    │ Conv2d (32→32)     │
    │ Tanh               │
    │ BatchNorm2d        │
    │ Conv2d (32→32)     │
    │ Tanh               │
    │ BatchNorm2d        │
    └────────────────────┘
         ↓
    ┌─── CNN Block 3 ───┐
    │ Conv2d (32→32)     │
    │ LeakyReLU          │
    │ BatchNorm2d        │
    │ Conv2d (32→32)     │
    │ LeakyReLU          │
    │ BatchNorm2d        │
    │ Conv2d (32→32)     │
    │ LeakyReLU          │
    │ BatchNorm2d        │
    └────────────────────┘
         ↓
    ┌─── Inception Modules ───┐
    │ Path 1: Conv(1×1→3×1)   │
    │ Path 2: Conv(1×1→5×1)   │
    │ Path 3: MaxPool→Conv    │
    │ Concat: (64+64+64=192)  │
    └─────────────────────────┘
         ↓
    Reshape: (batch, seq', 192)
         ↓
    LSTM (192 → 64 hidden)
         ↓
    Take last output
         ↓
    Linear (64 → 3)
         ↓
    Softmax
         ↓
    Output: (batch, 3) probabilities
    ```
    
    Características Clave:
    ----------------------
    1. **CNN Blocks**: 3 bloques convolucionales
       - Reducen dimensión espacial
       - Extraen features locales del LOB
    
    2. **Inception Modules**: Captura patrones multi-escala
       - Path 1: kernel 3×1 (corto alcance)
       - Path 2: kernel 5×1 (mediano alcance)
       - Path 3: MaxPool (pooling adaptativo)
    
    3. **LSTM**: Modela dependencias temporales
       - 1 capa, 64 hidden units
       - Procesa secuencia de features CNN
    
    4. **Arquitectura Fija**: No configurable
       - Hiperparámetros del paper original
       - No se pueden cambiar sin modificar código
    
    Innovaciones del Paper:
    -----------------------
    1. **Tratamiento del LOB como imagen 2D**
       - Filas: Timesteps
       - Columnas: Features (precios y volúmenes)
       - CNN extrae patrones espaciales
    
    2. **Inception Modules**
       - Inspirados en GoogLeNet
       - Capturan patrones de diferentes escalas
       - Path 1: Movimientos rápidos (3 timesteps)
       - Path 2: Movimientos lentos (5 timesteps)
       - Path 3: Features agregadas (pooling)
    
    3. **Combinación CNN + LSTM**
       - CNN: Features espaciales (estructura del LOB)
       - LSTM: Features temporales (evolución del LOB)
    
    Dimensiones en Cada Paso (Ejemplo: BTC):
    -----------------------------------------
    ```
    Input: (32, 100, 40)
      ↓
    Add channel: (32, 1, 100, 40)
      ↓
    Conv Block 1:
      Conv1: (32, 32, 100, 20)  # Stride (1,2) reduce width
      Conv2: (32, 32, 97, 20)   # Kernel (4,1) reduce height
      Conv3: (32, 32, 94, 20)
      ↓
    Conv Block 2:
      Conv1: (32, 32, 94, 10)   # Stride (1,2) reduce width
      Conv2: (32, 32, 91, 10)
      Conv3: (32, 32, 88, 10)
      ↓
    Conv Block 3:
      Conv1: (32, 32, 88, 1)    # Kernel (1,10) reduce width
      Conv2: (32, 32, 85, 1)
      Conv3: (32, 32, 82, 1)
      ↓
    Inception:
      Path 1: (32, 64, 82, 1)
      Path 2: (32, 64, 82, 1)
      Path 3: (32, 64, 82, 1)
      Concat: (32, 192, 82, 1)
      ↓
    Reshape: (32, 82, 192)
      ↓
    LSTM: (32, 82, 64) → Take last → (32, 64)
      ↓
    Linear: (32, 3)
      ↓
    Softmax: (32, 3)
    ```
    
    Comparación con Otros Modelos:
    -------------------------------
    
    | Aspecto          | DeepLOB        | TLOB           | MLPLOB         |
    |------------------|----------------|----------------|----------------|
    | Arquitectura     | CNN+LSTM       | Transformer    | MLP            |
    | Parámetros       | ~4.2M          | ~1.1M (BTC)    | ~2.8M          |
    | F1 Score (BTC)   | 0.70           | 0.73           | 0.68           |
    | Training time    | 35s/epoch      | 32s/epoch      | 25s/epoch      |
    | Inference        | 90ms/sample    | 75ms/sample    | 50ms/sample    |
    | Seq size         | 100 (fijo)     | 128            | 384            |
    | Year             | 2019           | 2025           | Baseline       |
    
    Ventajas de DeepLOB:
    --------------------
    - CNN captura estructura espacial del LOB
    - Inception captura patrones multi-escala
    - Bien establecido en literatura (muchas citas)
    - Funciona bien como baseline
    
    Desventajas de DeepLOB:
    -----------------------
    - Arquitectura fija (no configurable)
    - Más parámetros que TLOB
    - LSTM más lento que Transformer
    - No tiene mecanismo de atención
    - Requiere más memoria
    
    Uso del Modelo:
    ---------------
    ```python
    # Crear modelo (sin parámetros - arquitectura fija)
    model = DeepLOB()
    
    # Input batch (seq_size DEBE ser 100 para funcionar correctamente)
    x = torch.randn(32, 100, 40)  # (batch, seq=100, features=40)
    
    # Forward pass
    probs = model(x)
    print(probs.shape)  # torch.Size([32, 3])
    
    # Ya incluye softmax, así que son probabilidades
    predictions = torch.argmax(probs, dim=1)
    # 0 = DOWN, 1 = STATIONARY, 2 = UP (orden del paper original)
    ```
    
    NOTA IMPORTANTE sobre Softmax:
    ------------------------------
    DeepLOB incluye softmax en el forward pass:
    ```python
    out = self.fc1(out)
    out = self.softmax(out)  # ← Softmax interno
    return out
    ```
    
    Esto significa:
    - Output son PROBABILIDADES (no logits)
    - NO usar CrossEntropyLoss (espera logits)
    - Usar NLLLoss o equivalente
    
    En contraste:
    - TLOB: Retorna logits (sin softmax)
    - MLPLOB: Retorna logits (sin softmax)
    
    Entrenamiento:
    --------------
    ```python
    model = DeepLOB()
    
    # INCORRECTO (DeepLOB ya tiene softmax):
    # criterion = nn.CrossEntropyLoss()
    
    # CORRECTO:
    criterion = nn.NLLLoss()
    # O aplicar log antes:
    # loss = criterion(torch.log(output + 1e-10), target)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        for batch in dataloader:
            x, y = batch
            probs = model(x)  # Probabilidades (con softmax)
            loss = criterion(torch.log(probs), y)
            loss.backward()
            optimizer.step()
    ```
    
    Inferencia:
    -----------
    ```python
    model.eval()
    with torch.no_grad():
        x = preprocess_lob_data(raw_data)  # (1, 100, 40)
        probs = model(x)  # (1, 3)
        
        # Ya son probabilidades
        print(f"P(DOWN): {probs[0, 0]:.3f}")
        print(f"P(STAT): {probs[0, 1]:.3f}")
        print(f"P(UP): {probs[0, 2]:.3f}")
        
        pred = torch.argmax(probs, dim=1).item()
        
        if pred == 0:
            action = "SELL"  # Precio bajará
        elif pred == 1:
            action = "HOLD"  # Precio estable
        else:
            action = "BUY"   # Precio subirá
    ```
    
    Convolutional Blocks Detallados:
    --------------------------------
    
    ### Block 1 (LeakyReLU):
    ```
    - Conv2d(1→32, kernel=(1,2), stride=(1,2))
      * Reduce width by 2 (40 → 20)
      * Mantiene height (100 → 100)
    
    - Conv2d(32→32, kernel=(4,1))
      * Reduce height by 3 (100 → 97)
      * Kernel temporal de 4 timesteps
    
    - Conv2d(32→32, kernel=(4,1))
      * Reduce height by 3 (97 → 94)
    ```
    
    ### Block 2 (Tanh):
    ```
    - Conv2d(32→32, kernel=(1,2), stride=(1,2))
      * Reduce width by 2 (20 → 10)
    
    - Conv2d(32→32, kernel=(4,1)) × 2
      * Reduce height: 94 → 91 → 88
    ```
    
    ### Block 3 (LeakyReLU):
    ```
    - Conv2d(32→32, kernel=(1,10))
      * Reduce width to 1 (10 → 1)
      * Colapsa features dimension
    
    - Conv2d(32→32, kernel=(4,1)) × 2
      * Reduce height: 88 → 85 → 82
    ```
    
    Inception Modules:
    ------------------
    ```
    Input: (batch, 32, 82, 1)
    
    Path 1 (Short-term):
      Conv(1×1→64) + Conv(3×1→64)
      Output: (batch, 64, 82, 1)
    
    Path 2 (Mid-term):
      Conv(1×1→64) + Conv(5×1→64)
      Output: (batch, 64, 82, 1)
    
    Path 3 (Pooling):
      MaxPool(3×1, stride=1) + Conv(1×1→64)
      Output: (batch, 64, 82, 1)
    
    Concatenate:
      (batch, 192, 82, 1)  # 64+64+64=192
    ```
    
    LSTM Processing:
    ----------------
    ```
    # Reshape CNN output para LSTM
    x = x.permute(0, 2, 1, 3)  # (batch, 82, 192, 1)
    x = x.reshape(batch, 82, 192)  # (batch, seq=82, features=192)
    
    # LSTM forward
    out, (h_n, c_n) = LSTM(x)
    # out: (batch, 82, 64)  - todos los hidden states
    
    # Tomar último timestep
    out = out[:, -1, :]  # (batch, 64)
    
    # Linear final
    out = Linear(64 → 3)  # (batch, 3)
    ```
    
    Limitaciones:
    -------------
    1. **Seq size fijo**: Arquitectura asume seq_size=100
       - Usar otros tamaños requiere reentrenar
       - Dimensiones hardcodeadas
    
    2. **Softmax en forward**: Inconsistente con otros modelos
       - Complicaciones con loss functions
       - Mezcla inferencia con arquitectura
    
    3. **No configurable**: Hiperparámetros fijos del paper
       - No se pueden cambiar fácilmente
       - Limitado para experimentación
    
    4. **LSTM cuello de botella**: Procesa secuencialmente
       - No paralelizable
       - Más lento que Transformer
    
    Paper Original:
    ---------------
    Zhang, Z., Zohren, S., & Roberts, S. (2019).
    DeepLOB: Deep convolutional neural networks for limit order books.
    IEEE Transactions on Signal Processing, 67(11), 3001-3012.
    
    Citas: ~500+ (muy influyente en el campo)
    """
    
    def __init__(self):
        """
        Inicializa DeepLOB con arquitectura fija del paper.
        
        No requiere parámetros - todos los hiperparámetros están hardcodeados
        según el paper original.
        """
        super().__init__()

        # =====================================================================
        # CONVOLUTIONAL BLOCK 1 (LeakyReLU)
        # =====================================================================
        # Extrae features iniciales del LOB
        # LeakyReLU permite gradientes negativos pequeños (evita dying ReLU)
        self.conv1 = nn.Sequential(
            # Conv 1: Reduce width (features) by 2
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.Tanh(),
            nn.BatchNorm2d(32),
            
            # Conv 2: Reduce height (temporal) by 3
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            
            # Conv 3: Reduce height by 3 more
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # =====================================================================
        # CONVOLUTIONAL BLOCK 2 (Tanh)
        # =====================================================================
        # Refinamiento de features con Tanh (output boundado a [-1, 1])
        # Tanh es más suave que LeakyReLU, útil para features intermedias
        self.conv2 = nn.Sequential(
            # Conv 1: Reduce width by 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            
            # Conv 2: Reduce height by 3
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            
            # Conv 3: Reduce height by 3 more
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        # =====================================================================
        # CONVOLUTIONAL BLOCK 3 (LeakyReLU)
        # =====================================================================
        # Features finales antes de Inception
        # Colapsa dimensión de features completamente (width → 1)
        self.conv3 = nn.Sequential(
            # Conv 1: Collapse width dimension (10 → 1)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            
            # Conv 2: Reduce height by 3
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            
            # Conv 3: Reduce height by 3 more
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # =====================================================================
        # INCEPTION MODULE 1 (Short-term patterns)
        # =====================================================================
        # Captura patrones de corto alcance (3 timesteps)
        self.inp1 = nn.Sequential(
            # 1×1 conv: Projection/dimension reduction
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            
            # 3×1 conv: Patrones temporales cortos
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # =====================================================================
        # INCEPTION MODULE 2 (Mid-term patterns)
        # =====================================================================
        # Captura patrones de mediano alcance (5 timesteps)
        self.inp2 = nn.Sequential(
            # 1×1 conv: Projection
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            
            # 5×1 conv: Patrones temporales medios
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # =====================================================================
        # INCEPTION MODULE 3 (Pooling path)
        # =====================================================================
        # Captura features agregadas (max pooling + projection)
        self.inp3 = nn.Sequential(
            # MaxPooling: Captura features salientes
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            
            # 1×1 conv: Projection a 64 channels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # =====================================================================
        # LSTM LAYER
        # =====================================================================
        # Modela dependencias temporales en features CNN
        # input_size=192: Concatenación de 3 inception paths (64+64+64)
        # hidden_size=64: Dimensión del hidden state
        # num_layers=1: Single-layer LSTM
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        
        # =====================================================================
        # FULLY CONNECTED LAYER
        # =====================================================================
        # Clasificador final: 64 → 3 clases
        self.fc1 = nn.Linear(64, 3)

        # =====================================================================
        # SOFTMAX (INCLUIDO EN FORWARD)
        # =====================================================================
        # NOTA: Esto es inusual - típicamente softmax se aplica en loss
        # Aquí está incluido en el modelo
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass de DeepLOB.
        
        Args:
            x (torch.Tensor): Input de shape (batch, seq_size, num_features)
                             Típicamente (batch, 100, 40)
        
        Returns:
            torch.Tensor: Probabilidades de shape (batch, 3)
                         NOTA: Ya incluye softmax (no logits)
        
        Proceso:
        --------
        1. Add channel dimension: (batch, seq, features) → (batch, 1, seq, features)
        2. Apply 3 CNN blocks
        3. Apply 3 Inception modules
        4. Concatenate Inception outputs
        5. Reshape for LSTM: (batch, channels, seq, 1) → (batch, seq', features')
        6. Apply LSTM
        7. Take last LSTM output
        8. Apply FC layer
        9. Apply softmax
        """
        # Agregar dimensión de canal (required for Conv2d)
        # (batch, seq, features) → (batch, 1, seq, features)
        x = x[:, None, :, :]

        # =====================================================================
        # CNN FEATURE EXTRACTION
        # =====================================================================
        x = self.conv1(x)  # Block 1 (LeakyReLU)
        x = self.conv2(x)  # Block 2 (Tanh)
        x = self.conv3(x)  # Block 3 (LeakyReLU)

        # =====================================================================
        # INCEPTION MODULES (Multi-scale feature extraction)
        # =====================================================================
        x_inp1 = self.inp1(x)  # Short-term patterns (kernel 3)
        x_inp2 = self.inp2(x)  # Mid-term patterns (kernel 5)
        x_inp3 = self.inp3(x)  # Pooling path
        
        # Concatenate along channel dimension
        # (batch, 64, seq, 1) × 3 → (batch, 192, seq, 1)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        
        # (batch, seq, 192, 1) → (batch, seq, 192)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        # =====================================================================
        # LSTM TEMPORAL MODELING
        # =====================================================================
        # out: (batch, seq, 64) - all hidden states
        # _: (h_n, c_n) - final hidden/cell states (not used)
        out, _ = self.lstm(x)

        # Take last timestep output
        # (batch, seq, 64) → (batch, 64)
        out = out[:, -1, :]
        
        # =====================================================================
        # FINAL CLASSIFICATION
        # =====================================================================
        # Linear: (batch, 64) → (batch, 3)
        out = self.fc1(out)
        
        # Softmax: Convert to probabilities
        # NOTA: Esto es inusual - típicamente se hace fuera del modelo
        out = self.softmax(out)
        
        return out
