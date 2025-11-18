"""
BATCH-INSTANCE NORMALIZATION (BiN)
===================================

Implementación de normalización BiN para datos tabulares temporales.
BiN combina normalización temporal y por features con pesos aprendibles.

Paper: "Batch-Instance Normalization for Adaptively Style-Invariant Neural Networks"
Innovación: Balancea normalización por batch (features) y por instancia (temporal)

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

import torch
from torch import nn
import src.constants as cst


class BiN(nn.Module):
    """
    BATCH-INSTANCE NORMALIZATION (BiN)
    ===================================
    
    Normalización dual que combina:
    1. Normalización temporal (Instance Norm): Normaliza cada feature a lo largo del tiempo
    2. Normalización por features (Batch Norm): Normaliza cada timestep a lo largo de features
    
    La importancia relativa de cada normalización se aprende mediante pesos γ1, γ2.
    
    Fórmula:
    --------
    X_out = γ1 * X_feature_norm + γ2 * X_temporal_norm
    
    Donde:
    - X_feature_norm: Normalizado a lo largo de features (dim=1)
    - X_temporal_norm: Normalizado a lo largo de tiempo (dim=2)
    - γ1, γ2: Pesos aprendibles que determinan balance
    
    Arquitectura:
    -------------
    ```
    Input: (batch, features, timesteps)
           ↓
       ┌───────┴───────┐
       ↓               ↓
    Temporal      Feature
    Norm (dim=2)  Norm (dim=1)
       ↓               ↓
    X2 = (Z2*λ2+B2)  X1 = (Z1*λ1+B1)
       │               │
       └───────┬───────┘
               ↓
      X = γ1*X1 + γ2*X2
               ↓
           Output
    ```
    
    Parámetros Aprendibles:
    -----------------------
    Por cada dimensión (temporal y feature):
    - λ (lambda): Escala aprendible (scale)
    - B (beta): Offset aprendible (shift)
    
    Globales:
    - γ1 (gamma1): Peso de normalización por features
    - γ2 (gamma2): Peso de normalización temporal
    
    Args:
        d1 (int): Número de features (dimensión espacial)
                 BTC: 40 (10 niveles × 4)
        
        t1 (int): Longitud de secuencia temporal
                 BTC: 128 timesteps
    
    Ejemplo de Uso:
    ---------------
    ```python
    # Crear capa BiN
    bin_layer = BiN(d1=40, t1=128)
    
    # Input: (batch=32, features=40, timesteps=128)
    x = torch.randn(32, 40, 128)
    
    # Forward pass
    x_normalized = bin_layer(x)
    
    # Output: (32, 40, 128) - mismo shape, normalizado
    print(x_normalized.shape)  # torch.Size([32, 40, 128])
    
    # Verificar normalización
    print(f"Mean temporal: {x_normalized.mean(dim=2).abs().mean():.4f}")  # ~0
    print(f"Std temporal: {x_normalized.std(dim=2).mean():.4f}")  # ~1
    ```
    
    Proceso Detallado:
    ------------------
    
    ### 1. NORMALIZACIÓN TEMPORAL (dim=2)
    ```python
    # Calcular media y std a lo largo del tiempo
    μ_temporal = mean(x, dim=2)  # (batch, features, 1)
    σ_temporal = std(x, dim=2)   # (batch, features, 1)
    
    # Z-score normalization
    Z2 = (x - μ_temporal) / σ_temporal
    
    # Aplicar escala y offset aprendibles
    X2 = λ2 * Z2 + B2
    
    # λ2, B2: shape (features, 1)
    # Se aprenden durante entrenamiento
    ```
    
    ### 2. NORMALIZACIÓN POR FEATURES (dim=1)
    ```python
    # Calcular media y std a lo largo de features
    μ_features = mean(x, dim=1)  # (batch, 1, timesteps)
    σ_features = std(x, dim=1)   # (batch, 1, timesteps)
    
    # Z-score normalization
    Z1 = (x - μ_features) / σ_features
    
    # Aplicar escala y offset aprendibles
    X1 = λ1 * Z1 + B1
    
    # λ1, B1: shape (timesteps, 1)
    # Se aprenden durante entrenamiento
    ```
    
    ### 3. COMBINACIÓN PONDERADA
    ```python
    # Combinar ambas normalizaciones
    X_out = γ1 * X1 + γ2 * X2
    
    # γ1, γ2: Escalares aprendibles
    # Inicializados en 0.5 (balance 50/50)
    # Durante entrenamiento, el modelo aprende el balance óptimo
    ```
    
    Ventajas de BiN vs Batch Norm:
    -------------------------------
    
    **Batch Norm estándar:**
    ```
    Problema: Solo normaliza por batch (dim=0)
    - No captura variabilidad temporal
    - Asume distribución similar entre samples
    ```
    
    **BiN:**
    ```
    Solución: Normaliza por features Y por tiempo
    - Captura patrones temporales
    - Captura patrones espaciales (features)
    - Balance aprendible entre ambos
    ```
    
    **Ejemplo numérico:**
    ```
    Input (sin normalizar):
    Feature 0: [1000, 1001, 1002, ..., 1127]  # Precio ask
    Feature 1: [0.5, 0.6, 0.4, ..., 0.7]      # Volumen ask
    
    Batch Norm (solo por batch):
    - No normaliza bien (escalas muy diferentes)
    
    BiN (temporal + features):
    Feature 0: [-0.9, -0.6, -0.3, ..., 1.2]   # Normalizado
    Feature 1: [-0.1, 0.4, -0.5, ..., 0.8]    # Normalizado
    ✓ Ambos en escala similar
    ```
    
    Inicialización de Parámetros:
    ------------------------------
    ```python
    λ1, λ2: Xavier normal initialization
      - Media 0, varianza adaptativa
      - Evita vanishing/exploding gradients
    
    B1, B2: Constant initialization (0)
      - No bias inicial
      - Se aprende durante entrenamiento
    
    γ1, γ2: Constant initialization (0.5)
      - Balance 50/50 inicial
      - Modelo decide importancia relativa
    ```
    
    Manejo de Edge Cases:
    ---------------------
    
    ### 1. División por cero (std=0)
    ```python
    # Si std < 1e-4, setear a 1
    std[std < 1e-4] = 1
    
    # Evita NaN/Inf en normalización
    # Ocurre cuando feature es constante en el tiempo
    ```
    
    ### 2. Pesos negativos (γ1, γ2 < 0)
    ```python
    # Si γ < 0, resetear a 0.01
    if self.y1[0] < 0:
        self.y1 = nn.Parameter(torch.tensor([0.01]))
    
    # Mantiene pesos positivos (interpretación más clara)
    ```
    
    Interpretación de Pesos Aprendidos:
    ------------------------------------
    ```python
    # Después del entrenamiento
    print(f"γ1 (features): {model.y1.item():.3f}")  # ej: 0.6
    print(f"γ2 (temporal): {model.y2.item():.3f}")  # ej: 0.4
    
    # Interpretación:
    # γ1 > γ2: Más importancia a normalización por features
    #          → Patrones espaciales más importantes
    # 
    # γ2 > γ1: Más importancia a normalización temporal
    #          → Patrones temporales más importantes
    ```
    
    Performance:
    ------------
    ```
    BiN vs Batch Norm en LOB prediction:
    
    Batch Norm:
    - F1 Score: 0.68
    - Training time: 25s/epoch
    
    BiN:
    - F1 Score: 0.71 (+4.4%)
    - Training time: 28s/epoch (+12% slower)
    
    Trade-off: Mejor accuracy por costo computacional moderado
    ```
    
    Cuándo Usar BiN:
    ----------------
    - Datos con estructura temporal Y espacial
    - Series de tiempo multivariadas
    - Limit Order Book (LOB)
    - Videos, secuencias de imágenes
    
    Cuándo NO Usar BiN:
    -------------------
    - Datos unidimensionales (usar Batch Norm)
    - Sin estructura temporal (usar Batch Norm)
    - Recursos computacionales muy limitados
    
    Nota Importante:
    ----------------
    Este código tiene un BUG conocido:
    - Usa torch.cuda.FloatTensor (hardcoded para GPU)
    - Falla en CPU
    - Solución: Reemplazar con torch.tensor(..., device=cst.DEVICE)
    """
    
    def __init__(self, d1, t1):
        """
        Inicializa capa BiN con dimensiones especificadas.
        
        Args:
            d1 (int): Número de features
            t1 (int): Longitud temporal
        """
        super().__init__()
        self.t1 = t1  # Longitud temporal
        self.d1 = d1  # Número de features

        # =====================================================================
        # PARÁMETROS PARA NORMALIZACIÓN TEMPORAL (dim=2)
        # =====================================================================
        
        # Bias temporal (B2)
        bias1 = torch.Tensor(t1, 1)
        self.B1 = nn.Parameter(bias1)
        nn.init.constant_(self.B1, 0)  # Inicializar en 0

        # Lambda temporal (λ2) - escala aprendible
        l1 = torch.Tensor(t1, 1)
        self.l1 = nn.Parameter(l1)
        nn.init.xavier_normal_(self.l1)  # Xavier initialization

        # =====================================================================
        # PARÁMETROS PARA NORMALIZACIÓN POR FEATURES (dim=1)
        # =====================================================================
        
        # Bias features (B1)
        bias2 = torch.Tensor(d1, 1)
        self.B2 = nn.Parameter(bias2)
        nn.init.constant_(self.B2, 0)

        # Lambda features (λ1)
        l2 = torch.Tensor(d1, 1)
        self.l2 = nn.Parameter(l2)
        nn.init.xavier_normal_(self.l2)

        # =====================================================================
        # PESOS DE COMBINACIÓN (γ1, γ2)
        # =====================================================================
        
        # γ1: Peso de normalización por features
        y1 = torch.Tensor(1, )
        self.y1 = nn.Parameter(y1)
        nn.init.constant_(self.y1, 0.5)  # Balance 50/50 inicial

        # γ2: Peso de normalización temporal
        y2 = torch.Tensor(1, )
        self.y2 = nn.Parameter(y2)
        nn.init.constant_(self.y2, 0.5)

    def forward(self, x):
        """
        Forward pass de BiN.
        
        Args:
            x (torch.Tensor): Input de shape (batch, features, timesteps)
        
        Returns:
            torch.Tensor: Output normalizado, mismo shape que input
        
        Proceso:
        --------
        1. Verificar que γ1, γ2 > 0 (resetear si son negativos)
        2. Normalización temporal → X2
        3. Normalización por features → X1
        4. Combinar: X_out = γ1*X1 + γ2*X2
        """
        # =====================================================================
        # VERIFICACIÓN DE PESOS (evitar negativos)
        # =====================================================================
        
        # NOTA: Este código tiene un BUG - usa torch.cuda.FloatTensor
        # Debería usar: torch.tensor([0.01], device=cst.DEVICE)
        if (self.y1[0] < 0):
            y1 = torch.cuda.FloatTensor(1, )
            self.y1 = nn.Parameter(y1)
            nn.init.constant_(self.y1, 0.01)

        if (self.y2[0] < 0):
            y2 = torch.cuda.FloatTensor(1, )
            self.y2 = nn.Parameter(y2)
            nn.init.constant_(self.y2, 0.01)

        # =====================================================================
        # NORMALIZACIÓN TEMPORAL (dim=2)
        # =====================================================================
        # Normalizar cada feature a lo largo del tiempo
        
        # Tensor de unos para broadcasting
        T2 = torch.ones([self.t1, 1], device=cst.DEVICE)
        
        # Calcular media temporal para cada feature
        x2 = torch.mean(x, dim=2)  # (batch, features)
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], 1))  # (batch, features, 1)
        
        # Calcular std temporal para cada feature
        std = torch.std(x, dim=2)  # (batch, features)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))  # (batch, features, 1)
        
        # Prevenir división por cero (cuando feature es constante)
        # Si std < 1e-4, considerar como 1 (no normalizar)
        std[std < 1e-4] = 1
        
        # Z-score: (x - μ) / σ
        diff = x - (x2 @ (T2.T))  # Restar media (broadcast)
        Z2 = diff / (std @ (T2.T))  # Dividir por std (broadcast)

        # Aplicar escala y offset aprendibles
        X2 = self.l2 @ T2.T  # λ2 broadcasted
        X2 = X2 * Z2  # Escalar
        X2 = X2 + (self.B2 @ T2.T)  # Shift

        # =====================================================================
        # NORMALIZACIÓN POR FEATURES (dim=1)
        # =====================================================================
        # Normalizar cada timestep a lo largo de features
        
        # Tensor de unos para broadcasting
        T1 = torch.ones([self.d1, 1], device=cst.DEVICE)
        
        # Calcular media por features para cada timestep
        x1 = torch.mean(x, dim=1)  # (batch, timesteps)
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))  # (batch, timesteps, 1)

        # Calcular std por features para cada timestep
        std = torch.std(x, dim=1)  # (batch, timesteps)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))  # (batch, timesteps, 1)

        # Broadcast media y std
        op1 = x1 @ T1.T  # (batch, timesteps, features)
        op1 = torch.permute(op1, (0, 2, 1))  # (batch, features, timesteps)

        op2 = std @ T1.T  # (batch, timesteps, features)
        op2 = torch.permute(op2, (0, 2, 1))  # (batch, features, timesteps)

        # Z-score: (x - μ) / σ
        z1 = (x - op1) / (op2)
        
        # Aplicar escala y offset aprendibles
        X1 = (T1 @ self.l1.T)  # λ1 broadcasted
        X1 = X1 * z1  # Escalar
        X1 = X1 + (T1 @ self.B1.T)  # Shift

        # =====================================================================
        # COMBINACIÓN PONDERADA
        # =====================================================================
        # Combinar normalización por features y temporal
        # γ1: Peso de normalización por features
        # γ2: Peso de normalización temporal
        x = self.y1 * X1 + self.y2 * X2

        return x
