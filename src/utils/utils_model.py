"""
UTILIDADES PARA MODELOS
========================

Funciones auxiliares para instanciación y gestión de modelos.

Este módulo proporciona factory functions para crear instancias de los diferentes
modelos implementados en el proyecto, abstrayendo la lógica de inicialización
específica de cada arquitectura.

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

from src.models.mlplob import MLPLOB
from src.models.tlob import TLOB
from src.models.binctabl import BiN_CTABL
from src.models.deeplob import DeepLOB
from transformers import AutoModelForSeq2SeqLM


def pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, 
               num_heads=8, is_sin_emb=False, dataset_type=None):
    """
    FACTORY FUNCTION PARA CREAR INSTANCIAS DE MODELOS
    =================================================
    
    Crea y retorna una instancia del modelo especificado con los hiperparámetros dados.
    Actúa como un factory pattern que encapsula la lógica de inicialización
    específica de cada arquitectura.
    
    Args:
        model_type (str): Tipo de modelo a instanciar
                         Valores válidos: "MLPLOB", "TLOB", "BINCTABL", "DEEPLOB"
        
        hidden_dim (int): Dimensión del espacio latente de embeddings
                         - TLOB (BTC): 40
                         - TLOB (FI-2010): 144
                         - MLPLOB: Variable según dataset
                         - BINCTABL: No usado (usa 60 fijo)
                         - DEEPLOB: No usado
        
        num_layers (int): Número de capas (transformers/MLPs)
                         - TLOB: 4 (pares) = 8 capas totales
                         - MLPLOB: 3-5 típicamente
                         - BINCTABL: No usado
                         - DEEPLOB: No usado (arquitectura fija)
        
        seq_size (int): Longitud de la secuencia temporal
                       - BTC: 128 timesteps
                       - FI-2010: 100 timesteps
                       - LOBSTER: Variable (100-200)
        
        num_features (int): Número de features del input LOB
                           - BTC: 40 (10 niveles × 4)
                           - FI-2010: 40
                           - LOBSTER: 46 (con metadata adicional)
        
        num_heads (int, optional): Número de cabezas de atención (solo TLOB)
                                  - Default: 8
                                  - BTC: 1 (simplificado)
                                  - FI-2010: 8 (multi-head completo)
        
        is_sin_emb (bool, optional): Usar positional encoding sinusoidal (solo TLOB)
                                     - True: Sinusoidal (como "Attention is All You Need")
                                     - False: Aprendible (trainable parameters)
                                     - Default: False
        
        dataset_type (str, optional): Tipo de dataset ("BTC", "FI_2010", "LOBSTER")
                                     - Usado por TLOB y MLPLOB para ajustes específicos
                                     - LOBSTER requiere embedding adicional para order_type
    
    Returns:
        nn.Module: Instancia del modelo PyTorch especificado
    
    Raises:
        ValueError: Si model_type no es válido
    
    Uso Típico:
    -----------
    ### Ejemplo 1: Crear modelo TLOB para BTC
    ```python
    from src.utils.utils_model import pick_model
    
    model = pick_model(
        model_type="TLOB",
        hidden_dim=40,
        num_layers=4,      # 4 pares = 8 capas totales
        seq_size=128,
        num_features=40,
        num_heads=1,       # BTC usa single-head
        is_sin_emb=True,   # Encoding sinusoidal
        dataset_type="BTC"
    )
    
    print(f"Modelo TLOB creado con {sum(p.numel() for p in model.parameters())} parámetros")
    # Output: Modelo TLOB creado con ~1,100,000 parámetros
    ```
    
    ### Ejemplo 2: Crear modelo TLOB para FI-2010
    ```python
    model = pick_model(
        model_type="TLOB",
        hidden_dim=144,    # Mayor dim para FI-2010
        num_layers=4,
        seq_size=100,
        num_features=40,
        num_heads=8,       # Multi-head completo
        is_sin_emb=False,  # Encoding aprendible
        dataset_type="FI_2010"
    )
    ```
    
    ### Ejemplo 3: Crear modelo MLPLOB
    ```python
    model = pick_model(
        model_type="MLPLOB",
        hidden_dim=256,
        num_layers=3,
        seq_size=128,
        num_features=40,
        dataset_type="BTC"
    )
    # num_heads e is_sin_emb son ignorados para MLPLOB
    ```
    
    ### Ejemplo 4: Comparar múltiples modelos
    ```python
    models = {}
    for model_type in ["TLOB", "MLPLOB", "DEEPLOB", "BINCTABL"]:
        models[model_type] = pick_model(
            model_type=model_type,
            hidden_dim=40,
            num_layers=4,
            seq_size=128,
            num_features=40,
            dataset_type="BTC"
        )
    
    # Evaluar todos los modelos en el mismo dataset
    for name, model in models.items():
        accuracy = evaluate(model, test_loader)
        print(f"{name}: {accuracy:.2f}%")
    ```
    
    Detalles de Inicialización por Modelo:
    ---------------------------------------
    
    ### MLPLOB (Multi-Layer Perceptron for LOB)
    ```python
    MLPLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type)
    ```
    - Arquitectura: MLP simple con capas fully-connected
    - Input: Flattened sequence (seq_size × num_features)
    - Parámetros típicos: ~2.8M
    - Uso: Baseline simple y rápido
    
    ### TLOB (Transformer for LOB) ⭐
    ```python
    TLOB(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    ```
    - Arquitectura: Transformer con Dual Attention
    - Input: Sequence (batch, seq_size, num_features)
    - Parámetros típicos: ~1.1M (BTC), ~5M (FI-2010)
    - Uso: Modelo principal del proyecto, state-of-the-art
    - Innovaciones: Atención spatial + temporal, BiN normalization
    
    ### BINCTABL (Batch Independent Normalization + Tabular)
    ```python
    BiN_CTABL(60, num_features, seq_size, seq_size, 120, 5, 3, 1)
    ```
    - Arquitectura: MLP con BiN normalization
    - Input: Tabular features del LOB
    - Parámetros típicos: ~3.5M
    - Uso: Baseline con normalización mejorada
    - Nota: Inicialización con parámetros fijos (60, 120, 5, 3, 1)
    
    ### DEEPLOB (Deep Learning for LOB)
    ```python
    DeepLOB()
    ```
    - Arquitectura: CNN + LSTM (arquitectura fija)
    - Input: Sequence (batch, seq_size, num_features)
    - Parámetros típicos: ~4.2M
    - Uso: Modelo de referencia clásico
    - Paper: https://arxiv.org/abs/1808.03668
    - Nota: No requiere hiperparámetros (arquitectura predefinida)
    
    Flujo de Decisión:
    ------------------
    ```
    model_type input
         │
         ├─> "MLPLOB" ──> MLPLOB(hidden_dim, ...)
         │
         ├─> "TLOB" ────> TLOB(hidden_dim, num_heads, is_sin_emb, ...)
         │
         ├─> "BINCTABL" ─> BiN_CTABL(60 fixed, ...)
         │
         ├─> "DEEPLOB" ──> DeepLOB() [no params]
         │
         └─> other ─────> ValueError
    ```
    
    Errores Comunes:
    ----------------
    1. **ValueError: Model not found**
       - Causa: model_type no es uno de los valores válidos
       - Solución: Usar "MLPLOB", "TLOB", "BINCTABL", o "DEEPLOB"
    
    2. **RuntimeError: size mismatch**
       - Causa: Hiperparámetros inconsistentes con checkpoint cargado
       - Solución: Verificar que hidden_dim, num_layers, etc. coincidan
    
    3. **TypeError: missing required argument**
       - Causa: No se pasaron todos los argumentos requeridos
       - Solución: Asegurar que model_type, hidden_dim, etc. están definidos
    
    Integración con Training Pipeline:
    ----------------------------------
    Esta función típicamente se usa en conjunto con el Engine de entrenamiento:
    
    ```python
    from src.models.engine import Engine
    from src.utils.utils_model import pick_model
    
    # Crear modelo
    model = pick_model(
        model_type="TLOB",
        hidden_dim=40,
        num_layers=4,
        seq_size=128,
        num_features=40,
        num_heads=1,
        is_sin_emb=True,
        dataset_type="BTC"
    )
    
    # Envolver en Engine para entrenamiento
    engine = Engine(
        model=model,
        lr=1e-4,
        optimizer="Adam",
        # ... otros parámetros de entrenamiento
    )
    
    # Entrenar
    trainer.fit(engine, train_loader, val_loader)
    ```
    
    Nota sobre Compatibilidad:
    --------------------------
    - Todos los modelos retornados implementan nn.Module de PyTorch
    - Todos tienen método forward(input) que retorna logits de shape (batch, 3)
    - Las 3 clases de salida son: [DOWN, STATIONARY, UP] (o [0, 1, 2])
    - Todos son compatibles con PyTorch Lightning para entrenamiento
    
    Referencias:
    ------------
    - TLOB Paper: https://arxiv.org/pdf/2502.15757
    - DeepLOB Paper: https://arxiv.org/abs/1808.03668
    - Implementaciones: src/models/*.py
    """
    if model_type == "MLPLOB":
        return MLPLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type)
    
    elif model_type == "TLOB":
        return TLOB(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    
    elif model_type == "BINCTABL":
        # Nota: BiN_CTABL usa hiperparámetros fijos independientes de los inputs
        # Arquitectura: BiN_CTABL(hidden_dim, num_features, seq_size, seq_size, hidden_dim*2, 5, 3, 1)
        return BiN_CTABL(60, num_features, seq_size, seq_size, 120, 5, 3, 1)
    
    elif model_type == "DEEPLOB":
        # Nota: DeepLOB no requiere hiperparámetros (arquitectura fija del paper)
        return DeepLOB()
    
    else:
        raise ValueError(
            f"Model type '{model_type}' not found. "
            f"Valid options: 'MLPLOB', 'TLOB', 'BINCTABL', 'DEEPLOB'"
        )
