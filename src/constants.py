"""
CONSTANTES Y CONFIGURACIONES GLOBALES DEL PROYECTO
===================================================

Este módulo define todas las constantes, enumeraciones y configuraciones
globales utilizadas en todo el proyecto TLOB.

Organización:
-------------
1. Enumeraciones (Enums) para tipos de datos, modelos y muestreo
2. Estadísticas precomputadas para normalización de LOBSTER datasets
3. Configuraciones generales del proyecto (paths, device, hyperparámetros)

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

import torch
from enum import Enum
from src.preprocessing.dataset import Dataset  


# =============================================================================
# SECCIÓN 1: ENUMERACIONES (ENUMS)
# =============================================================================

class DatasetType(Enum):
    """
    TIPOS DE DATASETS SOPORTADOS
    =============================
    
    Enumeración de los diferentes tipos de datasets de Limit Order Book
    que el modelo puede procesar.
    
    Valores:
    --------
    - **LOBSTER**: Formato LOBSTER (stocks estadounidenses)
      * Características: Datos event-driven de alta frecuencia
      * Features: Order book levels + metadata de eventos
      * Uso: Principalmente para acciones individuales
      * Ejemplo: AAPL, TSLA, INTC
    
    - **FI_2010**: Dataset FI-2010 (Finnish Stock Market)
      * Características: 5 acciones finlandesas, 10 niveles LOB
      * Features: Solo precios y volúmenes (40 features)
      * Uso: Benchmark académico estándar
      * Paper: https://arxiv.org/abs/1705.03233
    
    - **BTC**: Bitcoin/Criptomonedas (Binance)
      * Características: Datos de criptomonedas de alta frecuencia
      * Features: 10 niveles LOB (precios + volúmenes)
      * Uso: Trading de criptomonedas
      * Ejemplo: BTCUSDT, ETHUSDT
    
    Uso Típico:
    -----------
    ```python
    from src.constants import DatasetType
    
    # Configurar tipo de dataset
    dataset_type = DatasetType.BTC
    
    # Usar en condicionales
    if dataset_type == DatasetType.BTC:
        hidden_dim = 40
    elif dataset_type == DatasetType.FI_2010:
        hidden_dim = 144
    ```
    
    Nota:
    -----
    El tipo de dataset afecta:
    - Dimensiones del modelo (hidden_dim)
    - Preprocesamiento de datos
    - Normalización aplicada
    - Formato de entrada esperado
    """
    LOBSTER = "LOBSTER"
    FI_2010 = "FI_2010"
    BTC = "BTC"
    

class ModelType(Enum):
    """
    TIPOS DE MODELOS IMPLEMENTADOS
    ===============================
    
    Enumeración de las arquitecturas de modelos disponibles en el proyecto.
    
    Valores:
    --------
    - **TLOB** (Transformer for Limit Order Book) ⭐
      * Arquitectura: Transformer con Dual Attention
      * Parámetros: ~1.1M
      * Innovaciones: Atención spatial + temporal, BiN normalization
      * Performance: State-of-the-art en BTC y FI-2010
      * Paper: https://arxiv.org/pdf/2502.15757
    
    - **MLPLOB** (MLP for Limit Order Book)
      * Arquitectura: Multi-Layer Perceptron simple
      * Parámetros: ~2.8M
      * Uso: Baseline para comparación
      * Performance: Inferior a TLOB pero más rápido
    
    - **DEEPLOB** (Deep Learning for Limit Order Book)
      * Arquitectura: CNN + LSTM
      * Parámetros: ~4.2M
      * Uso: Modelo clásico de referencia
      * Performance: Bueno pero superado por TLOB
      * Paper: https://arxiv.org/abs/1808.03668
    
    - **BINCTABL** (Batch Independent Normalization + Tabular)
      * Arquitectura: MLP con BiN normalization
      * Parámetros: ~3.5M
      * Uso: Baseline con normalización mejorada
      * Performance: Bueno, pero sin modelado temporal explícito
    
    Uso Típico:
    -----------
    ```python
    from src.constants import ModelType
    
    # Seleccionar modelo
    model_type = ModelType.TLOB
    
    # Cargar checkpoint correspondiente
    checkpoint_path = f"src/data/checkpoints/{model_type.value}/..."
    ```
    
    Comparación Rápida:
    -------------------
    | Modelo     | Accuracy (BTC) | Parámetros | Ventaja Principal        |
    |------------|----------------|------------|--------------------------|
    | TLOB       | 71.2%          | 1.1M       | Dual attention, eficiente|
    | MLPLOB     | 70.1%          | 2.8M       | Simple, rápido           |
    | DEEPLOB    | 69.8%          | 4.2M       | Modelo de referencia     |
    | BINCTABL   | 68.5%          | 3.5M       | BiN normalization        |
    """
    MLPLOB = "MLPLOB"
    TLOB = "TLOB"
    BINCTABL = "BINCTABL"
    DEEPLOB = "DEEPLOB"
    

class SamplingType(Enum):
    """
    TIPOS DE MUESTREO PARA DATASETS LOBSTER
    ========================================
    
    Enumeración de estrategias de muestreo para datos event-driven (LOBSTER).
    
    En datasets event-driven, los eventos no están espaciados uniformemente en el tiempo.
    El muestreo permite crear snapshots regulares del LOB.
    
    Valores:
    --------
    - **TIME**: Muestreo basado en intervalos de tiempo
      * Descripción: Toma un snapshot cada N milisegundos
      * Parámetro: sampling_time (ms)
      * Ejemplo: sampling_time=250 → 1 snapshot cada 250ms (4 Hz)
      * Ventaja: Datos uniformes en el tiempo
      * Desventaja: Puede perder eventos importantes en períodos de baja actividad
      * Uso típico: Trading de alta frecuencia, análisis temporal
    
    - **QUANTITY**: Muestreo basado en número de eventos
      * Descripción: Toma un snapshot cada N eventos del order book
      * Parámetro: sampling_quantity (eventos)
      * Ejemplo: sampling_quantity=100 → 1 snapshot cada 100 eventos
      * Ventaja: Captura todos los eventos importantes
      * Desventaja: Timestamps no uniformes
      * Uso típico: Análisis event-driven, estrategias de momentum
    
    - **NONE**: Sin muestreo
      * Descripción: Usa datos tal como vienen (event-driven puro)
      * Parámetro: N/A
      * Ventaja: Máxima resolución temporal
      * Desventaja: Computacionalmente costoso, volumen de datos alto
      * Uso típico: Análisis de microestructura de mercado
    
    Uso Típico:
    -----------
    ```python
    from src.constants import SamplingType
    
    # Configurar muestreo por tiempo
    sampling_type = SamplingType.TIME
    sampling_time = 250  # ms
    
    # O muestreo por cantidad
    sampling_type = SamplingType.QUANTITY
    sampling_quantity = 100  # eventos
    
    # Aplicar en LOBSTERDataBuilder
    data_builder = LOBSTERDataBuilder(
        sampling_type=sampling_type,
        sampling_time=sampling_time
    )
    ```
    
    Ejemplo Práctico:
    -----------------
    **Datos originales (event-driven)**:
    ```
    t=0ms:    Event (order submission)
    t=5ms:    Event (order cancellation)
    t=12ms:   Event (trade)
    t=15ms:   Event (order submission)
    t=200ms:  Event (trade)
    t=205ms:  Event (order cancellation)
    ...
    ```
    
    **Después de TIME sampling (250ms)**:
    ```
    Snapshot 0: t=0ms    (LOB después del 1er evento)
    Snapshot 1: t=250ms  (LOB después del último evento antes de 250ms)
    Snapshot 2: t=500ms  (LOB en ese momento)
    ...
    ```
    
    **Después de QUANTITY sampling (100 eventos)**:
    ```
    Snapshot 0: t=??? (después de evento 100)
    Snapshot 1: t=??? (después de evento 200)
    Snapshot 2: t=??? (después de evento 300)
    ...
    ```
    
    Nota:
    -----
    - El muestreo SOLO aplica a datasets LOBSTER (event-driven)
    - BTC y FI-2010 ya vienen pre-sampleados
    - La elección afecta significativamente el tamaño del dataset
    """
    TIME = "time"
    QUANTITY = "quantity"
    NONE = "none"


# =============================================================================
# SECCIÓN 2: ESTADÍSTICAS PRECOMPUTADAS PARA NORMALIZACIÓN (LOBSTER)
# =============================================================================

"""
ESTADÍSTICAS DE TESLA (TSLA) - 15 DÍAS DE TRADING
==================================================

Estas constantes almacenan estadísticas precomputadas del LOB de TSLA
para normalización rápida durante entrenamiento/inferencia.

Uso:
----
En lugar de calcular mean/std cada vez, usamos estos valores precomputados:

```python
if stock == "TSLA":
    normalized_prices = (prices - TSLA_LOB_MEAN_PRICE_10) / TSLA_LOB_STD_PRICE_10
    normalized_sizes = (sizes - TSLA_LOB_MEAN_SIZE_10) / TSLA_LOB_STD_SIZE_10
```

Ventajas:
---------
- Normalización consistente entre train/val/test
- Más rápido (sin cálculo de estadísticas)
- Permite normalización de nuevos datos con mismas estadísticas
"""

# LOB (Limit Order Book) Statistics - TSLA
TSLA_LOB_MEAN_SIZE_10 = 165.44670902537212      # Tamaño promedio de órdenes (shares)
TSLA_LOB_STD_SIZE_10 = 481.7127061897184        # Desviación estándar de tamaños
TSLA_LOB_MEAN_PRICE_10 = 20180.439318660694     # Precio promedio (centavos, $201.80)
TSLA_LOB_STD_PRICE_10 = 814.8782058033195       # Desviación estándar de precios

# Event Statistics - TSLA
TSLA_EVENT_MEAN_SIZE = 88.09459295373463        # Tamaño promedio de eventos
TSLA_EVENT_STD_SIZE = 86.55913199110894         # Desviación estándar
TSLA_EVENT_MEAN_PRICE = 20178.610720500274      # Precio promedio de eventos
TSLA_EVENT_STD_PRICE = 813.8188032145645        # Desviación estándar
TSLA_EVENT_MEAN_TIME = 0.08644932804905886      # Tiempo promedio entre eventos (s)
TSLA_EVENT_STD_TIME = 0.3512181506722207        # Desviación estándar
TSLA_EVENT_MEAN_DEPTH = 7.365325300819055       # Profundidad promedio del LOB
TSLA_EVENT_STD_DEPTH = 8.59342838063813         # Desviación estándar

"""
ESTADÍSTICAS DE INTEL (INTC) - 15 DÍAS DE TRADING
==================================================

Misma estructura que TSLA, pero para la acción INTC.

Nota: INTC tiene volúmenes mucho mayores que TSLA (6222 vs 165 shares promedio).
Esto refleja diferencias en la estructura de mercado entre ambas acciones.
"""

# LOB (Limit Order Book) Statistics - INTC
INTC_LOB_MEAN_SIZE_10 = 6222.424274871972       # Tamaño promedio significativamente mayor
INTC_LOB_STD_SIZE_10 = 7538.341086370264        # Alta variabilidad
INTC_LOB_MEAN_PRICE_10 = 3635.766219937785      # Precio promedio (centavos, $36.36)
INTC_LOB_STD_PRICE_10 = 44.15649995373795       # Menor volatilidad que TSLA

# Event Statistics - INTC
INTC_EVENT_MEAN_SIZE = 324.6800802006092        # Tamaño promedio de eventos
INTC_EVENT_STD_SIZE = 574.5781447696605         # Desviación estándar
INTC_EVENT_MEAN_PRICE = 3635.78165265669        # Precio promedio de eventos
INTC_EVENT_STD_PRICE = 43.872407609651184       # Desviación estándar
INTC_EVENT_MEAN_TIME = 0.025201754040915927     # Eventos más frecuentes que TSLA
INTC_EVENT_STD_TIME = 0.11013627432323592       # Menor variabilidad temporal
INTC_EVENT_MEAN_DEPTH = 1.3685517399834501      # LOB menos profundo que TSLA
INTC_EVENT_STD_DEPTH = 2.333747222206966        # Desviación estándar


# =============================================================================
# SECCIÓN 3: CONFIGURACIONES GENERALES DEL PROYECTO
# =============================================================================

# HORIZONTES DE PREDICCIÓN
# -------------------------
LOBSTER_HORIZONS = [10, 20, 50, 100]
"""
Horizontes de predicción soportados (en timesteps).

Para LOBSTER con sampling de 250ms:
- 10 timesteps = 2.5 segundos
- 20 timesteps = 5 segundos
- 50 timesteps = 12.5 segundos
- 100 timesteps = 25 segundos

Para BTC con sampling de 250ms: Igual

Uso:
----
```python
for horizon in LOBSTER_HORIZONS:
    model = train_model(horizon=horizon)
    evaluate(model, horizon=horizon)
```
"""

# CONFIGURACIÓN DEL MODELO
# -------------------------
PRECISION = 32
"""
Precisión de punto flotante para entrenamiento (32 bits).

Opciones:
- 32: float32 (estándar, buen balance)
- 16: float16 (mixed precision, más rápido en GPUs modernas)
- 64: float64 (mayor precisión, más lento, raramente necesario)

Uso en PyTorch Lightning:
```python
trainer = L.Trainer(precision=PRECISION)
```
"""

# ESTRUCTURA DEL LIMIT ORDER BOOK
# --------------------------------
N_LOB_LEVELS = 10
"""
Número de niveles del Limit Order Book considerados.

Estructura típica:
```
Level 1: Best Bid/Ask (más cercano al mid-price)
Level 2: Segundo mejor Bid/Ask
...
Level 10: Décimo nivel (más alejado)
```

Para BTC:
- 10 niveles × 2 (bid/ask) × 2 (price/volume) = 40 features
"""

LEN_LEVEL = 4
"""
Longitud de cada nivel del LOB (número de features por nivel).

Estructura:
```
Level_i = [ASK_PRICE_i, ASK_VOLUME_i, BID_PRICE_i, BID_VOLUME_i]
```

Total features: N_LOB_LEVELS × LEN_LEVEL = 10 × 4 = 40
"""

LEN_ORDER = 6
"""
Longitud de metadata de orden para datasets LOBSTER.

Metadata de cada evento:
```
[event_type, order_id, size, price, direction, timestamp]
```

Solo usado en datasets LOBSTER event-driven, no en BTC/FI-2010.
"""

LEN_SMOOTH = 10
"""
Ventana de suavizado para cálculo de mid-price y etiquetas.

Uso en etiquetado:
```python
# Suavizar precios para reducir ruido
smoothed_prices = moving_average(prices, window=LEN_SMOOTH)

# Calcular cambios porcentuales
pct_change = (smoothed_prices[t+horizon] - smoothed_prices[t]) / smoothed_prices[t]
```

Un valor de 10 timesteps (2.5s para 250ms sampling) ayuda a:
- Filtrar ruido de alta frecuencia
- Capturar tendencias reales
- Evitar overfitting a micro-movimientos
"""

# CONFIGURACIÓN DE HARDWARE
# --------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
Device para entrenamiento e inferencia (GPU si está disponible, CPU si no).

Uso:
----
```python
model.to(DEVICE)
input_tensor = input_tensor.to(DEVICE)
```

Performance típica:
- GPU (RTX 3080): ~15ms por inferencia
- CPU (i7): ~50ms por inferencia
"""

# DIRECTORIOS DEL PROYECTO
# -------------------------
DIR_EXPERIMENTS = "src/data/experiments"
"""
Directorio para guardar resultados de experimentos y logs.

Estructura típica:
```
src/data/experiments/
├── TLOB_BTC_horizon_10/
│   ├── logs.txt
│   ├── metrics.json
│   └── plots/
└── TLOB_FI2010_horizon_20/
    └── ...
```
"""

DIR_SAVED_MODEL = "src/data/checkpoints"
"""
Directorio para guardar checkpoints de modelos entrenados.

Estructura:
```
src/data/checkpoints/
├── TLOB/
│   └── BTC_seq_size_128_horizon_10_seed_42/
│       ├── pt/
│       │   └── val_loss=0.624_epoch=2.pt  ⭐ Modelo entrenado
│       └── predictions.npy
├── DEEPLOB/
└── MLPLOB/
```

Formato de nombre:
{DATASET}_{seq_size}_{horizon}_seed_{seed}
"""

DATA_DIR = "src/data"
"""
Directorio raíz de todos los datos.

Estructura completa:
```
src/data/
├── checkpoints/        (modelos entrenados)
├── experiments/        (logs y resultados)
├── BTC/               (datos Bitcoin)
│   ├── train.npy
│   ├── val.npy
│   ├── test.npy
│   ├── individual_examples/
│   └── raw_examples/
├── FI_2010/           (datos FI-2010)
└── LOBSTER/           (datos LOBSTER, si aplica)
    ├── TSLA/
    └── INTC/
```
"""

RECON_DIR = "src/data/reconstructions"
"""
Directorio para reconstrucciones (usado en modelos autoencoder, si aplica).

En este proyecto, mayormente sin uso, pero reservado para extensiones futuras.
"""

# CONFIGURACIÓN DE WANDB (WEIGHTS & BIASES)
# ------------------------------------------
PROJECT_NAME = "EvolutionData"
"""
Nombre del proyecto en Weights & Biases para logging de experimentos.

Uso:
----
```python
import wandb
wandb.init(project=PROJECT_NAME, name="TLOB_BTC_h10")
```

Permite:
- Tracking de métricas en tiempo real
- Comparación de experimentos
- Visualización de gráficos
- Almacenamiento de artefactos
"""

WANDB_API = ""
"""
API key de Weights & Biases (dejar vacío si no se usa).

Configurar:
-----------
```python
import wandb
wandb.login(key=WANDB_API)
```

O mediante variable de entorno:
```bash
export WANDB_API_KEY="tu_api_key_aqui"
```

Nota: Por seguridad, NO incluir la API key real en el código.
Usar variables de entorno o archivos de configuración privados.
"""

WANDB_USERNAME = ""
"""
Username de Weights & Biases (opcional).

Uso:
----
```python
wandb.init(project=PROJECT_NAME, entity=WANDB_USERNAME)
```

Permite compartir experimentos con el equipo.
"""

# SPLIT DE DATASETS
# ------------------
SPLIT_RATES = [0.8, 0.1, 0.1]
"""
Proporciones de split para train/val/test.

Valores:
--------
- Train: 80% de los datos (entrenamiento del modelo)
- Val: 10% de los datos (validación durante entrenamiento, early stopping)
- Test: 10% de los datos (evaluación final, NO TOCADO durante entrenamiento)

Uso:
----
```python
train_size = int(len(dataset) * SPLIT_RATES[0])
val_size = int(len(dataset) * SPLIT_RATES[1])
test_size = len(dataset) - train_size - val_size

train_data = dataset[:train_size]
val_data = dataset[train_size:train_size+val_size]
test_data = dataset[train_size+val_size:]
```

Nota Importante:
----------------
Para series temporales, el split debe ser CRONOLÓGICO (no aleatorio):
- Train: datos más antiguos (ej: Enero-Febrero)
- Val: datos intermedios (ej: Marzo)
- Test: datos más recientes (ej: Abril)

Esto evita data leakage y simula condiciones reales de trading.
"""
