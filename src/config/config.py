"""
CONFIGURACIÓN DEL PROYECTO CON HYDRA
=====================================

Define todas las configuraciones del proyecto usando Hydra y dataclasses.
Permite configuración flexible mediante archivos YAML y override por CLI.

Hydra permite:
- Configuración estructurada y type-safe
- Override de parámetros desde CLI
- Configuraciones composicionales
- Experimentos reproducibles

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

from typing import List
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from src.constants import DatasetType, ModelType, SamplingType
from omegaconf import MISSING, OmegaConf


# =============================================================================
# CONFIGURACIONES DE MODELOS
# =============================================================================

@dataclass
class Model:
    """
    CLASE BASE PARA CONFIGURACIÓN DE MODELOS
    =========================================
    
    Define la estructura común para todos los modelos.
    
    Atributos:
        hyperparameters_fixed (dict): Hiperparámetros fijos para entrenamiento normal
        hyperparameters_sweep (dict): Rango de hiperparámetros para sweep/tuning
        type (ModelType): Tipo de modelo (TLOB, MLPLOB, etc.)
    
    Uso:
    ----
    Esta es una clase abstracta, no se instancia directamente.
    Usar subclases específicas: TLOB, MLPLOB, etc.
    """
    hyperparameters_fixed: dict = MISSING
    hyperparameters_sweep: dict = MISSING
    type: ModelType = MISSING
    

@dataclass
class MLPLOB(Model):
    """
    CONFIGURACIÓN DEL MODELO MLPLOB
    ================================
    
    Multi-Layer Perceptron for Limit Order Book.
    Arquitectura: MLP simple con capas fully-connected.
    
    Hiperparámetros Fijos (entrenamiento normal):
    ----------------------------------------------
    - num_layers: 3
      * Número de capas MLP
      * Más capas = más capacidad pero mayor riesgo de overfitting
    
    - hidden_dim: 40
      * Dimensión de capas ocultas
      * Típicamente igual o menor que num_features para evitar overfitting
    
    - lr: 0.0003 (learning rate)
      * Tasa de aprendizaje para Adam optimizer
      * Valor relativamente alto para MLP (converge rápido)
    
    - seq_size: 384
      * Longitud de secuencia temporal
      * MLP usa seq_size más grande que TLOB (384 vs 128)
      * Compensa falta de atención con más contexto
    
    - all_features: True
      * Usar todos los features del LOB (40)
      * False usaría subset reducido
    
    Hiperparámetros Sweep (tuning con WandB):
    -----------------------------------------
    - num_layers: [3, 6]
      * Probar arquitecturas de 3 y 6 capas
    
    - hidden_dim: [128]
      * Dimensión oculta fija en sweep
    
    - lr: [0.0003]
      * Learning rate fijo en sweep
    
    - seq_size: [384]
      * Secuencia fija en sweep
    
    Uso con Hydra:
    --------------
    ```bash
    # Usar configuración default
    python src/main.py model=mlplob dataset=btc
    
    # Override hiperparámetros desde CLI
    python src/main.py model=mlplob model.hyperparameters_fixed.num_layers=6
    
    # Ejecutar sweep
    python src/main.py model=mlplob experiment.is_sweep=true
    ```
    
    Ejemplo en Python:
    ------------------
    ```python
    from src.config.config import MLPLOB
    
    config = MLPLOB()
    print(config.hyperparameters_fixed)
    # {'num_layers': 3, 'hidden_dim': 40, 'lr': 0.0003, 
    #  'seq_size': 384, 'all_features': True}
    ```
    """
    hyperparameters_fixed: dict = field(default_factory=lambda: {
        "num_layers": 3, 
        "hidden_dim": 40, 
        "lr": 0.0003, 
        "seq_size": 384, 
        "all_features": True
    })
    hyperparameters_sweep: dict = field(default_factory=lambda: {
        "num_layers": [3, 6], 
        "hidden_dim": [128], 
        "lr": [0.0003], 
        "seq_size": [384]
    })
    type: ModelType = ModelType.MLPLOB
    

@dataclass
class TLOB(Model):
    """
    CONFIGURACIÓN DEL MODELO TLOB ⭐
    ================================
    
    Transformer for Limit Order Book - Modelo principal del proyecto.
    Arquitectura: Transformer con Dual Attention (spatial + temporal).
    
    Hiperparámetros Fijos (entrenamiento normal):
    ----------------------------------------------
    - num_layers: 4
      * Número de PARES de Transformer layers
      * 4 pares = 8 capas totales (alternando spatial/temporal)
      * Más capas = mejor performance hasta 4 pares, después overfitting
    
    - hidden_dim: 40
      * Dimensión del espacio latente
      * BTC: 40 (igual a num_features)
      * FI-2010: 144 (expandido)
      * Se ajusta automáticamente en main.py según dataset
    
    - num_heads: 1
      * Número de cabezas de atención
      * BTC: 1 (simplificado)
      * FI-2010: 8 (multi-head completo)
    
    - is_sin_emb: True
      * Usar positional encoding sinusoidal (True) o aprendible (False)
      * True: Como paper "Attention is All You Need"
      * False: Embeddings aprendibles (más parámetros)
    
    - lr: 0.0001 (learning rate)
      * Tasa de aprendizaje para Adam optimizer
      * Más bajo que MLPLOB (Transformers requieren lr más bajo)
    
    - seq_size: 128
      * Longitud de secuencia temporal
      * BTC: 128 timesteps = 32 segundos @ 250ms
      * Más corto que MLPLOB porque atención captura mejor dependencias
    
    - all_features: True
      * Usar todos los 40 features del LOB
    
    Hiperparámetros Sweep (tuning con WandB):
    -----------------------------------------
    - num_layers: [4, 6]
      * Probar 4 y 6 pares de capas
    
    - hidden_dim: [128, 256]
      * Probar dimensiones latentes mayores
    
    - num_heads: [1]
      * Cabezas fijas en 1 para sweep
    
    - is_sin_emb: [True]
      * Encoding sinusoidal fijo
    
    - lr: [0.0001]
      * Learning rate fijo
    
    - seq_size: [128]
      * Secuencia fija
    
    Uso con Hydra:
    --------------
    ```bash
    # Configuración default para BTC
    python src/main.py model=tlob dataset=btc
    
    # BTC con 6 pares de capas
    python src/main.py model=tlob model.hyperparameters_fixed.num_layers=6
    
    # FI-2010 con multi-head attention
    python src/main.py model=tlob dataset=fi_2010 \
        model.hyperparameters_fixed.num_heads=8
    
    # Ejecutar hyperparameter sweep
    python src/main.py model=tlob experiment.is_sweep=true
    ```
    
    Nota Importante:
    ----------------
    hidden_dim se ajusta automáticamente en main.py:
    - BTC: 40
    - FI-2010: 144
    - LOBSTER: 46
    
    Esto es por compatibilidad con arquitectura específica de cada dataset.
    """
    hyperparameters_fixed: dict = field(default_factory=lambda: {
        "num_layers": 4, 
        "hidden_dim": 40, 
        "num_heads": 1, 
        "is_sin_emb": True, 
        "lr": 0.0001, 
        "seq_size": 128, 
        "all_features": True
    })
    hyperparameters_sweep: dict = field(default_factory=lambda: {
        "num_layers": [4, 6], 
        "hidden_dim": [128, 256], 
        "num_heads": [1], 
        "is_sin_emb": [True], 
        "lr": [0.0001], 
        "seq_size": [128]
    })
    type: ModelType = ModelType.TLOB
    

@dataclass
class BiNCTABL(Model):
    """
    CONFIGURACIÓN DEL MODELO BINCTABL
    ==================================
    
    Batch Independent Normalization + Tabular features.
    Arquitectura: MLP con BiN normalization.
    
    Hiperparámetros Fijos:
    ----------------------
    - lr: 0.001
      * Learning rate más alto que TLOB (modelo más simple)
    
    - seq_size: 10
      * Secuencia MUY corta (solo mira 10 timesteps)
      * Modelo diseñado para predicción de corto plazo
    
    - all_features: False
      * NO usa todos los features
      * Solo usa subset reducido (tabular features)
    
    Hiperparámetros Sweep:
    ----------------------
    - lr: [0.001]
    - seq_size: [10]
    
    Uso:
    ----
    ```bash
    python src/main.py model=binctabl dataset=btc
    ```
    
    Nota:
    -----
    BiNCTABL tiene arquitectura fija hardcodeada en src/models/binctabl.py.
    Los hiperparámetros aquí no controlan hidden_dim ni num_layers.
    """
    hyperparameters_fixed: dict = field(default_factory=lambda: {
        "lr": 0.001, 
        "seq_size": 10, 
        "all_features": False
    })
    hyperparameters_sweep: dict = field(default_factory=lambda: {
        "lr": [0.001], 
        "seq_size": [10]
    })
    type: ModelType = ModelType.BINCTABL


@dataclass
class DeepLOB(Model):
    """
    CONFIGURACIÓN DEL MODELO DEEPLOB
    =================================
    
    Deep Learning for Limit Order Book (Zhang et al., 2019).
    Arquitectura: CNN + LSTM (arquitectura fija del paper original).
    
    Hiperparámetros Fijos:
    ----------------------
    - lr: 0.01
      * Learning rate más alto (CNN + LSTM convergen más rápido)
    
    - seq_size: 100
      * Del paper original DeepLOB
      * Secuencia de 100 timesteps
    
    - all_features: False
      * DeepLOB original usa normalización específica
      * No todos los features en formato estándar
    
    Hiperparámetros Sweep:
    ----------------------
    - lr: [0.01]
    - seq_size: [100]
    
    Uso:
    ----
    ```bash
    python src/main.py model=deeplob dataset=btc
    ```
    
    Nota:
    -----
    DeepLOB tiene arquitectura completamente fija (no configurable).
    Ver src/models/deeplob.py para detalles de la arquitectura.
    Paper: https://arxiv.org/abs/1808.03668
    """
    hyperparameters_fixed: dict = field(default_factory=lambda: {
        "lr": 0.01, 
        "seq_size": 100, 
        "all_features": False
    })
    hyperparameters_sweep: dict = field(default_factory=lambda: {
        "lr": [0.01], 
        "seq_size": [100]
    })
    type: ModelType = ModelType.DEEPLOB


# =============================================================================
# CONFIGURACIONES DE DATASETS
# =============================================================================

@dataclass
class Dataset:
    """
    CLASE BASE PARA CONFIGURACIÓN DE DATASETS
    ==========================================
    
    Define la estructura común para todos los datasets.
    
    Atributos:
        type (DatasetType): Tipo de dataset (BTC, FI_2010, LOBSTER)
        dates (list): Rango de fechas [fecha_inicio, fecha_fin]
        batch_size (int): Tamaño de batch para entrenamiento
    """
    type: DatasetType = MISSING
    dates: list = MISSING
    batch_size: int = MISSING


@dataclass
class FI_2010(Dataset):
    """
    CONFIGURACIÓN PARA DATASET FI-2010
    ===================================
    
    Finnish Stock Market dataset (benchmark académico).
    
    Características:
    ----------------
    - 5 acciones finlandesas
    - 10 niveles del LOB
    - Datos de Junio 2010
    - Pre-normalizado
    
    Configuración:
    --------------
    - type: DatasetType.FI_2010
    - dates: ["2010-01-01", "2010-12-31"]
      * Rango simbólico (datos reales son Junio 2010)
    - batch_size: 32
      * Batch pequeño (dataset relativamente pequeño)
    
    Estructura de Datos:
    -------------------
    Los datos se encuentran en src/data/FI_2010/:
    - Train_Dst_NoAuction_ZScore_CF_*.zip
    - Test_Dst_NoAuction_ZScore_CF_*.zip
    
    Uso:
    ----
    ```bash
    python src/main.py model=tlob dataset=fi_2010
    ```
    """
    type: DatasetType = DatasetType.FI_2010
    dates: list = field(default_factory=lambda: ["2010-01-01", "2010-12-31"])
    batch_size: int = 32


@dataclass
class LOBSTER(Dataset):
    """
    CONFIGURACIÓN PARA DATASET LOBSTER
    ===================================
    
    LOBSTER format (event-driven data de stocks estadounidenses).
    
    Características:
    ----------------
    - Event-driven (no uniformemente sampleado)
    - Requiere sampling TIME o QUANTITY
    - Metadata de eventos (order_type, depth, etc.)
    - Múltiples stocks posibles
    
    Configuración:
    --------------
    - type: DatasetType.LOBSTER
    
    - dates: ["2015-01-02", "2015-01-30"]
      * Rango de fechas de trading (Enero 2015 por defecto)
    
    - sampling_type: SamplingType.QUANTITY
      * QUANTITY: Sample cada N eventos
      * TIME: Sample cada N milisegundos
    
    - sampling_time: "1s"
      * Si sampling_type=TIME, intervalo de sampling
      * Formato: "100ms", "1s", etc.
    
    - sampling_quantity: 500
      * Si sampling_type=QUANTITY, número de eventos por snapshot
    
    - training_stocks: ["INTC"]
      * Lista de stocks para entrenamiento
      * Opciones típicas: ["INTC", "TSLA", "AAPL"]
    
    - testing_stocks: ["INTC"]
      * Lista de stocks para testing
      * Puede ser diferente de training para cross-validation
    
    - batch_size: 128
      * Batch grande (LOBSTER genera muchos datos)
    
    Uso:
    ----
    ```bash
    # INTC con sampling por cantidad
    python src/main.py model=tlob dataset=lobster \
        dataset.training_stocks=["INTC"]
    
    # TSLA con sampling por tiempo
    python src/main.py model=tlob dataset=lobster \
        dataset.training_stocks=["TSLA"] \
        dataset.sampling_type=TIME \
        dataset.sampling_time="500ms"
    
    # Multi-stock training
    python src/main.py model=tlob dataset=lobster \
        dataset.training_stocks=["INTC","TSLA"] \
        dataset.testing_stocks=["AAPL"]
    ```
    
    Estructura de Datos:
    -------------------
    Datos se guardan en src/data/LOBSTER/{STOCK}/:
    - train.npy
    - val.npy
    - test.npy
    
    Se generan con LOBSTERDataBuilder en preprocesamiento.
    """
    type: DatasetType = DatasetType.LOBSTER
    dates: list = field(default_factory=lambda: ["2015-01-02", "2015-01-30"])
    sampling_type: SamplingType = SamplingType.QUANTITY
    sampling_time: str = "1s"
    sampling_quantity: int = 500
    training_stocks: list = field(default_factory=lambda: ["INTC"])
    testing_stocks: list = field(default_factory=lambda: ["INTC"])
    batch_size: int = 128
    

@dataclass
class BTC(Dataset):
    """
    CONFIGURACIÓN PARA DATASET BTC (BITCOIN)
    =========================================
    
    Datos de Bitcoin/criptomonedas de Binance.
    
    Características:
    ----------------
    - BTCUSDT perpetual futures
    - Sampling fijo de 250ms (4 Hz)
    - 10 niveles del LOB
    - Ya pre-sampleado (sampling_type=NONE)
    
    Configuración:
    --------------
    - type: DatasetType.BTC
    
    - dates: ["2023-01-09", "2023-01-20"]
      * Rango de fechas simbólico
      * Datos reales pueden variar
    
    - sampling_type: SamplingType.NONE
      * Ya viene sampleado de Binance a 250ms
      * No requiere re-sampling
    
    - sampling_time: "100ms"
      * No usado (sampling_type=NONE)
      * Mantenido por compatibilidad
    
    - sampling_quantity: 0
      * No usado (sampling_type=NONE)
    
    - batch_size: 128
      * Batch grande (BTC tiene muchos datos)
    
    - training_stocks: ["BTC"]
      * Simbólico (BTC no es "stock")
      * Mantenido por compatibilidad con LOBSTER
    
    - testing_stocks: ["BTC"]
      * Simbólico
    
    Estructura de Datos:
    -------------------
    Datos en src/data/BTC/:
    - train.npy: ~2.7M timesteps × 44 features
    - val.npy: ~344K timesteps × 44 features
    - test.npy: ~605K timesteps × 44 features
    
    Nota: Features 0-39 son LOB, 40-43 son metadata.
    Solo features 0-39 se usan en el modelo.
    
    Uso:
    ----
    ```bash
    # Configuración standard
    python src/main.py model=tlob dataset=btc
    
    # Con horizonte de 20 timesteps
    python src/main.py model=tlob dataset=btc experiment.horizon=20
    
    # Debug mode (subset pequeño)
    python src/main.py model=tlob dataset=btc experiment.is_debug=true
    ```
    """
    type: DatasetType = DatasetType.BTC
    dates: list = field(default_factory=lambda: ["2023-01-09", "2023-01-20"])
    sampling_type: SamplingType = SamplingType.NONE
    sampling_time: str = "100ms"
    sampling_quantity: int = 0
    batch_size: int = 128
    training_stocks: list = field(default_factory=lambda: ["BTC"])
    testing_stocks: list = field(default_factory=lambda: ["BTC"])


# =============================================================================
# CONFIGURACIÓN DE EXPERIMENTO
# =============================================================================

@dataclass
class Experiment:
    """
    CONFIGURACIÓN DEL EXPERIMENTO
    =============================
    
    Controla el comportamiento del entrenamiento y experimento.
    
    Atributos:
    ----------
    is_data_preprocessed (bool): Si los datos ya están preprocesados
        - False: Ejecutar preprocesamiento (default)
        - True: Cargar datos preprocesados directamente
        - Útil para evitar re-procesar en múltiples runs
    
    is_wandb (bool): Usar Weights & Biases para logging
        - True: Log métricas a WandB (default)
        - False: Entrenamiento local sin logging
    
    is_sweep (bool): Ejecutar hyperparameter sweep
        - False: Entrenamiento normal (default)
        - True: Ejecutar grid search con WandB
        - Usa hyperparameters_sweep del modelo
    
    type (list): Tipo de experimento
        - ["TRAINING"]: Entrenar desde cero (default)
        - ["FINETUNING"]: Fine-tune desde checkpoint
        - ["EVALUATION"]: Solo evaluar (no entrenar)
        - Se pueden combinar: ["TRAINING", "EVALUATION"]
    
    is_debug (bool): Modo debug
        - False: Usar dataset completo (default)
        - True: Usar subset pequeño (1K train, 1K val, 10K test)
        - Útil para desarrollo rápido
    
    checkpoint_reference (str): Path relativo a checkpoint para cargar
        - "": No cargar checkpoint (default)
        - "BTC_seq_size_128_horizon_10_seed_42/pt/model.pt": Ejemplo
        - Usado en FINETUNING y EVALUATION
    
    seed (int): Semilla para reproducibilidad
        - Default: 1
        - Afecta: torch, numpy, random
        - Importante para reproducir resultados exactos
    
    horizon (int): Horizonte de predicción en timesteps
        - Default: 10
        - Opciones: 10, 20, 50, 100
        - Determina qué tan adelante predecir
        - horizon=10 @ 250ms = 2.5 segundos adelante
    
    max_epochs (int): Número máximo de épocas
        - Default: 10
        - Early stopping puede detener antes
        - Típicamente no se necesitan muchas épocas (2-5 suficientes)
    
    dir_ckpt (str): Directorio para guardar checkpoints
        - Default: "model.ckpt"
        - Se sobrescribe automáticamente con nombre descriptivo:
          "{DATASET}_seq_size_{seq}_horizon_{h}_seed_{s}"
        - Ej: "BTC_seq_size_128_horizon_10_seed_42"
    
    optimizer (str): Optimizador a usar
        - Default: "Adam"
        - Opciones: "Adam", "SGD", "AdamW"
        - Adam funciona mejor en la mayoría de casos
    
    Uso con Hydra:
    --------------
    ```bash
    # Entrenamiento normal
    python src/main.py model=tlob dataset=btc
    
    # Entrenamiento con WandB desactivado
    python src/main.py model=tlob dataset=btc experiment.is_wandb=false
    
    # Modo debug (rápido)
    python src/main.py model=tlob dataset=btc experiment.is_debug=true
    
    # Horizonte de 20 timesteps
    python src/main.py model=tlob dataset=btc experiment.horizon=20
    
    # Hyperparameter sweep
    python src/main.py model=tlob dataset=btc \
        experiment.is_sweep=true experiment.is_wandb=true
    
    # Fine-tuning desde checkpoint
    python src/main.py model=tlob dataset=btc \
        experiment.type=["FINETUNING"] \
        experiment.checkpoint_reference="BTC_seq_size_128_horizon_10_seed_42/pt/model.pt"
    
    # Solo evaluación
    python src/main.py model=tlob dataset=btc \
        experiment.type=["EVALUATION"] \
        experiment.checkpoint_reference="path/to/checkpoint.pt"
    ```
    
    Ejemplo en Python:
    ------------------
    ```python
    from src.config.config import Experiment
    
    exp = Experiment(
        is_wandb=True,
        horizon=20,
        seed=42,
        max_epochs=5
    )
    print(f"Horizonte: {exp.horizon}")
    # Output: Horizonte: 20
    ```
    """
    is_data_preprocessed: bool = False
    is_wandb: bool = True
    is_sweep: bool = False
    type: list = field(default_factory=lambda: ["TRAINING"])
    is_debug: bool = False
    checkpoint_reference: str = ""
    seed: int = 1
    horizon: int = 10
    max_epochs: int = 10
    dir_ckpt: str = "model.ckpt"
    optimizer: str = "Adam"
    

# =============================================================================
# CONFIGURACIÓN PRINCIPAL
# =============================================================================

defaults = [Model, Experiment, Dataset]

@dataclass
class Config:
    """
    CONFIGURACIÓN PRINCIPAL DEL PROYECTO
    =====================================
    
    Combina configuraciones de modelo, dataset y experimento.
    
    Atributos:
        model (Model): Configuración del modelo (TLOB, MLPLOB, etc.)
        dataset (Dataset): Configuración del dataset (BTC, FI_2010, etc.)
        experiment (Experiment): Configuración del experimento
        defaults (List): Configuración de Hydra logging
    
    Uso con Hydra:
    --------------
    Hydra lee config.yaml en runtime y permite override desde CLI.
    
    Estructura de config.yaml:
    ```yaml
    defaults:
      - model: tlob
      - dataset: btc
      - _self_
    
    experiment:
      horizon: 10
      seed: 42
    ```
    
    Ejemplo Completo:
    -----------------
    ```bash
    # Entrenar TLOB en BTC con horizonte 10
    python src/main.py model=tlob dataset=btc experiment.horizon=10
    
    # Entrenar MLPLOB en FI-2010 con horizonte 50
    python src/main.py model=mlplob dataset=fi_2010 experiment.horizon=50
    
    # TLOB en LOBSTER (TSLA) con sampling por tiempo
    python src/main.py model=tlob dataset=lobster \
        dataset.training_stocks=["TSLA"] \
        dataset.sampling_type=TIME \
        dataset.sampling_time="500ms" \
        experiment.horizon=20 \
        experiment.seed=42
    
    # Override múltiples parámetros
    python src/main.py \
        model=tlob \
        model.hyperparameters_fixed.num_layers=6 \
        model.hyperparameters_fixed.num_heads=8 \
        dataset=fi_2010 \
        dataset.batch_size=64 \
        experiment.horizon=100 \
        experiment.max_epochs=15 \
        experiment.seed=42
    ```
    
    Nota sobre Hydra:
    -----------------
    Hydra crea un directorio de salida para cada run:
    ```
    outputs/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── .hydra/
            │   └── config.yaml  # Config usado
            └── main.log         # Logs del run
    ```
    
    Para más info: https://hydra.cc/docs/intro/
    """
    model: Model
    dataset: Dataset
    experiment: Experiment = field(default_factory=Experiment)
    defaults: List = field(default_factory=lambda: [
        {"hydra/job_logging": "disabled"},
        {"hydra/hydra_logging": "disabled"},
        "_self_"
    ])
    

# =============================================================================
# REGISTRO EN CONFIG STORE (HYDRA)
# =============================================================================

# ConfigStore permite registrar configs para uso con Hydra
cs = ConfigStore.instance()

# Registrar config principal
cs.store(name="config", node=Config)

# Registrar configs de modelos (accesibles con model=nombre)
cs.store(group="model", name="mlplob", node=MLPLOB)
cs.store(group="model", name="tlob", node=TLOB)
cs.store(group="model", name="binctabl", node=BiNCTABL)
cs.store(group="model", name="deeplob", node=DeepLOB)

# Registrar configs de datasets (accesibles con dataset=nombre)
cs.store(group="dataset", name="lobster", node=LOBSTER)
cs.store(group="dataset", name="fi_2010", node=FI_2010)
cs.store(group="dataset", name="btc", node=BTC)
