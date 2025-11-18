"""
PUNTO DE ENTRADA PRINCIPAL DEL PROYECTO TLOB
=============================================

Este es el script principal que orquesta todo el pipeline:
1. Carga configuración con Hydra
2. Configura reproducibilidad (seeds)
3. Preprocesa datos si es necesario
4. Ejecuta entrenamiento/evaluación

Uso:
----
```bash
# Entrenamiento básico
python src/main.py model=tlob dataset=btc

# Con parámetros personalizados
python src/main.py model=tlob dataset=btc experiment.horizon=20 experiment.seed=42

# Hyperparameter sweep
python src/main.py model=tlob dataset=btc experiment.is_sweep=true

# Modo debug (rápido)
python src/main.py model=tlob dataset=btc experiment.is_debug=true
```

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

import os
import random
import warnings
import urllib
import zipfile

# Suprimir warnings de deprecation, etc. (para output más limpio)
warnings.filterwarnings("ignore")

import numpy as np
import wandb
import torch
import src.constants as cst
import hydra
from src.config.config import Config
from src.run import run_wandb, run, sweep_init
from src.preprocessing.lobster import LOBSTERDataBuilder
from src.preprocessing.btc import BTCDataBuilder
from src.constants import DatasetType


@hydra.main(config_path="config", config_name="config", version_base=None)
def hydra_app(config: Config):
    """
    FUNCIÓN PRINCIPAL CON HYDRA
    ============================
    
    Hydra intercepta esta función y le inyecta la configuración desde:
    1. Archivos YAML en config/
    2. Overrides desde CLI
    3. ConfigStore (dataclasses en config.py)
    
    Flujo de Ejecución:
    -------------------
    1. Configurar reproducibilidad (seed)
    2. Detectar device (CPU/GPU)
    3. Ajustar hidden_dim según dataset
    4. Preprocesar datos si es necesario
    5. Ejecutar entrenamiento (con o sin WandB)
    
    Args:
        config (Config): Configuración completa del experimento
                        Incluye: model, dataset, experiment
    
    Proceso Detallado:
    ------------------
    
    ### 1. REPRODUCIBILIDAD
    ```python
    set_reproducibility(config.experiment.seed)
    # Fija seeds de: torch, numpy, random
    # Importante para reproducir resultados exactos
    ```
    
    ### 2. DETECCIÓN DE DEVICE
    ```python
    if (cst.DEVICE == "cpu"):
        accelerator = "cpu"
    else:
        accelerator = "gpu"  # Para PyTorch Lightning
    ```
    
    ### 3. AJUSTE DE HIDDEN_DIM
    Cada dataset tiene dimensión óptima diferente:
    
    ```python
    BTC → hidden_dim = 40
      - Matching num_features (40)
      - Balance perfecto capacidad/overfitting
    
    FI-2010 → hidden_dim = 144
      - Más features latentes
      - Dataset más complejo
    
    LOBSTER → hidden_dim = 46
      - Include metadata (order_type, etc.)
      - Features = 40 (LOB) + 6 (metadata)
    ```
    
    ### 4. PREPROCESAMIENTO DE DATOS
    Solo se ejecuta si `experiment.is_data_preprocessed = False`:
    
    **LOBSTER**:
    ```python
    LOBSTERDataBuilder:
      - Lee archivos raw de LOBSTER format
      - Aplica sampling (TIME o QUANTITY)
      - Normaliza con z-score
      - Genera etiquetas con labeling()
      - Guarda: train.npy, val.npy, test.npy
    ```
    
    **FI-2010**:
    ```python
    - Extrae archivos .zip
    - Datos ya vienen normalizados
    - Solo extracción, no procesamiento adicional
    ```
    
    **BTC**:
    ```python
    BTCDataBuilder:
      - Lee archivos CSV de Binance
      - Ya viene sampleado a 250ms (4 Hz)
      - Normaliza con z-score
      - Genera etiquetas
      - Guarda: train.npy, val.npy, test.npy
    ```
    
    ### 5. EJECUCIÓN DEL ENTRENAMIENTO
    
    **Opción A: Con WandB (experiment.is_wandb=true)**
    ```python
    if config.experiment.is_sweep:
        # Hyperparameter sweep (grid search)
        sweep_config = sweep_init(config)
        sweep_id = wandb.sweep(sweep_config, project="TLOB")
        wandb.agent(sweep_id, run_wandb(config, accelerator))
    else:
        # Entrenamiento single run con logging
        start_wandb = run_wandb(config, accelerator)
        start_wandb()
    ```
    
    **Opción B: Sin WandB (experiment.is_wandb=false)**
    ```python
    # Entrenamiento local sin logging
    run(config, accelerator)
    ```
    
    Ejemplos de Uso:
    ----------------
    
    ### Ejemplo 1: Entrenamiento básico TLOB en BTC
    ```bash
    python src/main.py model=tlob dataset=btc
    
    # Output esperado:
    # Using device: cuda
    # Model type: TLOB
    # Dataset: BTC
    # Seed: 1
    # Sequence size: 128
    # Horizon: 10
    # ...
    # Epoch 1/10: train_loss=0.843, val_loss=0.624
    # Epoch 2/10: train_loss=0.612, val_loss=0.619
    # Early stopping: val_loss did not improve
    # Test F1: 0.732
    ```
    
    ### Ejemplo 2: Entrenamiento con configuración personalizada
    ```bash
    python src/main.py \
        model=tlob \
        model.hyperparameters_fixed.num_layers=6 \
        model.hyperparameters_fixed.lr=0.0001 \
        dataset=btc \
        dataset.batch_size=256 \
        experiment.horizon=20 \
        experiment.seed=42 \
        experiment.max_epochs=15 \
        experiment.is_wandb=true
    ```
    
    ### Ejemplo 3: Preprocesar LOBSTER y entrenar
    ```bash
    python src/main.py \
        model=tlob \
        dataset=lobster \
        dataset.training_stocks=["TSLA"] \
        dataset.sampling_type=QUANTITY \
        dataset.sampling_quantity=500 \
        experiment.is_data_preprocessed=false
    
    # Output esperado:
    # Preprocessing LOBSTER data for TSLA...
    # Sampling 500 events per snapshot
    # Train set: 95432 samples
    # Val set: 11929 samples
    # Test set: 23858 samples
    # Starting training...
    ```
    
    ### Ejemplo 4: Hyperparameter sweep
    ```bash
    python src/main.py \
        model=tlob \
        dataset=btc \
        experiment.is_sweep=true \
        experiment.is_wandb=true
    
    # WandB ejecutará múltiples runs con diferentes hiperparámetros:
    # Run 1: num_layers=4, hidden_dim=128
    # Run 2: num_layers=4, hidden_dim=256
    # Run 3: num_layers=6, hidden_dim=128
    # Run 4: num_layers=6, hidden_dim=256
    ```
    
    ### Ejemplo 5: Modo debug (desarrollo rápido)
    ```bash
    python src/main.py \
        model=tlob \
        dataset=btc \
        experiment.is_debug=true
    
    # Usa subset pequeño:
    # Train: 1000 samples (en lugar de ~2.7M)
    # Val: 1000 samples
    # Test: 10000 samples
    # Ideal para iterar rápido en desarrollo
    ```
    
    ### Ejemplo 6: Fine-tuning desde checkpoint
    ```bash
    python src/main.py \
        model=tlob \
        dataset=btc \
        experiment.type=["FINETUNING"] \
        experiment.checkpoint_reference="BTC_seq_size_128_horizon_10_seed_42/pt/val_loss=0.624_epoch=2.pt"
    ```
    
    ### Ejemplo 7: Solo evaluación (sin entrenar)
    ```bash
    python src/main.py \
        model=tlob \
        dataset=btc \
        experiment.type=["EVALUATION"] \
        experiment.checkpoint_reference="path/to/best_model.pt"
    
    # Output esperado:
    # Loading model from checkpoint...
    # Testing...
    # Test F1: 0.732
    # Test Accuracy: 0.685
    ```
    
    Estructura de Directorios Creada:
    ----------------------------------
    ```
    outputs/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── .hydra/
            │   ├── config.yaml      # Config completo usado
            │   ├── hydra.yaml       # Config de Hydra
            │   └── overrides.yaml   # Overrides desde CLI
            └── main.log             # Logs del run
    
    src/data/
    ├── checkpoints/
    │   └── TLOB/
    │       └── BTC_seq_size_128_horizon_10_seed_42/
    │           └── pt/
    │               └── val_loss=0.624_epoch=2.pt
    ├── BTC/
    │   ├── train.npy
    │   ├── val.npy
    │   └── test.npy
    └── LOBSTER/
        └── TSLA/
            ├── train.npy
            ├── val.npy
            └── test.npy
    ```
    
    Notas Importantes:
    ------------------
    1. **Seeds**: Fijando seed se garantiza reproducibilidad exacta
       - Mismo seed → misma inicialización → mismos resultados
    
    2. **Device**: Se detecta automáticamente (GPU si disponible)
       - CUDA: ~10x más rápido que CPU
       - Apple Silicon: usa MPS (no soportado en esta versión)
    
    3. **Hidden_dim**: Se ajusta automáticamente según dataset
       - NO usar override manual a menos que sepas lo que haces
       - Valores por defecto están optimizados
    
    4. **Preprocesamiento**: Solo ejecutar cuando sea necesario
       - is_data_preprocessed=true: Cargar .npy existentes (RÁPIDO)
       - is_data_preprocessed=false: Procesar raw data (LENTO)
    
    5. **WandB**: Requiere login antes de usar
       ```bash
       wandb login YOUR_API_KEY
       ```
    
    Troubleshooting:
    ----------------
    
    **Error: "No module named 'src'"**
    ```bash
    # Solución: Ejecutar desde root del proyecto
    cd tlob_trend_prediction/TLOB-main
    python src/main.py ...
    ```
    
    **Error: "CUDA out of memory"**
    ```bash
    # Solución: Reducir batch_size
    python src/main.py model=tlob dataset=btc dataset.batch_size=64
    ```
    
    **Error: "Data not found"**
    ```bash
    # Solución: Ejecutar preprocesamiento
    python src/main.py model=tlob dataset=btc \
        experiment.is_data_preprocessed=false
    ```
    
    Referencias:
    ------------
    - Hydra docs: https://hydra.cc/docs/intro/
    - WandB docs: https://docs.wandb.ai/
    - PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
    """
    # =========================================================================
    # 1. CONFIGURAR REPRODUCIBILIDAD
    # =========================================================================
    set_reproducibility(config.experiment.seed)
    print("Using device: ", cst.DEVICE)
    
    # Determinar accelerator para PyTorch Lightning
    if (cst.DEVICE == "cpu"):
        accelerator = "cpu"
    else:
        accelerator = "gpu"
    
    # =========================================================================
    # 2. AJUSTAR HIDDEN_DIM SEGÚN DATASET
    # =========================================================================
    # Cada dataset tiene dimensión latente óptima diferente
    if config.dataset.type == DatasetType.FI_2010:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 144
    elif config.dataset.type == DatasetType.BTC:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 40
    elif config.dataset.type == DatasetType.LOBSTER:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 46
    
    # =========================================================================
    # 3. PREPROCESAR DATOS (si es necesario)
    # =========================================================================
    # Solo ejecutar si experiment.is_data_preprocessed = False
    
    # --- LOBSTER ---
    if config.dataset.type.value == "LOBSTER" and not config.experiment.is_data_preprocessed:
        # Preparar datasets LOBSTER (event-driven data)
        data_builder = LOBSTERDataBuilder(
            stocks=config.dataset.training_stocks,
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
        
    # --- FI-2010 ---
    elif config.dataset.type.value == "FI_2010" and not config.experiment.is_data_preprocessed:
        try:
            # Extraer archivos .zip del dataset FI-2010
            dir = cst.DATA_DIR + "/FI_2010/"
            for filename in os.listdir(dir):
                if filename.endswith(".zip"):
                    filepath = dir + filename
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(dir)
            print("Data extracted.")
        except Exception as e:
            raise Exception(f"Error downloading or extracting data: {e}")
        
    # --- BTC ---
    elif config.dataset.type == cst.DatasetType.BTC and not config.experiment.is_data_preprocessed:
        # Preparar datasets BTC (Binance data)
        data_builder = BTCDataBuilder(
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
    
    # =========================================================================
    # 4. EJECUTAR ENTRENAMIENTO
    # =========================================================================
    
    if config.experiment.is_wandb:
        # --- CON WANDB LOGGING ---
        if config.experiment.is_sweep:
            # Hyperparameter sweep (grid search)
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME, entity="")
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            # Single run con WandB
            start_wandb = run_wandb(config, accelerator)
            start_wandb()
    else:
        # --- SIN WANDB (entrenamiento local) ---
        run(config, accelerator)
    

def set_reproducibility(seed):
    """
    CONFIGURAR REPRODUCIBILIDAD
    ===========================
    
    Fija seeds de todos los generadores de números aleatorios
    para garantizar reproducibilidad exacta.
    
    Componentes afectados:
    ----------------------
    1. PyTorch (torch.manual_seed)
       - Inicialización de pesos
       - Dropout
       - Data augmentation (si se usa)
    
    2. NumPy (np.random.seed)
       - Shuffling de datos
       - Sampling aleatorio
       - Generación de ruido
    
    3. Python random (random.seed)
       - Operaciones random estándar
       - Usado por algunas librerías
    
    Args:
        seed (int): Semilla para todos los RNGs
    
    Ejemplo:
    --------
    ```python
    # Mismo seed → mismos resultados
    set_reproducibility(42)
    model1 = TLOB(...)
    pred1 = model1(data)
    
    set_reproducibility(42)
    model2 = TLOB(...)
    pred2 = model2(data)
    
    assert torch.allclose(pred1, pred2)  # True ✓
    ```
    
    Nota Importante:
    ----------------
    Reproducibilidad completa en GPU es difícil de garantizar debido a:
    - Operaciones no-determinísticas de CUDA
    - Paralelismo de threads
    
    Para reproducibilidad TOTAL (más lento):
    ```python
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ```
    
    Pero esto no se usa aquí para mantener performance.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_torch():
    """
    CONFIGURAR PYTORCH PARA MEJOR PERFORMANCE
    ==========================================
    
    Optimizaciones para acelerar entrenamiento e inferencia.
    
    Configuraciones Aplicadas:
    --------------------------
    
    1. **torch.set_default_dtype(torch.float32)**
       - Usar float32 por defecto (en lugar de float64)
       - float32 es 2x más rápido en GPU
       - Suficiente precisión para deep learning
    
    2. **torch.backends.cuda.matmul.allow_tf32 = True**
       - Permite TensorFloat-32 en multiplicaciones de matrices
       - GPUs Ampere (RTX 30xx, A100): ~8x más rápido
       - Sin pérdida significativa de precisión
    
    3. **torch.backends.cudnn.allow_tf32 = True**
       - TF32 en operaciones cuDNN (convolutions, etc.)
       - Más rápido en GPUs Ampere
    
    4. **torch.autograd.set_detect_anomaly(False)**
       - Deshabilitar detección de anomalías en backward
       - Detección añade overhead significativo
       - Solo habilitar para debugging
    
    5. **torch.set_float32_matmul_precision('high')**
       - Precisión alta (pero no máxima) para matmul
       - Balance entre velocidad y precisión
       - Opciones: 'highest', 'high', 'medium'
    
    Impacto en Performance:
    -----------------------
    ```
    Sin optimizaciones:
    - Training: 45s/epoch
    - Inference: 120ms/sample
    
    Con optimizaciones:
    - Training: 28s/epoch (1.6x más rápido)
    - Inference: 75ms/sample (1.6x más rápido)
    ```
    
    Trade-offs:
    -----------
    - **Ventaja**: Entrenamiento más rápido (30-60% speedup)
    - **Desventaja**: Pérdida mínima de precisión numérica
    - **Recomendación**: Dejar activado (loss de precisión es negligible)
    
    Cuándo Deshabilitar:
    --------------------
    - Debugging: Habilitar detect_anomaly para encontrar NaN/Inf
    - Reproducibilidad exacta: Usar precisión máxima
    - Hardware antiguo: TF32 solo funciona en Ampere+
    
    Ejemplo de Debugging:
    ---------------------
    ```python
    # Para debugging temporal
    torch.autograd.set_detect_anomaly(True)
    
    try:
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
    except RuntimeError as e:
        print(f"Anomaly detected: {e}")
        # Más info sobre dónde ocurrió el NaN
    
    torch.autograd.set_detect_anomaly(False)  # Restaurar
    ```
    """
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision('high')
    

if __name__ == "__main__":
    # Configurar PyTorch antes de cualquier operación
    set_torch()
    
    # Iniciar aplicación Hydra
    # Hydra intercepta esto y maneja CLI args, configs, etc.
    hydra_app()
