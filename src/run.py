"""
PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN
=======================================

Contiene toda la lógica de entrenamiento, evaluación y logging del proyecto.
Incluye funciones para:
- Cargar datasets (BTC, FI-2010, LOBSTER)
- Entrenar modelos con/sin WandB
- Evaluar modelos
- Hyperparameter sweeps

Este es el módulo más importante del proyecto después de los modelos.

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

import lightning as L
import omegaconf
import torch
import glob
import os
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.config.config import Config
from src.models.engine import Engine
from src.preprocessing.fi_2010 import fi_2010_load
from src.preprocessing.lobster import lobster_load
from src.preprocessing.btc import btc_load
from src.preprocessing.dataset import Dataset, DataModule
import src.constants as cst
from src.constants import DatasetType, SamplingType

# Permitir serialización de ListConfig de Hydra (necesario para checkpoints)
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])


def run(config: Config, accelerator):
    """
    EJECUTA ENTRENAMIENTO SIN WANDB
    ================================
    
    Función principal para entrenamiento local sin logging a WandB.
    Útil para desarrollo rápido y debugging.
    
    Args:
        config (Config): Configuración completa del experimento
        accelerator (str): "cpu" o "gpu" para PyTorch Lightning
    
    Proceso:
    --------
    1. Construir nombre de directorio de checkpoint
    2. Configurar PyTorch Lightning Trainer
    3. Ejecutar entrenamiento (función train())
    
    Ejemplo:
    --------
    ```python
    from src.config.config import Config
    config = Config(...)
    run(config, accelerator="gpu")
    ```
    
    Output:
    -------
    - Checkpoints guardados en src/data/checkpoints/
    - Logs en console (TQDM progress bars)
    - Métricas impresas en console
    
    Nota:
    -----
    Para logging completo a WandB, usar run_wandb() en su lugar.
    """
    # =========================================================================
    # CONFIGURAR NOMBRE DE CHECKPOINT
    # =========================================================================
    seq_size = config.model.hyperparameters_fixed["seq_size"]
    dataset = config.dataset.type.value
    horizon = config.experiment.horizon
    
    # Construir nombre descriptivo del checkpoint
    if dataset == "LOBSTER":
        training_stocks = config.dataset.training_stocks
        config.experiment.dir_ckpt = f"{dataset}_{training_stocks}_seq_size_{seq_size}_horizon_{horizon}_seed_{config.experiment.seed}"
    else:
        config.experiment.dir_ckpt = f"{dataset}_seq_size_{seq_size}_horizon_{horizon}_seed_{config.experiment.seed}"
    
    # =========================================================================
    # CONFIGURAR TRAINER
    # =========================================================================
    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,  # 32-bit float
        max_epochs=config.experiment.max_epochs,
        callbacks=[
            # Early stopping: Para si val_loss no mejora
            EarlyStopping(
                monitor="val_loss", 
                mode="min", 
                patience=1,  # Para después de 1 época sin mejora
                verbose=True, 
                min_delta=0.002  # Mejora mínima de 0.002
            ),
            # Progress bar en console
            TQDMProgressBar(refresh_rate=100)  # Actualizar cada 100 batches
        ],
        num_sanity_val_steps=0,  # Sin validation sanity check
        detect_anomaly=False,  # No detectar anomalías (más rápido)
        profiler=None,  # Sin profiling
        check_val_every_n_epoch=1  # Validar cada época
    )
    
    # =========================================================================
    # ENTRENAR
    # =========================================================================
    train(config, trainer)


def train(config: Config, trainer: L.Trainer, run=None):
    """
    FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
    ===================================
    
    Maneja todo el pipeline de entrenamiento:
    1. Cargar datasets según tipo
    2. Crear o cargar modelo
    3. Entrenar
    4. Evaluar en test set
    
    Args:
        config (Config): Configuración del experimento
        trainer (L.Trainer): PyTorch Lightning Trainer configurado
        run (wandb.Run, optional): WandB run para logging
                                   Si None, no log a WandB
    
    Proceso Completo:
    -----------------
    
    ### 1. CARGAR DATASETS
    Dependiendo de config.dataset.type:
    - BTC: btc_load() de archivos .npy
    - FI-2010: fi_2010_load() de archivos .zip
    - LOBSTER: lobster_load() para múltiples stocks
    
    ### 2. IMPRIMIR ESTADÍSTICAS
    - Shapes de train/val/test
    - Distribución de clases (UP/STAT/DOWN)
    
    ### 3. CREAR O CARGAR MODELO
    Dependiendo de config.experiment.type:
    - "TRAINING": Crear modelo nuevo
    - "FINETUNING": Cargar checkpoint + continuar
    - "EVALUATION": Cargar checkpoint + solo evaluar
    
    ### 4. ENTRENAR (si aplicable)
    - trainer.fit(model, train_loader, val_loader)
    - Early stopping automático
    - Checkpoints guardados automáticamente
    
    ### 5. EVALUAR EN TEST
    - trainer.test(model, test_loader)
    - Log F1 score a WandB (si run != None)
    
    Ejemplo de Uso:
    ---------------
    ```python
    # Entrenamiento local
    trainer = L.Trainer(...)
    train(config, trainer, run=None)
    
    # Entrenamiento con WandB
    wandb_run = wandb.init(...)
    train(config, trainer, run=wandb_run)
    ```
    
    Output Típico:
    --------------
    ```
    Model type: TLOB
    Dataset: BTC
    Seed: 42
    Sequence size: 128
    Horizon: 10
    ...
    Train set shape: torch.Size([2700000, 40])
    Val set shape: torch.Size([344000, 40])
    Test set shape: torch.Size([605000, 40])
    Classes distribution in train set: up 0.35 stat 0.30 down 0.35
    Classes distribution in val set: up 0.34 stat 0.31 down 0.35
    Classes distribution in test set: up 0.35 stat 0.30 down 0.35
    
    Total number of parameters: 1,100,000
    
    Epoch 1/10: 100%|████| 21094/21094 [00:32<00:00]
    train_loss=0.843, val_loss=0.624
    
    Epoch 2/10: 100%|████| 21094/21094 [00:32<00:00]
    train_loss=0.612, val_loss=0.619
    
    Early stopping: val_loss did not improve by 0.002
    
    Best model path: src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_42/pt/val_loss=0.619_epoch=2.pt
    
    Testing: 100%|████| 4727/4727 [00:12<00:00]
    Test F1: 0.732
    Test Accuracy: 0.685
    ```
    """
    # =========================================================================
    # PRINT SETUP
    # =========================================================================
    print_setup(config)
    
    dataset_type = config.dataset.type.value
    seq_size = config.model.hyperparameters_fixed["seq_size"]
    horizon = config.experiment.horizon
    model_type = config.model.type
    checkpoint_ref = config.experiment.checkpoint_reference
    checkpoint_path = os.path.join(cst.DIR_SAVED_MODEL, model_type.value, checkpoint_ref)
    
    # ========================================================================
    # CARGAR DATASETS
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # DATASET: FI-2010
    # ------------------------------------------------------------------------
    if dataset_type == "FI_2010":
        path = cst.DATA_DIR + "/FI_2010"
        train_input, train_labels, val_input, val_labels, test_input, test_labels = fi_2010_load(
            path, seq_size, horizon, config.model.hyperparameters_fixed["all_features"]
        )
        
        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        test_set = Dataset(test_input, test_labels, seq_size)
        
        # Modo debug: Usar subset pequeño
        if config.experiment.is_debug:
            train_set.length = 1000
            val_set.length = 1000
            test_set.length = 10000
        
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,  # Test con batch más grande
            num_workers=4  # 4 workers para DataLoader
        )
        test_loaders = [data_module.test_dataloader()]
    
    # ------------------------------------------------------------------------
    # DATASET: BTC
    # ------------------------------------------------------------------------
    elif dataset_type == "BTC":
        # Cargar archivos .npy preprocesados
        train_input, train_labels = btc_load(cst.DATA_DIR + "/BTC/train.npy", cst.LEN_SMOOTH, horizon, seq_size)
        val_input, val_labels = btc_load(cst.DATA_DIR + "/BTC/val.npy", cst.LEN_SMOOTH, horizon, seq_size)
        test_input, test_labels = btc_load(cst.DATA_DIR + "/BTC/test.npy", cst.LEN_SMOOTH, horizon, seq_size)
        
        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        test_set = Dataset(test_input, test_labels, seq_size)
        
        # Modo debug: Usar subset pequeño
        if config.experiment.is_debug:
            train_set.length = 1000
            val_set.length = 1000
            test_set.length = 10000
        
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        )
        test_loaders = [data_module.test_dataloader()]
        
    # ------------------------------------------------------------------------
    # DATASET: LOBSTER
    # ------------------------------------------------------------------------
    elif dataset_type == "LOBSTER":
        training_stocks = config.dataset.training_stocks
        testing_stocks = config.dataset.testing_stocks
        
        # Cargar múltiples stocks de training
        for i in range(len(training_stocks)):
            if i == 0:
                # Primer stock: Inicializar train y val
                for j in range(2):
                    if j == 0:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/train.npy"
                        train_input, train_labels = lobster_load(
                            path, config.model.hyperparameters_fixed["all_features"], 
                            cst.LEN_SMOOTH, horizon, seq_size
                        )
                    if j == 1:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/val.npy"
                        val_input, val_labels = lobster_load(
                            path, config.model.hyperparameters_fixed["all_features"], 
                            cst.LEN_SMOOTH, horizon, seq_size
                        )
            else:
                # Stocks subsiguientes: Concatenar con padding
                for j in range(2):
                    if j == 0:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/train.npy"
                        # Agregar padding de zeros entre stocks
                        train_labels = torch.cat((train_labels, torch.zeros(seq_size+horizon-1, dtype=torch.long)), 0)
                        train_input_tmp, train_labels_tmp = lobster_load(
                            path, config.model.hyperparameters_fixed["all_features"], 
                            cst.LEN_SMOOTH, horizon, seq_size
                        )
                        train_input = torch.cat((train_input, train_input_tmp), 0)
                        train_labels = torch.cat((train_labels, train_labels_tmp), 0)
                    if j == 1:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/val.npy"
                        # Agregar padding de zeros entre stocks
                        val_labels = torch.cat((val_labels, torch.zeros(seq_size+horizon-1, dtype=torch.long)), 0)
                        val_input_tmp, val_labels_tmp = lobster_load(
                            path, config.model.hyperparameters_fixed["all_features"], 
                            cst.LEN_SMOOTH, horizon, seq_size
                        )
                        val_input = torch.cat((val_input, val_input_tmp), 0)
                        val_labels = torch.cat((val_labels, val_labels_tmp), 0)
        
        # Cargar múltiples stocks de testing (cada uno en su propio loader)
        test_loaders = []
        for i in range(len(testing_stocks)):
            path = cst.DATA_DIR + "/" + testing_stocks[i] + "/test.npy"
            test_input, test_labels = lobster_load(
                path, config.model.hyperparameters_fixed["all_features"], 
                cst.LEN_SMOOTH, horizon, seq_size
            )
            test_set = Dataset(test_input, test_labels, seq_size)
            test_dataloader = DataLoader(
                dataset=test_set,
                batch_size=config.dataset.batch_size*4,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=4,
                persistent_workers=True
            )
            test_loaders.append(test_dataloader)
        
        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        
        # Modo debug: Usar subset pequeño
        if config.experiment.is_debug:
            train_set.length = 1000
            val_set.length = 1000
            test_set.length = 10000
        
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # ========================================================================
    # IMPRIMIR ESTADÍSTICAS DE DATASETS
    # ========================================================================
    
    counts_train = torch.unique(train_labels, return_counts=True)
    counts_val = torch.unique(val_labels, return_counts=True)
    counts_test = torch.unique(test_labels, return_counts=True)
    
    print()
    print("Train set shape: ", train_input.shape)
    print("Val set shape: ", val_input.shape)
    print("Test set shape: ", test_input.shape)
    print(f"Classes distribution in train set: up {(counts_train[1][0].item()/train_labels.shape[0]):.2f} "
          f"stat {(counts_train[1][1].item()/train_labels.shape[0]):.2f} "
          f"down {(counts_train[1][2].item()/train_labels.shape[0]):.2f}")
    print(f"Classes distribution in val set: up {(counts_val[1][0].item()/val_labels.shape[0]):.2f} "
          f"stat {(counts_val[1][1].item()/val_labels.shape[0]):.2f} "
          f"down {(counts_val[1][2].item()/val_labels.shape[0]):.2f}")
    print(f"Classes distribution in test set: up {(counts_test[1][0].item()/test_labels.shape[0]):.2f} "
          f"stat {(counts_test[1][1].item()/test_labels.shape[0]):.2f} "
          f"down {(counts_test[1][2].item()/test_labels.shape[0]):.2f}")
    print()
    
    # ========================================================================
    # CARGAR O CREAR MODELO
    # ========================================================================
    
    experiment_type = config.experiment.type
    
    # ------------------------------------------------------------------------
    # CARGAR CHECKPOINT (Fine-tuning o Evaluation)
    # ------------------------------------------------------------------------
    if "FINETUNING" in experiment_type or "EVALUATION" in experiment_type:
        if checkpoint_ref != "":
            checkpoint = torch.load(checkpoint_path, map_location=cst.DEVICE, weights_only=True)
        
        print("Loading model from checkpoint: ", config.experiment.checkpoint_reference)
        
        # Extraer hiperparámetros del checkpoint
        lr = checkpoint["hyper_parameters"]["lr"]
        dir_ckpt = checkpoint["hyper_parameters"]["dir_ckpt"]
        hidden_dim = checkpoint["hyper_parameters"]["hidden_dim"]
        num_layers = checkpoint["hyper_parameters"]["num_layers"]
        optimizer = checkpoint["hyper_parameters"]["optimizer"]
        model_type = checkpoint["hyper_parameters"]["model_type"]
        max_epochs = checkpoint["hyper_parameters"]["max_epochs"]
        horizon = checkpoint["hyper_parameters"]["horizon"]
        seq_size = checkpoint["hyper_parameters"]["seq_size"]
        
        # Cargar modelo según tipo
        if model_type == "MLPLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                map_location=cst.DEVICE,
            )
        elif model_type == "TLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=checkpoint["hyper_parameters"]["num_heads"],
                is_sin_emb=checkpoint["hyper_parameters"]["is_sin_emb"],
                map_location=cst.DEVICE,
                len_test_dataloader=len(test_loaders[0])
            )
        elif model_type == "BINCTABL":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                map_location=cst.DEVICE,
                len_test_dataloader=len(test_loaders[0])
            )
        elif model_type == "DEEPLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                map_location=cst.DEVICE,
                len_test_dataloader=len(test_loaders[0])
            )
    
    # ------------------------------------------------------------------------
    # CREAR MODELO NUEVO (Training)
    # ------------------------------------------------------------------------
    else:
        # Crear nuevo modelo según tipo
        if model_type == cst.ModelType.MLPLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0])
            )
        elif model_type == cst.ModelType.TLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=config.model.hyperparameters_fixed["num_heads"],
                is_sin_emb=config.model.hyperparameters_fixed["is_sin_emb"],
                len_test_dataloader=len(test_loaders[0])
            )
        elif model_type == cst.ModelType.BINCTABL:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0])
            )
        elif model_type == cst.ModelType.DEEPLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0])
            )
    
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    train_dataloader, val_dataloader = data_module.train_dataloader(), data_module.val_dataloader()
    
    # ========================================================================
    # ENTRENAR Y EVALUAR
    # ========================================================================
    
    if "TRAINING" in experiment_type or "FINETUNING" in experiment_type:
        # Entrenar modelo
        trainer.fit(model, train_dataloader, val_dataloader)
        best_model_path = model.last_path_ckpt
        print("Best model path: ", best_model_path)
        
        # Cargar mejor modelo
        try:
            best_model = Engine.load_from_checkpoint(best_model_path, map_location=cst.DEVICE)
        except:
            print("No checkpoints have been saved, selecting the last model")
            best_model = model
        
        best_model.experiment_type = "EVALUATION"
        
        # Evaluar en test set(s)
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            output = trainer.test(best_model, test_dataloader)
            
            # Log a WandB si run != None
            if run is not None and dataset_type == "LOBSTER":
                run.log({f"f1 {testing_stocks[i]} best": output[0]["f1_score"]}, commit=False)
            elif run is not None and dataset_type == "FI_2010":
                run.log({f"f1 FI_2010": output[0]["f1_score"]}, commit=False)
    else:
        # Solo evaluación (sin entrenamiento)
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            output = trainer.test(model, test_dataloader)
            
            # Log a WandB si run != None
            if run is not None and dataset_type == "LOBSTER":
                run.log({f"f1 {testing_stocks[i]} best": output[0]["f1_score"]}, commit=False)
            elif run is not None and dataset_type == "FI_2010":
                run.log({f"f1 FI_2010": output[0]["f1_score"]}, commit=False)


def run_wandb(config: Config, accelerator):
    """
    EJECUTA ENTRENAMIENTO CON WANDB LOGGING
    ========================================
    
    Wrapper para entrenamiento con logging a Weights & Biases.
    Crea instancia de WandB, configura logging y ejecuta train().
    
    Args:
        config (Config): Configuración del experimento
        accelerator (str): "cpu" o "gpu"
    
    Returns:
        function: Callback de WandB para ejecutar
    
    Uso:
    ----
    ```python
    # Entrenamiento single run
    start_wandb = run_wandb(config, "gpu")
    start_wandb()
    
    # Hyperparameter sweep
    sweep_config = sweep_init(config)
    sweep_id = wandb.sweep(sweep_config, project="TLOB")
    wandb.agent(sweep_id, run_wandb(config, "gpu"), count=10)
    ```
    
    Logging a WandB:
    ----------------
    - Config completa
    - Métricas de training (loss, accuracy)
    - Métricas de validation (loss, f1)
    - Métricas de test (f1, accuracy)
    - Distribución de clases
    - Hiperparámetros
    
    Run Name:
    ---------
    Formato: {DATASET}_seq_size_{seq}_horizon_{h}_seed_{s}
    Ejemplo: BTC_seq_size_128_horizon_10_seed_42
    """
    def wandb_sweep_callback():
        # Configurar WandB logger
        wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model=False, save_dir=cst.DIR_SAVED_MODEL)
        run_name = None
        
        # Construir run name (si no es sweep)
        if not config.experiment.is_sweep:
            run_name = ""
            for param in config.model.keys():
                value = config.model[param]
                if param == "hyperparameters_sweep":
                    continue
                if type(value) == omegaconf.dictconfig.DictConfig:
                    for key in value.keys():
                        run_name += str(key[:2]) + "_" + str(value[key]) + "_"
                else:
                    run_name += str(param[:2]) + "_" + str(value.value) + "_"
        
        # Inicializar WandB run
        run = wandb.init(project=cst.PROJECT_NAME, name=run_name, entity="")  # set entity to your wandb username
        
        # Obtener hiperparámetros (de sweep o config fijo)
        if config.experiment.is_sweep:
            model_params = run.config
        else:
            model_params = config.model.hyperparameters_fixed
        
        # Actualizar config con parámetros de sweep
        wandb_instance_name = ""
        for param in config.model.hyperparameters_fixed.keys():
            if param in model_params:
                config.model.hyperparameters_fixed[param] = model_params[param]
                wandb_instance_name += str(param) + "_" + str(model_params[param]) + "_"
        
        run.name = wandb_instance_name
        seq_size = config.model.hyperparameters_fixed["seq_size"]
        horizon = config.experiment.horizon
        dataset = config.dataset.type.value
        seed = config.experiment.seed
        
        # Construir nombre de checkpoint
        if dataset == "LOBSTER":
            training_stocks = config.dataset.training_stocks
            config.experiment.dir_ckpt = f"{dataset}_{training_stocks}_seq_size_{seq_size}_horizon_{horizon}_seed_{seed}"
        else:
            config.experiment.dir_ckpt = f"{dataset}_seq_size_{seq_size}_horizon_{horizon}_seed_{seed}"
        
        wandb_instance_name = config.experiment.dir_ckpt
        
        # Configurar Trainer con WandB logger
        trainer = L.Trainer(
            accelerator=accelerator,
            precision=cst.PRECISION,
            max_epochs=config.experiment.max_epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=True, min_delta=0.002),
                TQDMProgressBar(refresh_rate=1000)
            ],
            num_sanity_val_steps=0,
            logger=wandb_logger,  # WandB logger
            detect_anomaly=False,
            check_val_every_n_epoch=1,
        )
        
        # Log simulation details in WANDB console
        run.log({"model": config.model.type.value}, commit=False)
        run.log({"dataset": config.dataset.type.value}, commit=False)
        run.log({"seed": config.experiment.seed}, commit=False)
        run.log({"all_features": config.model.hyperparameters_fixed["all_features"]}, commit=False)
        
        # Log LOBSTER-specific details
        if config.dataset.type == cst.DatasetType.LOBSTER:
            for i in range(len(config.dataset.training_stocks)):
                run.log({f"training stock{i}": config.dataset.training_stocks[i]}, commit=False)
            for i in range(len(config.dataset.testing_stocks)):
                run.log({f"testing stock{i}": config.dataset.testing_stocks[i]}, commit=False)
            run.log({"sampling_type": config.dataset.sampling_type.value}, commit=False)
            if config.dataset.sampling_type == SamplingType.TIME:
                run.log({"sampling_time": config.dataset.sampling_time}, commit=False)
            elif config.dataset.sampling_type == SamplingType.QUANTITY:
                run.log({"sampling_quantity": config.dataset.sampling_quantity}, commit=False)
        
        # Entrenar
        train(config, trainer, run)
        run.finish()
    
    return wandb_sweep_callback


def sweep_init(config: Config):
    """
    INICIALIZA CONFIGURACIÓN PARA HYPERPARAMETER SWEEP
    ===================================================
    
    Configura WandB sweep (grid search) para hyperparameter tuning.
    
    Args:
        config (Config): Configuración del experimento
    
    Returns:
        dict: Configuración de sweep para WandB
    
    Sweep Strategy:
    ---------------
    - Method: Grid (explora todas las combinaciones)
    - Metric: val_loss (minimizar)
    - Early terminate: Hyperband (para runs malos)
    - Run cap: 100 (máximo de runs)
    
    Ejemplo:
    --------
    ```python
    sweep_config = sweep_init(config)
    # {
    #   'method': 'grid',
    #   'metric': {'goal': 'minimize', 'name': 'val_loss'},
    #   'parameters': {
    #     'num_layers': {'values': [4, 6]},
    #     'hidden_dim': {'values': [128, 256]},
    #     'lr': {'values': [0.0001]}
    #   }
    # }
    
    # Ejecutar sweep
    sweep_id = wandb.sweep(sweep_config, project="TLOB")
    wandb.agent(sweep_id, run_wandb(config, "gpu"), count=10)
    ```
    
    Hiperparámetros Típicos:
    ------------------------
    TLOB:
    - num_layers: [4, 6]
    - hidden_dim: [128, 256]
    - num_heads: [1, 8]
    - lr: [0.0001]
    
    MLPLOB:
    - num_layers: [3, 6]
    - hidden_dim: [128]
    - lr: [0.0003]
    
    Total Runs:
    -----------
    - TLOB: 2 × 2 × 2 × 1 = 8 runs
    - MLPLOB: 2 × 1 × 1 = 2 runs
    
    Early Termination:
    ------------------
    - Type: Hyperband
    - min_iter: 3 (mínimo 3 épocas antes de terminar)
    - eta: 1.5 (factor de terminación)
    
    Si un run tiene val_loss mucho peor que el mejor run,
    se termina early para ahorrar tiempo.
    """
    # Put your wandb key here
    wandb.login("")
    
    # Convertir hyperparameters_sweep a formato WandB
    parameters = {}
    for key in config.model.hyperparameters_sweep.keys():
        parameters[key] = {'values': list(config.model.hyperparameters_sweep[key])}
    
    # Configuración de sweep
    sweep_config = {
        'method': 'grid',  # Explorar todas las combinaciones
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,  # Mínimo 3 épocas
            'eta': 1.5  # Factor de terminación
        },
        'run_cap': 100,  # Máximo 100 runs
        'parameters': {**parameters}
    }
    
    return sweep_config


def print_setup(config: Config):
    """
    IMPRIME CONFIGURACIÓN DEL EXPERIMENTO
    ======================================
    
    Muestra todos los parámetros del experimento en console.
    Útil para debugging y verificación de configuración.
    
    Args:
        config (Config): Configuración del experimento
    
    Output Ejemplo:
    ---------------
    ```
    Model type: TLOB
    Dataset: BTC
    Seed: 42
    Sequence size: 128
    Horizon: 10
    All features: True
    Is data preprocessed: True
    Is wandb: False
    Is sweep: False
    ['TRAINING']
    Is debug: False
    ```
    """
    print("Model type: ", config.model.type)
    print("Dataset: ", config.dataset.type)
    print("Seed: ", config.experiment.seed)
    print("Sequence size: ", config.model.hyperparameters_fixed["seq_size"])
    print("Horizon: ", config.experiment.horizon)
    print("All features: ", config.model.hyperparameters_fixed["all_features"])
    print("Is data preprocessed: ", config.experiment.is_data_preprocessed)
    print("Is wandb: ", config.experiment.is_wandb)
    print("Is sweep: ", config.experiment.is_sweep)
    print(config.experiment.type)
    print("Is debug: ", config.experiment.is_debug)
    
    # LOBSTER-specific info
    if config.dataset.type == cst.DatasetType.LOBSTER:
        print("Training stocks: ", config.dataset.training_stocks)
        print("Testing stocks: ", config.dataset.testing_stocks)
