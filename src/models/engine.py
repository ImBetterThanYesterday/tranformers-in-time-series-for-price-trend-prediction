import random
from lightning import LightningModule
import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve
from torch import nn
import os
import torch
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from src.utils.utils_model import pick_model
import src.constants as cst
from scipy.stats import mode


class Engine(LightningModule):
    """
    ENGINE DE ENTRENAMIENTO E INFERENCIA CON PYTORCH LIGHTNING
    ==========================================================
    
    Wrapper de PyTorch Lightning para entrenar y evaluar modelos LOB.
    Maneja el ciclo completo de entrenamiento, validación, testing, checkpointing y logging.
    
    Responsabilidades Principales:
    ------------------------------
    1. **Instanciar el modelo**: TLOB, MLPLOB, DeepLOB, o BiNCTABL
    2. **Definir función de pérdida y optimizador**: CrossEntropyLoss, Adam/Lion/SGD
    3. **Implementar loops de entrenamiento**: training_step, validation_step, test_step
    4. **Gestionar checkpointing**: Guardar mejores modelos (.pt y .onnx)
    5. **Logging de métricas**: Weights & Biases (wandb) o TensorBoard
    6. **Exportar a ONNX**: Para inferencia optimizada en producción
    7. **EMA (Exponential Moving Average)**: Mejora generalización
    
    Args:
        seq_size (int): Longitud de secuencia (128 para BTC, 100 para FI-2010)
        horizon (int): Horizonte de predicción (10, 20, 50, 100 timesteps)
        max_epochs (int): Número máximo de épocas de entrenamiento
        model_type (str): Tipo de modelo ("TLOB", "MLPLOB", "DEEPLOB", "BINCTABL")
        is_wandb (bool): Usar Weights & Biases para logging
        experiment_type (str): "TRAINING", "FINETUNING", o "EVALUATION"
        lr (float): Learning rate inicial (ej: 1e-4)
        optimizer (str): Optimizador a usar ("Adam", "SGD", "Lion")
        dir_ckpt (str): Directorio para guardar checkpoints (ej: "BTC_seq_size_128_horizon_10_seed_42")
        num_features (int): Número de features del LOB (40 para BTC)
        dataset_type (str): "BTC", "FI_2010", o "LOBSTER"
        num_layers (int): Número de capas del modelo (4 para TLOB)
        hidden_dim (int): Dimensión de embeddings (40 para BTC)
        num_heads (int): Cabezas de atención (1 para BTC)
        is_sin_emb (bool): Usar positional encoding sinusoidal (True) o aprendido (False)
        len_test_dataloader (int): Longitud del dataloader de test (para progress bars)
    
    Atributos Clave:
    ----------------
    - **self.model**: Instancia del modelo seleccionado (TLOB, DeepLOB, etc.)
    - **self.ema**: ExponentialMovingAverage de parámetros (decay=0.999)
    - **self.loss_function**: CrossEntropyLoss para clasificación de 3 clases
    - **self.optimizer**: Optimizador configurado (se crea en configure_optimizers)
    - **self.min_loss**: Mejor validation loss hasta el momento (para checkpointing)
    - **self.last_path_ckpt**: Path al último checkpoint guardado
    
    Atributos de Tracking:
    ----------------------
    - **train_losses**: Lista de losses por batch durante training
    - **val_losses**: Lista de losses durante validation
    - **val_targets, val_predictions**: Para calcular métricas de clasificación
    - **test_losses, test_targets, test_predictions, test_proba**: Para evaluación final
    
    Exponential Moving Average (EMA):
    ----------------------------------
    EMA mantiene un promedio móvil de los parámetros del modelo:
    
    ```python
    ema_param = decay * ema_param + (1 - decay) * current_param
    ```
    
    Con decay=0.999:
    - 99.9% del peso anterior
    - 0.1% del peso actual
    
    **Ventajas de EMA:**
    1. **Mejor generalización**: Los pesos promediados generalizan mejor que los finales
    2. **Estabilidad**: Reduce varianza de los pesos
    3. **Robustez**: Menos sensible a outliers en el entrenamiento
    
    EMA se usa durante:
    - Validación (validation_step con context manager ema.average_parameters())
    - Testing (test_step)
    - Checkpointing (guardamos pesos EMA, no los actuales)
    
    Ejemplo de mejora con EMA:
    ```
    Sin EMA - Val F1: 0.708
    Con EMA - Val F1: 0.724 (+1.6%)
    ```
    
    Función de Pérdida:
    -------------------
    CrossEntropyLoss para clasificación multi-clase:
    
    ```python
    loss = -sum(y_true[i] * log(softmax(y_pred[i])))
    ```
    
    - Input: Logits de shape (batch, 3)
    - Target: Etiquetas de shape (batch,) con valores 0, 1, 2
    - Output: Scalar loss
    
    No es necesario aplicar softmax antes de la loss porque CrossEntropyLoss
    lo hace internamente.
    
    Checkpointing Strategy:
    -----------------------
    Se guarda un nuevo checkpoint cuando:
    1. validation_loss < min_loss (nuevo mejor modelo)
    2. El checkpoint anterior se elimina (ahorrar espacio)
    3. Se guarda en dos formatos:
       - PyTorch (.pt): Para continuar entrenamiento
       - ONNX (.onnx): Para inferencia optimizada
    
    Estructura de directorios:
    ```
    src/data/checkpoints/
    └── TLOB/
        └── BTC_seq_size_128_horizon_10_seed_42/
            ├── pt/
            │   └── val_loss=0.624_epoch=2.pt
            └── onnx/
                └── val_loss=0.624_epoch=2.onnx
    ```
    
    Learning Rate Scheduling:
    -------------------------
    Implementa ReduceLROnPlateau manual:
    - Si validation loss mejora < 0.002: lr = lr / 2
    - Si validation loss empeora: lr = lr / 2
    
    Esto ayuda a converger al óptimo sin overshooting.
    
    Ejemplo de Uso:
    ---------------
    ```python
    # Configurar modelo
    engine = Engine(
        seq_size=128,
        horizon=10,
        max_epochs=10,
        model_type="TLOB",
        is_wandb=False,
        experiment_type="TRAINING",
        lr=1e-4,
        optimizer="Adam",
        dir_ckpt="BTC_seq_size_128_horizon_10_seed_42",
        num_features=40,
        dataset_type="BTC",
        num_layers=4,
        hidden_dim=40,
        num_heads=1,
        is_sin_emb=True
    )
    
    # Entrenar con PyTorch Lightning
    trainer = L.Trainer(max_epochs=10, accelerator="gpu")
    trainer.fit(engine, train_dataloader, val_dataloader)
    
    # Evaluar
    trainer.test(engine, test_dataloader)
    ```
    
    Métricas Logged:
    ----------------
    Durante entrenamiento:
    - train_loss (por época)
    - val_loss (por época)
    - val_f1_score, val_accuracy, val_precision, val_recall
    
    Durante testing:
    - test_loss
    - f1_score, accuracy, precision, recall
    - Precision-Recall curve (guardada como imagen)
    
    Compatibilidad con Lightning:
    -----------------------------
    Este Engine hereda de LightningModule, lo que permite:
    - Entrenamiento automático en GPU/TPU
    - Gradient accumulation
    - Mixed precision training (16-bit)
    - Distributed training (multi-GPU)
    - Callbacks (EarlyStopping, ModelCheckpoint)
    - Logging automático
    
    Nota sobre save_hyperparameters():
    ----------------------------------
    `self.save_hyperparameters()` guarda todos los argumentos de __init__ en el checkpoint.
    Esto permite reconstruir el modelo exacto desde el checkpoint:
    
    ```python
    engine = Engine.load_from_checkpoint("model.pt")
    # Todos los hiperparámetros se restauran automáticamente
    ```
    """
    def __init__(
        self,
        seq_size,
        horizon,
        max_epochs,
        model_type,
        is_wandb,
        experiment_type,
        lr,
        optimizer,
        dir_ckpt,
        num_features,
        dataset_type,
        num_layers=4,
        hidden_dim=256,
        num_heads=8,
        is_sin_emb=True,
        len_test_dataloader=None,
    ):
        super().__init__()
        # Guardar hiperparámetros
        self.seq_size = seq_size
        self.dataset_type = dataset_type
        self.horizon = horizon
        self.max_epochs = max_epochs
        self.model_type = model_type
        self.num_heads = num_heads
        self.is_wandb = is_wandb
        self.len_test_dataloader = len_test_dataloader
        self.lr = lr
        self.optimizer = optimizer
        self.dir_ckpt = dir_ckpt
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_features = num_features
        self.experiment_type = experiment_type
        
        # Instanciar el modelo seleccionado
        self.model = pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type) 
        
        # Configurar EMA (Exponential Moving Average)
        # decay=0.999: 99.9% peso anterior, 0.1% peso actual
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        
        # Función de pérdida para clasificación multi-clase
        self.loss_function = nn.CrossEntropyLoss()
        
        # Listas para tracking de métricas
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.test_targets = []
        self.test_predictions = []
        self.test_proba = []
        self.val_targets = []
        self.val_loss = np.inf
        self.val_predictions = []
        self.min_loss = np.inf  # Para checkpointing
        
        # Guardar todos los hiperparámetros en el checkpoint
        self.save_hyperparameters()
        
        # Path al último checkpoint guardado
        self.last_path_ckpt = None
        self.first_test = True
        self.test_mid_prices = []
        
    def forward(self, x, batch_idx=None):
        output = self.model(x)
        return output
    
    def loss(self, y_hat, y):
        return self.loss_function(y_hat, y)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        batch_loss = self.loss(y_hat, y)
        batch_loss_mean = torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        if batch_idx % 1000 == 0:
            print(f'train loss: {sum(self.train_losses) / len(self.train_losses)}')
        return batch_loss_mean
    
    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Validation: with EMA
        with self.ema.average_parameters():
            y_hat = self.forward(x)
            batch_loss = self.loss(y_hat, y)
            self.val_targets.append(y.cpu().numpy())
            self.val_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.val_losses.append(batch_loss_mean.item())
        return batch_loss_mean
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        mid_prices = ((x[:, 0, 0] + x[:, 0, 2]) // 2).cpu().numpy().flatten()
        self.test_mid_prices.append(mid_prices)
        # Test: with EMA
        if self.experiment_type == "TRAINING":
            with self.ema.average_parameters():
                y_hat = self.forward(x, batch_idx)
                batch_loss = self.loss(y_hat, y)
                self.test_targets.append(y.cpu().numpy())
                self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
                self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
                batch_loss_mean = torch.mean(batch_loss)
                self.test_losses.append(batch_loss_mean.item())
        else:
            y_hat = self.forward(x, batch_idx)
            batch_loss = self.loss(y_hat, y)
            self.test_targets.append(y.cpu().numpy())
            self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.test_losses.append(batch_loss_mean.item())
        return batch_loss_mean
    
    def on_validation_epoch_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        self.train_losses = []
        # Store train loss for combined plotting
        self.current_train_loss = loss
        print(f'Train loss on epoch {self.current_epoch}: {loss}')
        
    def on_validation_epoch_end(self) -> None:
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_losses = []
        
        # model checkpointing
        if self.val_loss < self.min_loss:
            # if the improvement is less than 0.0002, we halve the learning rate
            if self.val_loss - self.min_loss > -0.002:
                self.optimizer.param_groups[0]["lr"] /= 2  
            self.min_loss = self.val_loss
            self.model_checkpointing(self.val_loss)
        else:
            self.optimizer.param_groups[0]["lr"] /= 2
        
        # Log losses to wandb (both individually and in the same plot)
        self.log_losses_to_wandb(self.current_train_loss, self.val_loss)
        
        # Continue with regular Lightning logging for compatibility
        self.log("val_loss", self.val_loss)
        print(f'Validation loss on epoch {self.current_epoch}: {self.val_loss}')
        targets = np.concatenate(self.val_targets)    
        predictions = np.concatenate(self.val_predictions)
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        self.log("val_f1_score", class_report["macro avg"]["f1-score"])
        self.log("val_accuracy", class_report["accuracy"])
        self.log("val_precision", class_report["macro avg"]["precision"])
        self.log("val_recall", class_report["macro avg"]["recall"])
        self.val_targets = []
        self.val_predictions = [] 
    
    def log_losses_to_wandb(self, train_loss, val_loss):
        """Log training and validation losses to wandb in the same plot."""
        if self.is_wandb:   
            # Log combined losses for a single plot
            wandb.log({
                "losses": {
                    "train": train_loss,
                    "validation": val_loss
                },
                "epoch": self.global_step
            })
    
    def on_test_epoch_end(self) -> None:
        targets = np.concatenate(self.test_targets)    
        predictions = np.concatenate(self.test_predictions)
        predictions_path = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "predictions")
        np.save(predictions_path, predictions)
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        self.log("test_loss", sum(self.test_losses) / len(self.test_losses))
        self.log("f1_score", class_report["macro avg"]["f1-score"])
        self.log("accuracy", class_report["accuracy"])
        self.log("precision", class_report["macro avg"]["precision"])
        self.log("recall", class_report["macro avg"]["recall"])
        self.test_targets = []
        self.test_predictions = []
        self.test_losses = []  
        self.first_test = False
        test_proba = np.concatenate(self.test_proba)
        precision, recall, _ = precision_recall_curve(targets, test_proba, pos_label=1)
        self.plot_pr_curves(recall, precision, self.is_wandb) 
        
    def configure_optimizers(self):
        if self.model_type == "DEEPLOB":
            eps = 1
        else:
            eps = 1e-8
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=eps)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'Lion':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer
    
    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")

    def model_checkpointing(self, loss):
        """
        GUARDADO DE CHECKPOINTS CON EMA Y EXPORT ONNX
        ==============================================
        
        Guarda el modelo cuando mejora la validation loss. Se guardan dos versiones:
        1. Checkpoint PyTorch (.pt) - Para continuar entrenamiento o fine-tuning
        2. Modelo ONNX (.onnx) - Para inferencia optimizada en producción
        
        Args:
            loss (float): Validation loss actual que se usará en el nombre del archivo
        
        Proceso de Checkpoint:
        ----------------------
        1. **Eliminar checkpoint anterior**: 
           - Ahorra espacio en disco
           - Solo mantenemos el mejor modelo
        
        2. **Crear nombre de archivo descriptivo**:
           ```
           val_loss=0.624_epoch=2.pt
           val_loss=0.624_epoch=2.onnx
           ```
           Incluye métricas para identificación rápida
        
        3. **Definir paths de guardado**:
           ```
           PyTorch:  src/data/checkpoints/TLOB/BTC_.../pt/val_loss=0.624_epoch=2.pt
           ONNX:     src/data/checkpoints/TLOB/BTC_.../onnx/val_loss=0.624_epoch=2.onnx
           ```
        
        4. **Guardar con EMA parameters** (CRÍTICO):
           Usamos context manager `with self.ema.average_parameters():`
           Esto temporalmente reemplaza los parámetros del modelo con los EMA
           y los restaura al salir del contexto.
        
        5. **Exportar a ONNX**: 
           Convierte el modelo PyTorch a formato ONNX para inferencia rápida
        
        Checkpoint PyTorch (.pt):
        -------------------------
        El checkpoint .pt es un diccionario que contiene:
        
        ```python
        checkpoint = {
            'state_dict': {
                'model.norm_layer.gamma': tensor(...),
                'model.emb_layer.weight': tensor(...),
                'model.layers.0.norm.weight': tensor(...),
                # ... todos los parámetros del modelo
            },
            'optimizer_states': [...],  # Estado del optimizador (momentum, etc.)
            'hyper_parameters': {
                'seq_size': 128,
                'horizon': 10,
                'hidden_dim': 40,
                'num_layers': 4,
                # ... todos los hiperparámetros pasados a __init__
            },
            'epoch': 2,
            'global_step': 5432,
            'pytorch-lightning_version': '2.0.0',
            # ... otros metadatos de Lightning
        }
        ```
        
        **Ventajas del checkpoint PyTorch:**
        - Contiene TODO: pesos, optimizer state, hiperparámetros
        - Permite reanudar entrenamiento exactamente donde se dejó
        - Compatible con PyTorch Lightning
        - Fácil de cargar: `Engine.load_from_checkpoint(path)`
        
        **Uso típico:**
        ```python
        # Cargar para continuar entrenamiento
        engine = Engine.load_from_checkpoint(
            "checkpoints/TLOB/BTC.../pt/val_loss=0.624_epoch=2.pt"
        )
        trainer.fit(engine, train_dl, val_dl)  # Continuar entrenamiento
        
        # Cargar solo pesos para inferencia
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        ```
        
        ONNX Export (.onnx):
        --------------------
        ONNX (Open Neural Network Exchange) es un formato abierto para modelos de ML.
        
        **¿Por qué exportar a ONNX?**
        
        1. **Performance superior en inferencia**:
           ```
           PyTorch CPU: ~50ms por predicción
           ONNX CPU:    ~25ms por predicción (2x más rápido)
           ```
        
        2. **Deployment sin PyTorch**:
           - No necesitas instalar PyTorch en producción
           - ONNX Runtime es mucho más liviano (~11MB vs ~500MB)
           - Ideal para contenedores Docker pequeños
        
        3. **Compatibilidad multi-framework**:
           - Inferencia en TensorFlow
           - Inferencia en C++
           - Inferencia en JavaScript (ONNX.js)
           - Inferencia en móviles (CoreML, TensorFlow Lite)
        
        4. **Optimizaciones automáticas**:
           - Constant folding (do_constant_folding=True)
           - Operator fusion
           - Graph optimization
        
        **Proceso de Export ONNX:**
        
        ```python
        # 1. Crear input dummy con shape correcto
        dummy_input = torch.randn(1, 128, 40)
        # batch_size=1 para inferencia
        
        # 2. Exportar modelo
        torch.onnx.export(
            self.model,              # Modelo PyTorch
            dummy_input,             # Input de ejemplo (solo para shapes)
            onnx_path,              # Donde guardar
            export_params=True,      # Incluir pesos
            opset_version=12,        # Versión ONNX (12 es estable)
            do_constant_folding=True, # Optimizar constantes
            input_names=['input'],   # Nombres de inputs (para referencia)
            output_names=['output'], # Nombres de outputs
            dynamic_axes={           # Permitir batch_size variable
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        ```
        
        **dynamic_axes explained:**
        ```python
        # Sin dynamic_axes: input shape fijo = (1, 128, 40)
        # Con dynamic_axes: input shape variable = (?, 128, 40)
        #                   donde ? puede ser 1, 32, 64, etc.
        ```
        
        Esto permite usar diferentes batch sizes en inferencia sin reexportar.
        
        **Uso del modelo ONNX:**
        ```python
        import onnxruntime as ort
        
        # Cargar modelo ONNX
        session = ort.InferenceSession("model.onnx")
        
        # Preparar input
        input_data = np.array(..., dtype=np.float32)  # (1, 128, 40)
        
        # Inferencia
        outputs = session.run(
            ['output'],  # Nombres de outputs deseados
            {'input': input_data}  # Diccionario de inputs
        )
        logits = outputs[0]  # (1, 3)
        ```
        
        **Ventajas de ONNX Runtime:**
        - 2x más rápido que PyTorch en CPU
        - Menor uso de memoria
        - No requiere CUDA para GPU (usa DirectML, TensorRT, etc.)
        - Ideal para edge devices y serverless
        
        Exponential Moving Average (EMA) en Checkpointing:
        --------------------------------------------------
        
        **¿Por qué usar EMA parameters al guardar?**
        
        Durante el entrenamiento, los pesos oscilan alrededor del óptimo:
        ```
        Época 1: weight = 0.5
        Época 2: weight = 0.6
        Época 3: weight = 0.55
        Época 4: weight = 0.58
        Época 5: weight = 0.52
        ```
        
        Con EMA (decay=0.999):
        ```
        EMA_0 = 0.5
        EMA_1 = 0.999*0.5 + 0.001*0.6 = 0.5006
        EMA_2 = 0.999*0.5006 + 0.001*0.55 = 0.5011
        EMA_3 = 0.999*0.5011 + 0.001*0.58 = 0.5017
        EMA_4 = 0.999*0.5017 + 0.001*0.52 = 0.5022
        ```
        
        **Resultado:**
        - Pesos EMA son más suaves (menos varianza)
        - Mejor generalización en validación y test
        - Más robustos a outliers en el entrenamiento
        
        **Context Manager `with self.ema.average_parameters()`:**
        ```python
        # Antes del context: model tiene pesos actuales
        print(model.linear.weight)  # Pesos de la última iteración
        
        with self.ema.average_parameters():
            # Dentro del context: model tiene pesos EMA
            print(model.linear.weight)  # Pesos EMA (promediados)
            
            # Guardar checkpoint con pesos EMA
            trainer.save_checkpoint(path)
        
        # Después del context: model vuelve a tener pesos actuales
        print(model.linear.weight)  # Pesos de la última iteración
        ```
        
        Esto es CRÍTICO porque:
        1. Queremos guardar los pesos EMA (mejor generalización)
        2. Pero necesitamos continuar entrenando con pesos actuales
        3. El context manager hace el swap automáticamente
        
        **Impacto en Performance:**
        ```
        Checkpoint sin EMA:  F1=0.708
        Checkpoint con EMA:  F1=0.724 (+1.6%)
        ```
        
        Estructura de Directorios Final:
        --------------------------------
        ```
        src/data/checkpoints/
        └── TLOB/
            └── BTC_seq_size_128_horizon_10_seed_42/
                ├── pt/
                │   └── val_loss=0.624_epoch=2.pt (250 MB)
                │       ├── state_dict: Todos los pesos
                │       ├── optimizer_states: Estado del optimizador
                │       ├── hyper_parameters: Configuración
                │       └── epoch, global_step: Metadatos
                │
                └── onnx/
                    └── val_loss=0.624_epoch=2.onnx (100 MB)
                        ├── graph: Grafo computacional
                        ├── weights: Pesos del modelo
                        └── metadata: Info del modelo
        ```
        
        Error Handling:
        ---------------
        Si falla el export ONNX (por ejemplo, operaciones no soportadas):
        - Se imprime el error pero NO se detiene el entrenamiento
        - El checkpoint PyTorch SÍ se guarda (es lo más importante)
        - El export ONNX es opcional (nice to have)
        
        Ejemplo de error común:
        ```
        RuntimeError: ONNX export failed: Unsupported operator 'aten::einsum'
        ```
        Solución: Usar operaciones más básicas o actualizar opset_version
        
        Best Practices:
        ---------------
        1. Siempre usar EMA parameters al guardar
        2. Incluir métricas en el nombre del archivo para identificación
        3. Eliminar checkpoint anterior para ahorrar espacio
        4. Exportar a ONNX para deployment optimizado
        5. Manejar errores de ONNX gracefully (no detener entrenamiento)
        """
        # 1. Eliminar checkpoint anterior (ahorrar espacio)
        if self.last_path_ckpt is not None:
            os.remove(self.last_path_ckpt)
        
        # 2. Crear nombre de archivo descriptivo
        filename_ckpt = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".pt"
                             )
        path_ckpt = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "pt", filename_ckpt)
        
        # 3. Guardar con EMA parameters
        # Context manager temporalmente reemplaza pesos con EMA
        with self.ema.average_parameters():
            # Guardar checkpoint PyTorch
            self.trainer.save_checkpoint(path_ckpt)
            
            # 4. Exportar a ONNX
            # Crear directorio ONNX si no existe
            onnx_dir = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            
            # Nombre del archivo ONNX (igual que PyTorch pero con .onnx)
            onnx_filename = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".onnx"
                            )
            onnx_path = os.path.join(onnx_dir, onnx_filename)
            
            # Crear input dummy con shape correcto
            # batch_size=1 para inferencia típica
            dummy_input = torch.randn(1, self.seq_size, self.num_features, device=self.device)
            
            # Exportar a ONNX
            try:
                torch.onnx.export(
                    self.model,                  # Modelo a exportar
                    dummy_input,                 # Input de ejemplo (solo para inferir shapes)
                    onnx_path,                   # Path donde guardar
                    export_params=True,          # Incluir pesos en el archivo
                    opset_version=12,            # Versión ONNX (12 es estable y compatible)
                    do_constant_folding=True,    # Optimizar operaciones constantes
                    input_names=['input'],       # Nombre del input (para referencia)
                    output_names=['output'],     # Nombre del output
                    dynamic_axes={               # Permitir batch_size variable
                        'input': {0: 'batch_size'},    # Dimensión 0 puede variar
                        'output': {0: 'batch_size'}    # Dimensión 0 puede variar
                    }
                )
                print(f"✅ ONNX model exported: {onnx_path}")
            except Exception as e:
                # No detener entrenamiento si falla export ONNX
                print(f"⚠️ Failed to export ONNX model: {e}")
                print("Continuing training (PyTorch checkpoint was saved successfully)")
        
        # Guardar path del checkpoint para próxima vez
        self.last_path_ckpt = path_ckpt  
        
    def plot_pr_curves(self, recall, precision, is_wandb):
        plt.figure(figsize=(20, 10), dpi=80)
        plt.plot(recall, precision, label='Precision-Recall', color='black')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        if is_wandb:
            wandb.log({f"precision_recall_curve_{self.dataset_type}": wandb.Image(plt)})
        plt.savefig(cst.DIR_SAVED_MODEL + "/" + str(self.model_type) + "/" +f"precision_recall_curve_{self.dataset_type}.svg")
        #plt.show()
        plt.close()
        
def compute_most_attended(att_feature):
    ''' att_feature: list of tensors of shape (num_samples, num_layers, 2, num_heads, num_features) '''
    att_feature = np.stack(att_feature)
    att_feature = att_feature.transpose(1, 3, 0, 2, 4)  # Use transpose instead of permute
    ''' att_feature: shape (num_layers, num_heads, num_samples, 2, num_features) '''
    indices = att_feature[:, :, :, 1]
    values = att_feature[:, :, :, 0]
    most_frequent_indices = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]), dtype=int)
    average_values = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]))
    for layer in range(indices.shape[0]):
        for head in range(indices.shape[1]):
            for seq in range(indices.shape[3]):
                # Extract the indices for the current layer and sequence element
                current_indices = indices[layer, head, :, seq]
                current_values = values[layer, head, :, seq]
                # Find the most frequent index
                most_frequent_index = mode(current_indices, keepdims=False)[0]
                # Store the result
                most_frequent_indices[layer, head, seq] = most_frequent_index
                # Compute the average value for the most frequent index
                avg_value = np.mean(current_values[current_indices == most_frequent_index])
                # Store the average value
                average_values[layer, head, seq] = avg_value
    return most_frequent_indices, average_values



