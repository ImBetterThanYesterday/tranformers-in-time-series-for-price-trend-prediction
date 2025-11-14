## TLOB Knowledge Base

### 1. Arquitectura general del repo
- Entrada única `main.py` con Hydra; selecciona dataset/modelo/experimento y prepara datos según `config/config.py`.
- `preprocessing/*` descarga o transforma datos crudos (`FI_2010`, `BTC`, `LOBSTER`) y guarda `train/val/test.npy`.
- `run.py` arma `Lightning Trainer`, construye `Dataset` deslizante y orquesta train/val/test; todos los modelos comparten el mismo `Engine`.
- `models/` contiene implementaciones de referencia (TLOB, MLPLOB, DeepLOB, BiN-CTABL) y utilidades como `BiN` para normalización bilineal.

### 2. Flujo de datos por dataset
- **FI-2010** (`data/FI_2010/*.txt`): 40 features de LOB + 104 engineered + 5 filas de etiquetas (h=10,20,30,50,100). El loader selecciona 40 o 144 features y alinea etiquetas restando 1 para obtener `{0,1,2}`.
- **BTC** (`data/BTC/train.npy`, etc.): generado vía `BTCDataBuilder`. Cada fila = 40 columnas de LOB normalizado (10 niveles × 4) + columnas `label_h{10,20,50,100}` creadas con `utils.utils_data.labeling`.
- **LOBSTER (TSLA/INTC)** (`data/<TICKER>/*.npy`): combina mensajes normalizados (time,event_type,size,price,direction,depth) y 40 columnas LOB. Cuando `all_features=True`, los modelos reciben 46 canales; si es `False`, solo 40 LOB.

```11:32:preprocessing/dataset.py
def __getitem__(self, i):
    input = self.x[i:i+self.seq_size, :]
    return input, self.y[i]
```
- Cada fila de `.npy` se convierte en ventanas deslizantes de tamaño `seq_size`; la etiqueta es el valor alineado al último timestamp de la ventana.

### 3. Entradas esperadas por modelo
| Modelo | Tensor de entrada | Notas claves |
| --- | --- | --- |
| `MLPLOB` | `[batch, seq_size, num_features]` | En LOBSTER extrae `order_type` (`input[:,:,41]`), aplica embedding y vuelve a concatenar antes de BiN + bloques MLP duales. |
| `TLOB` | `[batch, seq_size, num_features]` → lineal a `hidden_dim` | BiN → proyección → embedding posicional. Alterna `TransformerLayer` temporal y espacial, luego aplana y pasa por capas densas. |
| `BiN-CTABL`, `DeepLOB` | Mismo formato | Internamente permutan a `[batch, features, seq]` (BiN-CTABL) o `[batch, 1, seq, features]` (DeepLOB) para conv/TABL. |

### 4. Inside TLOB: pipeline de inferencia
1. **Normalización bilineal** (`BiN(num_features, seq_size)`) alinea estadísticas temporal y espacialmente.
2. **Embedding inicial**: `nn.Linear(num_features, hidden_dim)` + suma de embedding posicional sinusoidal (opción `is_sin_emb`).
3. **Bloques duales**:
   - Capa temporal: `nn.MultiheadAttention` opera sobre snapshots consecutivos.
   - Capa espacial: misma lógica pero permutando a `[batch, features, seq]` para capturar relaciones entre características del LOB.
4. **Head final**: reacomoda a vector (`hidden_dim//4 * seq_size//4`), pasa por capas densas `Linear+GELU` hasta producir logits tamaño 3.

```52:121:models/tlob.py
        x = rearrange(x, 'b s f -> b f s')
        x = self.norm_layer(x)
        x = rearrange(x, 'b f s -> b s f')
        x = self.emb_layer(x)
        x = x[:] + self.pos_encoder
        for i in range(len(self.layers)):
            x, att = self.layers[i](x)
            x = x.permute(0, 2, 1)
        x = rearrange(x, 'b s f -> b (f s) 1')
        x = x.reshape(x.shape[0], -1)
```
- **Inferencia**: basta cargar una ventana `[seq_size, num_features]` (por ej. `torch.tensor(window).unsqueeze(0)`), llamar al checkpoint `Engine.load_from_checkpoint(...).model(window)` y aplicar `softmax`.

### 5. Configuración y ejecución (BTC + TLOB)
1. Instalar deps (CPU):  
   ```bash
   python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python3 -m pip install einops hydra-core lightning lion-pytorch matplotlib numpy \
     omegaconf pandas pytorch_lightning requests scikit-learn scipy seaborn \
     torch-ema transformers wandb kagglehub onnx onnxruntime
   ```
2. Preparar datos y entrenar (usa KaggleHub para descargar BTC):  
   ```bash
   python3 main.py +model=tlob +dataset=btc \
     experiment.type='[TRAINING]' experiment.horizon=10 \
     experiment.max_epochs=10 hydra.job.chdir=False
   ```
   - Al primer run `config.experiment.is_data_preprocessed=False`; `BTCDataBuilder` guardará `train/val/test.npy`.
   - Checkpoints & métricas se guardan en `data/checkpoints/TLOB/BTC_seq_size_128_horizon_<h>_seed_<seed>/`.
3. Inferencia rápida (ejemplo):
   ```python
   import torch, numpy as np
   from models.engine import Engine
   engine = Engine.load_from_checkpoint(
       "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_42/pt/val_loss=0.624_epoch=2.pt",
       seq_size=128, horizon=10, max_epochs=10, model_type="TLOB",
       is_wandb=False, experiment_type="EVALUATION", lr=1e-4, optimizer="Adam",
       dir_ckpt="", hidden_dim=40, num_layers=4, num_features=40,
       dataset_type="BTC", num_heads=1, is_sin_emb=True, len_test_dataloader=1)
   window = torch.from_numpy(np.load('data/BTC/test.npy')[:128,:40]).unsqueeze(0).float()
   logits = engine.model(window)
   probs = torch.softmax(logits, dim=-1)
   ```

---

## 6. Scripts de Inferencia y Análisis de Datos

### 🎯 Scripts Creados

El repositorio ahora incluye scripts completos para inferencia y análisis:

```
TLOB-main/
├── inference_pytorch.py      # Inferencia con PyTorch (checkpoint .pt)
├── inference_onnx.py          # Inferencia con ONNX Runtime (optimizado)
├── extract_examples.py        # Extraer ejemplos personalizados
├── inspect_data.py            # Inspeccionar y visualizar datos
└── docs/
    └── inference_guide.md     # Documentación detallada de inferencia
```

### 📊 Formato de Entrada Detallado

**Archivos .npy del dataset BTC:**
```python
# Shape de los datos
train.npy: (2,780,963 timesteps, 44 features)
val.npy:   (344,454 timesteps, 44 features)
test.npy:  (605,453 timesteps, 44 features)

# Composición de las 44 features:
Features 0-39:  LOB (Limit Order Book)
  ├─ 0-9:   ASK Prices (10 niveles)
  ├─ 10-19: ASK Volumes (10 niveles)
  ├─ 20-29: BID Prices (10 niveles)
  └─ 30-39: BID Volumes (10 niveles)

Features 40-43: Metadata adicional (no usada en entrenamiento)

# ⚠️ IMPORTANTE: El modelo fue entrenado con SOLO 40 features
# Los scripts de inferencia extraen [:, :, :40] automáticamente
```

**Ventanas de entrada al modelo:**
```python
# Cada predicción requiere una ventana temporal
seq_size = 128  # 128 snapshots consecutivos del LOB

# Shape esperado por el modelo:
input_shape = (batch_size, 128, 40)

# Ejemplo:
X = np.array([...])  # Shape: (5, 128, 40)
# → 5 ejemplos, cada uno con 128 timesteps, 40 features LOB
```

### 🚀 Uso de los Scripts

#### 1. Extraer ejemplos personalizados
```bash
# Extraer 5 ejemplos aleatorios
python3 extract_examples.py --split train --num 5 --random

# Extraer ejemplos específicos
python3 extract_examples.py --split train --indices 0 1000 2000 3000 4000

# Extraer ventanas consecutivas
python3 extract_examples.py --split test --num 10 --consecutive --start 5000
```

#### 2. Inferencia con PyTorch
```bash
python3 inference_pytorch.py

# Salida esperada:
# ✓ Modelo cargado: 1,135,974 parámetros
# ✓ Predicciones guardadas en inference_results/
```

#### 3. Inferencia con ONNX (más rápido)
```bash
python3 inference_onnx.py

# Rendimiento:
# → Tiempo promedio: 2.94 ± 0.14 ms
# → Throughput: 1,699.7 ejemplos/segundo
# → Latencia por ejemplo: 0.59 ms
```

#### 4. Inspeccionar datos
```bash
python3 inspect_data.py

# Genera:
# - inspection_results/feature_distributions.png
# - inspection_results/temporal_evolution.png
# - inspection_results/window_heatmap.png
```

### 🎯 Resultados de Inferencia Real (BTC)

**Ejecutado el 14-Nov-2025 sobre 5 ejemplos:**

| Ejemplo | Predicción | Confianza | Logits [DOWN, STAT, UP] |
|---------|------------|-----------|-------------------------|
| 1 | STATIONARY | 92.30% | [-0.163, **2.429**, -2.331] |
| 2 | STATIONARY | 98.96% | [-1.853, **3.499**, -1.662] |
| 3 | STATIONARY | 98.90% | [-1.020, **3.661**, -2.629] |
| 4 | STATIONARY | 96.68% | [-2.559, **3.036**, -0.451] |
| 5 | STATIONARY | 98.99% | [-2.475, **3.666**, -1.150] |

**Observaciones clave:**
- ✅ Confianza muy alta en todos los casos (>92%)
- ✅ Clase STATIONARY domina con logits positivos (+2.4 a +3.7)
- ⚠️ Clases DOWN y UP fuertemente suprimidas (logits negativos)
- 💡 El modelo predice estabilidad de precio en horizonte de 10 timesteps

### 📚 Documentación Adicional

Ver `docs/inference_guide.md` para:
- Arquitectura detallada del modelo TLOB
- Flujo de datos paso a paso
- Ejemplo de integración en sistemas de trading
- Métricas y benchmarks completos
- Limitaciones y consideraciones

---

## 7. Ejecución real del entrenamiento (BTC · 2025-11-14)
- **Instalación**: fue necesario instalar PyTorch CPU y todos los paquetes de `requirements.txt`. Algunos scripts quedaron en `~/Library/Python/3.9/bin`; añade esa ruta al `PATH` si quieres usar `wandb`, `hf` CLI, etc.
- **Descarga BTC**: el primer `main.py ... +dataset=btc` con `experiment.is_data_preprocessed=False` ejecuta `BTCDataBuilder` (descarga Kaggle → genera 12 CSV diarios → normaliza → produce `data/BTC/{train,val,test}.npy`).
- **Problema wandb**: al no tener API key configurada, el run con `experiment.is_wandb=True` abortó. Solución: volver a lanzar con `experiment.is_wandb=False experiment.is_data_preprocessed=True`.
- **Comando final ejecutado**  
  ```bash
  python3 main.py +model=tlob +dataset=btc \
      experiment.is_wandb=False \
      experiment.is_data_preprocessed=True \
      hydra.job.chdir=False
  ```
  - Usa hiperparámetros por defecto del config: `seq_size=128`, `hidden_dim=40`, `num_layers=4`, `num_heads=1`, `lr=1e-4`, `max_epochs=10`, `horizon=10`, `seed=1`.
- **Checkpoints generados**: `data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/`
  - Mejor checkpoint: `pt/val_loss=0.623_epoch=2.pt`
  - ONNX exportado: `onnx/val_loss=0.623_epoch=2.onnx`
  - `predictions.npy` y curva PR en el mismo directorio.
- **Métricas**  
  - Entrenamiento converge rápido (val_loss 0.63 tras epoch 0; 0.623 en epoch 2). LR se reduce automáticamente cuando la mejora < 0.002.
  - Test (h=10) luego de seleccionar el mejor checkpoint:  
    | Métrica | Valor |
    | --- | --- |
    | accuracy | 0.780 |
    | macro F1 | 0.744 |
    | macro precision | 0.817 |
    | macro recall | 0.709 |
    | `test_loss` | 0.5547 |
    | Soporte clases | `{0:138,058 · 1:326,025 · 2:141,233}` |
  - Matriz de desempeño por clase (test):  
    - Clase 0 (up): P=0.856, R=0.596, F1=0.703  
    - Clase 1 (stable): P=0.746, R=0.940, F1=0.832  
    - Clase 2 (down): P=0.848, R=0.592, F1=0.697
- **Re-uso del checkpoint**: Usa el snippet de inferencia anterior cambiando la ruta del `.pt` y `len_test_dataloader` → `math.ceil(test_len / (batch*4))` si vas a evaluar dentro de Lightning; para inferencia manual solo necesitas `Engine.model`.
- **Recomendaciones**:
  - Para reentrenar desde cero sin descargar de nuevo la data: `experiment.is_data_preprocessed=True`.
  - Si deseas repetir el run con otro horizonte: cambia `experiment.horizon`, pero también ajusta `config.experiment.dir_ckpt` o `seed` para no sobrescribir.

### 6. Observaciones útiles
- `config.model.hyperparameters_fixed.seq_size` cambia según dataset (FI-2010 → 128, BTC → 128, LOBSTER → 128 por defecto; MLPLOB usa 384 en BTC).
- `utils.utils_data.labeling` define etiquetas ternarias usando ventanas suavizadas y `alpha = mean(|Δ%|)/2`; cambiar `alpha` permite emular el experimento de spread promedio.
- `run.py` imprime distribución de clases para train/val/test, útil para diagnosticar umbrales.
- Checkpoints incluyen exportación ONNX automática (`data/checkpoints/TLOB/.../onnx/*.onnx`), aprovechable para despliegue.



