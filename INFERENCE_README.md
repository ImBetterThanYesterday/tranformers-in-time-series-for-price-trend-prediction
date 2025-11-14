# 🚀 Guía Rápida de Inferencia - TLOB Model

> Scripts listos para ejecutar predicciones con el modelo TLOB entrenado

---

## 📦 Archivos Incluidos

```
TLOB-main/
├── 📄 inference_pytorch.py       # Inferencia con PyTorch
├── 📄 inference_onnx.py           # Inferencia con ONNX (más rápido)
├── 📄 extract_examples.py         # Extraer ejemplos del dataset
├── 📄 inspect_data.py             # Visualizar estructura de datos
├── 📂 docs/
│   ├── knowledge.md               # Documentación completa del proyecto
│   └── inference_guide.md         # Guía detallada de inferencia
└── 📂 data/BTC/
    ├── train.npy                  # Dataset de entrenamiento (933 MB)
    ├── val.npy                    # Dataset de validación (116 MB)
    ├── test.npy                   # Dataset de prueba (203 MB)
    └── inference_examples.npy     # 5 ejemplos ya extraídos
```

---

## ⚡ Quick Start (3 pasos)

### 1️⃣ Verificar que tienes los datos y el checkpoint

```bash
# Verificar datos BTC
ls -lh data/BTC/*.npy

# Verificar checkpoint entrenado
ls -lh data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/*.pt
```

Deberías ver:
- ✅ `train.npy`, `val.npy`, `test.npy` en `data/BTC/`
- ✅ `val_loss=0.623_epoch=2.pt` en `data/checkpoints/.../pt/`

### 2️⃣ (Opcional) Extraer nuevos ejemplos

```bash
# Ya hay 5 ejemplos en data/BTC/inference_examples.npy
# Si quieres extraer otros ejemplos:

python3 extract_examples.py --split train --num 5 --random
```

### 3️⃣ Ejecutar inferencia

```bash
# Opción A: PyTorch (más compatible)
python3 inference_pytorch.py

# Opción B: ONNX (3x más rápido)
python3 inference_onnx.py
```

**Salida esperada:**
```
================================================================================
INFERENCIA CON PYTORCH - MODELO TLOB
================================================================================

✓ Modelo cargado: 1,135,974 parámetros
✓ Ejemplos cargados: (5, 128, 40)

--- Ejemplo 1 ---
  Probabilidades:
    • DOWN:       6.91%
    • STATIONARY: 92.30% ← PREDICCIÓN
    • UP:         0.79%
  Confianza: 92.30%

...

✓ Resultados guardados en: inference_results/
  → predictions_pytorch.npy
  → probabilities_pytorch.npy
  → logits_pytorch.npy
```

---

## 📊 ¿Qué es una "entrada" al modelo?

### Estructura del dato

Cada entrada es una **ventana temporal** del Limit Order Book (LOB):

```python
# Shape de la entrada
input_shape = (batch_size, seq_size, num_features)
            = (5, 128, 40)

# Significado:
# - 5 ejemplos (predicciones en paralelo)
# - 128 timesteps consecutivos (ventana temporal)
# - 40 features del LOB por timestep
```

### Las 40 features del LOB

```
Feature 0-9:   ASK Prices  (10 niveles de profundidad)
Feature 10-19: ASK Volumes (volúmenes de cada nivel ASK)
Feature 20-29: BID Prices  (10 niveles de profundidad)
Feature 30-39: BID Volumes (volúmenes de cada nivel BID)
```

**Ejemplo de un snapshot en el tiempo `t`:**
```python
snapshot = [
    # ASK Prices (precios de venta, de menor a mayor)
    100.50, 100.51, 100.52, ..., 100.59,
    
    # ASK Volumes (cantidad ofrecida en cada precio)
    5.2, 3.1, 2.8, ..., 1.5,
    
    # BID Prices (precios de compra, de mayor a menor)
    100.49, 100.48, 100.47, ..., 100.40,
    
    # BID Volumes (cantidad demandada en cada precio)
    7.3, 4.2, 3.9, ..., 2.1
]
```

---

## 🎯 ¿Qué predice el modelo?

El modelo clasifica la **tendencia del precio** en 3 clases:

| Clase | Significado | Interpretación |
|-------|-------------|----------------|
| **0 - DOWN** | Precio bajará | Tendencia bajista en próximos 10 timesteps |
| **1 - STATIONARY** | Precio estable | Sin cambio significativo |
| **2 - UP** | Precio subirá | Tendencia alcista en próximos 10 timesteps |

**Horizonte de predicción:** 10 timesteps hacia el futuro

---

## 🔍 Explorar los datos visualmente

```bash
python3 inspect_data.py
```

Esto generará visualizaciones en `inspection_results/`:
- `feature_distributions.png` - Histogramas de las primeras 10 features
- `temporal_evolution.png` - Series temporales de 4 features clave
- `window_heatmap.png` - Heatmap de una ventana completa (128×40)

---

## 📈 Resultados Reales

**Inferencia ejecutada el 14-Nov-2025 sobre 5 ejemplos del dataset BTC:**

| Ejemplo | Predicción | Confianza | Interpretación |
|---------|------------|-----------|----------------|
| 1 | STATIONARY | 92.30% | Precio estable con alta confianza |
| 2 | STATIONARY | 98.96% | Precio muy estable |
| 3 | STATIONARY | 98.90% | Precio muy estable |
| 4 | STATIONARY | 96.68% | Precio estable |
| 5 | STATIONARY | 98.99% | Confianza máxima en estabilidad |

**Observación:** Todos los ejemplos predicen estabilidad de precio, lo cual puede indicar que el mercado estaba en un período de baja volatilidad durante esas ventanas temporales.

---

## 🛠️ Uso Avanzado

### Extraer ejemplos específicos

```bash
# Por índices específicos
python3 extract_examples.py --split test --indices 0 5000 10000 15000 20000

# Ventanas consecutivas (sin solapamiento)
python3 extract_examples.py --split train --num 10 --consecutive --start 100000

# Aleatorios del validation set
python3 extract_examples.py --split val --num 20 --random
```

### Cambiar checkpoint (diferentes horizontes)

Edita `inference_pytorch.py` o `inference_onnx.py` y cambia:

```python
# Línea 20 en inference_pytorch.py
CHECKPOINT_PATH = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_20_seed_42/pt/val_loss=0.822_epoch=1.pt"
#                                                              ^^
#                                                   Horizonte de predicción
```

Horizontes disponibles: `10`, `20`, `50`, `100` timesteps

### Inferencia en batch grande

```python
# En tu código Python
import numpy as np
import torch
from models.tlob import TLOB

# Cargar 1000 ventanas
num_examples = 1000
examples = []
for i in range(num_examples):
    window = train_data[i*128:(i+1)*128, :40]
    examples.append(window)

X = torch.from_numpy(np.array(examples)).float()

# Inferencia en batch (eficiente)
with torch.no_grad():
    logits = model(X)  # Shape: (1000, 3)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

# 1000 predicciones en ~3 segundos (CPU)
```

---

## 📚 Documentación Completa

Para información detallada, consulta:

1. **`docs/inference_guide.md`**
   - Arquitectura completa del modelo TLOB
   - Flujo de datos paso a paso
   - Integración en sistemas de trading
   - Métricas y benchmarks

2. **`docs/knowledge.md`**
   - Overview completo del repositorio
   - Pipeline de preprocesamiento
   - Todos los modelos disponibles
   - Comandos de entrenamiento

---

## ❓ FAQ

### ¿Por qué el modelo usa 40 features si los datos tienen 44?

Los datos `.npy` tienen 44 columnas (40 LOB + 4 metadata), pero el modelo fue entrenado solo con las 40 del LOB. Los scripts de inferencia extraen automáticamente `[:, :, :40]`.

### ¿Puedo usar estos scripts en producción?

Sí, especialmente `inference_onnx.py` está optimizado para producción:
- ⚡ Latencia: 0.59 ms por ejemplo
- 🚀 Throughput: 1,699 ejemplos/segundo
- 📦 No requiere PyTorch instalado (solo ONNX Runtime)

### ¿Cómo obtengo las probabilidades en lugar de solo la clase?

Las probabilidades se guardan automáticamente en:
```python
probs = np.load('inference_results/probabilities_pytorch.npy')
# Shape: (5, 3) - [P(DOWN), P(STATIONARY), P(UP)] para cada ejemplo
```

### ¿Qué son los "logits"?

Los logits son las salidas crudas del modelo antes de aplicar softmax:
```python
logits = [-0.163, 2.429, -2.331]  # Valores sin normalizar

# Después de softmax → probabilidades
probs = [0.0691, 0.9230, 0.0079]  # Suman 1.0
```

Logits más altos indican mayor confianza en esa clase.

---

## 🐛 Troubleshooting

### Error: "FileNotFoundError: data/BTC/train.npy"

**Solución:** Ejecuta primero el preprocesamiento de datos:
```bash
python3 main.py +model=tlob +dataset=btc \
    experiment.is_wandb=False \
    experiment.is_data_preprocessed=False
```

### Error: "No such file or directory: checkpoint.pt"

**Solución:** Verifica la ruta del checkpoint en el script:
```bash
ls -la data/checkpoints/TLOB/BTC_*/pt/*.pt
```

Y actualiza `CHECKPOINT_PATH` en `inference_pytorch.py`.

### Warning: "LibreSSL 2.8.3" o "NotOpenSSLWarning"

**No es crítico.** Es solo un warning de compatibilidad de urllib3. La inferencia funciona correctamente.

---

## 📞 Contacto y Referencias

- **Paper:** "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction"
- **Autores:** Leonardo Berti (Sapienza), Gjergji Kasneci (TUM)
- **Dataset:** Bitcoin LOB (Kaggle, 2023-01-09 to 2023-01-20)

---

**🎉 ¡Listo para hacer predicciones!**

Ejecuta `python3 inference_pytorch.py` y verás las predicciones en segundos.

