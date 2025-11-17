# 🚀 Demo: Procesamiento e Inferencia con Datos Crudos de BTC

Este documento muestra cómo procesar datos **crudos** (raw) del CSV original de Kaggle y realizar inferencia con el modelo TLOB entrenado.

## 📊 Dataset Original

- **Fuente**: [Kaggle - Bitcoin Perpetual LOB Data](https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data)
- **Exchange**: Binance (BTCUSDT.P)
- **Período**: 9-20 Enero 2023 (12 días consecutivos)
- **Frecuencia**: 250ms (4 muestras por segundo)
- **Total de filas**: 3,730,870

### Estructura del CSV Original

| Columna | Descripción |
|---------|-------------|
| 0 | Index |
| 1 | Timestamp (microsegundos UTC) |
| 2 | Datetime |
| 3-12 | BID Price Levels 1-10 |
| 13-22 | BID Volume Levels 1-10 |
| 23-32 | ASK Price Levels 1-10 |
| 33-42 | ASK Volume Levels 1-10 |

**Total**: 43 columnas (1 index + 42 columnas de datos)

---

## 🔄 Pipeline Completo

### Paso 1: Procesar Muestras Crudas

El script `process_raw_btc_samples.py` realiza:

1. **Carga** el CSV original
2. **Reordena** las columnas al formato esperado por el modelo:
   - ASK Price, ASK Vol, BID Price, BID Vol (alternando por cada nivel)
3. **Extrae** N muestras aleatorias de 128 timesteps consecutivos
4. **Normaliza** con Z-score (usando estadísticas de la propia muestra)
5. **Guarda** archivos `.npy` listos para inferencia

```bash
# Procesar 10 muestras del CSV original
python3 process_raw_btc_samples.py --num_samples 10

# Opciones adicionales
python3 process_raw_btc_samples.py \
    --num_samples 20 \
    --seq_size 128 \
    --csv_path data/BTC/original_source/1-09-1-20.csv \
    --output_dir data/BTC/raw_samples
```

**Salida**:
```
data/BTC/raw_samples/
├── raw_sample_1.npy         # Muestra individual (128, 40)
├── raw_sample_2.npy
├── ...
├── raw_sample_10.npy
├── raw_samples_batch.npy    # Todas las muestras (10, 128, 40)
├── metadata.json            # Metadatos y estadísticas
└── README.md                # Documentación
```

---

### Paso 2: Inferencia Individual

Usa `inference_single_file.py` para predecir sobre una muestra:

```bash
python3 inference_single_file.py data/BTC/raw_samples/raw_sample_1.npy
```

**Ejemplo de salida**:

```
================================================================================
                           🎯 RESULTADO DE INFERENCIA                            
================================================================================

📂 Archivo: data/BTC/raw_samples/raw_sample_1.npy
📊 Shape: (128, 40) (timesteps × features)

📥 Estadísticas de Entrada:
   Mean:  -0.0000
   Std:    0.9998
   Min:   -1.0011
   Max:    1.0001

🎲 Probabilidades:
   📉 DOWN:         0.82%
   ➡️  STATIONARY:  96.12%
   📈 UP:           3.06%

********************************************************************************
                     🎯 PREDICCIÓN: ➡️ STATIONARY (clase 1)                      
                              💪 CONFIANZA:  96.12%                              
********************************************************************************
```

---

### Paso 3: Inferencia en Batch

Procesa todas las muestras de una vez:

```bash
# Opción 1: Archivo batch
python3 inference_pytorch.py \
    --examples_path data/BTC/raw_samples/raw_samples_batch.npy

# Opción 2: Iterar sobre individuales
for i in {1..10}; do
    python3 inference_single_file.py data/BTC/raw_samples/raw_sample_${i}.npy
done
```

---

## 🧪 Ejemplo Completo: Del CSV Raw a la Predicción

### 1. Verificar el CSV original

```bash
# Ver primeras líneas
head -5 data/BTC/original_source/1-09-1-20.csv

# Contar filas
wc -l data/BTC/original_source/1-09-1-20.csv
```

### 2. Procesar muestras

```bash
python3 process_raw_btc_samples.py --num_samples 10
```

### 3. Inspeccionar muestras generadas

```python
import numpy as np

# Cargar una muestra
sample = np.load('data/BTC/raw_samples/raw_sample_1.npy')
print(f"Shape: {sample.shape}")  # (128, 40)
print(f"Mean: {sample.mean():.4f}")
print(f"Std: {sample.std():.4f}")

# Ver primeras 5 filas, 10 columnas
print(sample[:5, :10])
```

### 4. Realizar inferencia

```bash
python3 inference_single_file.py data/BTC/raw_samples/raw_sample_1.npy
```

### 5. Ver resultados guardados

```bash
# Resultado numérico
cat data/BTC/raw_samples/raw_sample_1_result.txt

# Resultado como array
python3 -c "
import numpy as np
result = np.load('data/BTC/raw_samples/raw_sample_1_result.npy')
print(f'Logits: {result}')
"
```

---

## 📐 Detalles Técnicos del Preprocesamiento

### Reordenamiento de Columnas

**CSV Original**:
```
[Index, Timestamp, Datetime, BID_P1-P10, BID_V1-V10, ASK_P1-P10, ASK_V1-V10]
```

**Formato del Modelo** (alternando por nivel):
```
[Timestamp, ASK_P1, ASK_V1, BID_P1, BID_V1, ASK_P2, ASK_V2, BID_P2, BID_V2, ...]
```

### Normalización Z-Score

Para cada feature:
```python
normalized_value = (value - mean) / std
```

- **Precios** (columnas pares): Usan `mean_prices` y `std_prices`
- **Volúmenes** (columnas impares): Usan `mean_size` y `std_size`

### Ventana de Inferencia

- **Tamaño**: 128 timesteps consecutivos
- **Duración temporal**: ~32 segundos (128 × 250ms)
- **Features**: 40 (10 niveles del LOB × 4 tipos de datos)

---

## 🎯 Casos de Uso

### Caso 1: Testing con Nuevos Períodos

Si obtienes datos de otro período temporal (e.g., Febrero 2023):

```bash
# Procesar el nuevo CSV
python3 process_raw_btc_samples.py \
    --csv_path data/BTC/original_source/2-01-2-15.csv \
    --num_samples 20 \
    --output_dir data/BTC/raw_samples_feb
```

### Caso 2: Inferencia en Tiempo Real (Simulado)

Extrae ventanas consecutivas en lugar de aleatorias:

```python
# Modificar extract_samples() para ventanas consecutivas
start_indices = range(0, max_start_idx, seq_size)  # Sin overlapping
```

### Caso 3: Evaluar Diferentes Horizontes

El modelo actual predice a **horizon=10** (10 timesteps adelante = 2.5 segundos).

Para evaluar otros horizontes, necesitas:
1. Entrenar modelos con diferentes `h` (20, 50, 100)
2. Usar los checkpoints correspondientes en inferencia

---

## 📊 Comparación: Datos Preprocesados vs Raw

| Aspecto | `train.npy` (Preprocesado) | Raw CSV |
|---------|----------------------------|---------|
| **Fuente** | Ya procesado y guardado | CSV original de Kaggle |
| **Normalización** | Estadísticas del training set | Estadísticas propias |
| **Labels** | Incluye 4 columnas de labels | Solo LOB (40 features) |
| **Formato** | (N, 44) | (N, 43) → (ventana, 40) |
| **Uso** | Training y evaluación | Inferencia en nuevos datos |

**Ventaja de usar raw**: Puedes procesar **cualquier** período temporal nuevo sin depender de los datos preprocesados.

---

## 🔧 Troubleshooting

### Error: "Index out of bounds"

**Causa**: El CSV tiene menos filas que `num_samples × seq_size`.

**Solución**:
```bash
# Reducir número de muestras
python3 process_raw_btc_samples.py --num_samples 5
```

### Error: "Columns mismatch"

**Causa**: El CSV tiene formato diferente.

**Solución**: Verificar que el CSV tenga exactamente 43 columnas (1 index + 42 datos).

### Advertencia: "Normalization stats differ"

**Causa**: Las estadísticas de normalización de la muestra raw difieren del training set.

**Impacto**: Puede afectar ligeramente la precisión del modelo. Para máxima precisión, usa las estadísticas del training set.

---

## 📚 Referencias

- **Dataset Original**: https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data
- **Artículo TLOB**: Temporal Limit Order Book for Price Trend Prediction
- **Código de Preprocesamiento**: `preprocessing/btc.py`
- **Script de Procesamiento Raw**: `process_raw_btc_samples.py`
- **Script de Inferencia**: `inference_single_file.py`

---

## ✅ Checklist para Nuevos Datos

- [ ] Descargar CSV de Kaggle o exchange
- [ ] Verificar estructura (43 columnas)
- [ ] Ejecutar `process_raw_btc_samples.py`
- [ ] Verificar archivos `.npy` generados
- [ ] Probar inferencia con una muestra
- [ ] Analizar resultados y métricas

---

**Última actualización**: $(date)
**Generado automáticamente por el pipeline de procesamiento**

