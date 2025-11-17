# Ejemplos con Procesamiento Idéntico a btc.py

## ⚠️ IMPORTANTE
**Estos ejemplos usan el MISMO procesamiento que btc.py**

### Reordenamiento de Columnas
```
CSV Original:
- cols 0-1: index, timestamp
- cols 2: datetime  
- cols 3-12: BID prices (buy1-buy10)
- cols 13-22: BID volumes (vbuy1-vbuy10)
- cols 23-32: ASK prices (sell1-sell10)
- cols 33-42: ASK volumes (vsell1-vsell10)

Después de reordenar (como btc.py línea 77):
[timestamp, sell1, vsell1, buy1, vbuy1, sell2, vsell2, buy2, vbuy2, ...]
```

### Normalización
Usa `z_score_orderbook()` de `utils/utils_data.py`:
- **Precios** (columnas pares): `(x - mean_prices) / std_prices`
- **Volúmenes** (columnas impares): `(x - mean_size) / std_size`

## Información General
- **Fuente**: `data/BTC/original_source/1-09-1-20.csv`
- **Dataset**: Binance Bitcoin Perpetual (BTCUSDT.P)
- **Período**: 9-20 Enero 2023
- **Frecuencia**: 250ms

## Archivos Generados

### Por cada ejemplo (N = 1 a 7):
- `raw_example_N.csv` - CSV crudo con timestamp
- `raw_example_N.npy` - NPY crudo sin timestamp
- `normalized_example_N.npy` - NPY normalizado (listo para inferencia)

### Metadata
- `metadata.json` - Información completa
- `README.md` - Este archivo

## Formato de Datos

### CSV Crudo
- **Shape**: (128, 41)
- **Columnas**: timestamp + 40 features LOB
- **Valores**: Sin normalizar (precios BTC reales)

### NPY Crudo
- **Shape**: (128, 40)
- **Valores**: Sin normalizar
- **Orden**: sell1, vsell1, buy1, vbuy1, ...

### NPY Normalizado
- **Shape**: (128, 40)
- **Valores**: Z-score normalizados
- **Mean**: ≈ 0.0
- **Std**: ≈ 1.0

## Ejemplos Extraídos

| Ejemplo | Filas | Fecha | Mean (raw) | Mean (norm) |
|---------|-------|-------|------------|-------------|
| 1 | 0 - 128 | 2023-01-09 22:17:40 | 8593.41 | Ver archivo |
| 2 | 532,963 - 533,091 | 2023-01-11 11:23:38 | 8715.39 | Ver archivo |
| 3 | 1,065,926 - 1,066,054 | 2023-01-13 00:32:25 | 9405.97 | Ver archivo |
| 4 | 1,598,889 - 1,599,017 | 2023-01-14 13:39:26 | 10454.74 | Ver archivo |
| 5 | 2,131,852 - 2,131,980 | 2023-01-16 02:46:25 | 10614.61 | Ver archivo |
| 6 | 2,664,815 - 2,664,943 | 2023-01-17 15:54:23 | 10572.64 | Ver archivo |
| 7 | 3,197,778 - 3,197,906 | 2023-01-19 05:02:27 | 10410.38 | Ver archivo |


## Estadísticas de Normalización

Usando las MISMAS estadísticas para todos los ejemplos (como en btc.py):

- **Mean Prices**: 17182.652187
- **Std Prices**: 1.423941
- **Mean Volumes**: 4.171510
- **Std Volumes**: 9.590984

## Uso en Streamlit

Los archivos `normalized_example_N.npy` están listos para inferencia:

```python
import numpy as np

# Cargar ejemplo normalizado
data = np.load('normalized_example_1.npy')
print(f"Shape: {data.shape}")  # (128, 40)
print(f"Mean: {data.mean():.6f}")  # ≈ 0.0
print(f"Std: {data.std():.6f}")    # ≈ 1.0
```

## Inferencia

```bash
# Con ejemplo normalizado
python3 inference_single_file.py data/BTC/raw_examples/normalized_example_1.npy
```

## Validación

Para verificar que el procesamiento es correcto:

```python
# Comparar con train.npy del dataset procesado
train = np.load('data/BTC/train.npy')
example = np.load('data/BTC/raw_examples/normalized_example_1.npy')

print(f"Train stats: mean={train[:, :40].mean():.6f}, std={train[:, :40].std():.6f}")
print(f"Example stats: mean={example.mean():.6f}, std={example.std():.6f}")
# Deberían ser similares
```

---
Generado: 2025-11-16 22:15:59
