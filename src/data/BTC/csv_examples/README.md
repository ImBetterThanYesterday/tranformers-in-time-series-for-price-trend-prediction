# Ejemplos RAW desde CSV Original

## Información General
- **Fuente**: `data/BTC/original_source/1-09-1-20.csv`
- **Dataset**: Binance Bitcoin Perpetual (BTCUSDT.P)
- **Período**: 9-20 Enero 2023
- **Frecuencia**: 250ms
- **Formato**: RAW (sin procesar)

## Archivos Generados
- `csv_example_N.csv`: Ejemplos individuales en CSV (N = 1 a 7)
- `csv_example_N.npy`: Ejemplos individuales en NPY (N = 1 a 7)
- `csv_examples_batch.npy`: Todos los ejemplos en un batch
- `metadata.json`: Metadatos completos
- `README.md`: Este archivo

## Formato de Datos
- **Shape por ejemplo**: (128, 42)
- **Columnas**: 42 (formato original del CSV)
- **Normalización**: NINGUNA (datos RAW)
- **Reordenamiento**: NINGUNO (orden original del CSV)

## ⚠️ IMPORTANTE
Los datos están en formato RAW y requieren:
1. Reordenamiento de columnas al formato del modelo
2. Normalización Z-score
3. Eliminación del timestamp

**Streamlit se encarga automáticamente de este procesamiento al cargar.**

## Ejemplos Extraídos

| Ejemplo | Filas | Fecha | Timestamp Inicial |
|---------|-------|-------|-------------------|
| 1 | 0 - 128 | 1970-01-20 08:48:22 | 1673302660926 |
| 2 | 532,963 - 533,091 | 1970-01-20 08:50:36 | 1673436218075 |
| 3 | 1,065,926 - 1,066,054 | 1970-01-20 08:52:49 | 1673569945917 |
| 4 | 1,598,889 - 1,599,017 | 1970-01-20 08:55:03 | 1673703566197 |
| 5 | 2,131,852 - 2,131,980 | 1970-01-20 08:57:17 | 1673837185335 |
| 6 | 2,664,815 - 2,664,943 | 1970-01-20 08:59:30 | 1673970863254 |
| 7 | 3,197,778 - 3,197,906 | 1970-01-20 09:01:44 | 1674104547342 |


## Uso

### Cargar Ejemplo Individual CSV
```python
import pandas as pd

# Cargar un ejemplo CSV
example = pd.read_csv('csv_example_1.csv')
print(f"Shape: {example.shape}")  # (128, 42)
```

### Cargar Ejemplo Individual NPY
```python
import numpy as np

# Cargar un ejemplo NPY
example = np.load('csv_example_1.npy')
print(f"Shape: {example.shape}")  # (128, 42)
```

### En Streamlit
Los ejemplos aparecen automáticamente en el dropdown de "Ejemplos desde CSV".
Streamlit se encarga del procesamiento (reordenamiento + normalización).

---
Generado: 2025-11-16 22:15:10
