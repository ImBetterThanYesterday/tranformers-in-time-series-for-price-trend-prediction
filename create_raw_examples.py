#!/usr/bin/env python3
"""
Script para Crear Ejemplos CRUDOS (sin normalizar)
==================================================
Usa el MISMO reordenamiento y normalización que btc.py

Uso:
    python3 create_raw_examples.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ============================================================================
# FUNCIÓN z_score_orderbook (COPIADA DE utils/utils_data.py)
# ============================================================================
def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """ 
    Z-score normalization para orderbook
    Normaliza precios y volúmenes por separado
    """
    if (mean_size is None) or (std_size is None):
        mean_size = data.iloc[:, 1::2].stack().mean()  # Columnas impares = volúmenes
        std_size = data.iloc[:, 1::2].stack().std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data.iloc[:, 0::2].stack().mean()  # Columnas pares = precios
        std_prices = data.iloc[:, 0::2].stack().std()

    # Aplicar z-score
    price_cols = data.columns[0::2]
    size_cols = data.columns[1::2]

    for col in size_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_size) / std_size

    for col in price_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_prices) / std_prices

    # Check for null values
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    return data, mean_size, mean_prices, std_size, std_prices

print("=" * 80)
print("CREAR EJEMPLOS CRUDOS (MISMO PROCESAMIENTO QUE BTC.PY)")
print("=" * 80)

# Configuración
CSV_PATH = "data/BTC/original_source/1-09-1-20.csv"
OUTPUT_DIR = Path("data/BTC/raw_examples")
NUM_EXAMPLES = 7
SEQ_SIZE = 128

print(f"\n⚙️ Configuración:")
print(f"   CSV Original: {CSV_PATH}")
print(f"   Número de ejemplos: {NUM_EXAMPLES}")
print(f"   Tamaño de ventana: {SEQ_SIZE} timesteps")
print(f"   Directorio de salida: {OUTPUT_DIR}")
print(f"   ⚠️  Usa el MISMO procesamiento que btc.py")

# ============================================================================
# 1. CARGAR Y REORDENAR CSV (IDÉNTICO A BTC.PY LÍNEA 72-90)
# ============================================================================
print(f"\n📂 Cargando CSV: {CSV_PATH}")
print("   (Este proceso puede tardar ~30 segundos con 3.7M filas...)")

df = pd.read_csv(CSV_PATH, index_col='Unnamed: 0')
print(f"   ✓ Cargado: {len(df):,} filas × {len(df.columns)} columnas")

print("\n🔄 Reordenando columnas (MISMO ORDEN QUE BTC.PY)...")

# Renombrar columnas a números (btc.py línea 73)
df.columns = np.arange(42)

# IMPORTANTE: Usar el MISMO reordenamiento que btc.py línea 77
# [1, 22,23, 2,3, 24,25, 4,5, 26,27, ...]
# = [timestamp, sell1, vsell1, buy1, vbuy1, sell2, vsell2, buy2, vbuy2, ...]
df_reordered = df.loc[:,[
    1,   # timestamp
    22, 23,  # sell1, vsell1 (ASK price level 1, ASK volume level 1)
    2, 3,    # buy1, vbuy1   (BID price level 1, BID volume level 1)
    24, 25,  # sell2, vsell2
    4, 5,    # buy2, vbuy2
    26, 27,  # sell3, vsell3
    6, 7,    # buy3, vbuy3
    28, 29,  # sell4, vsell4
    8, 9,    # buy4, vbuy4
    30, 31,  # sell5, vsell5
    10, 11,  # buy5, vbuy5
    32, 33,  # sell6, vsell6
    12, 13,  # buy6, vbuy6
    34, 35,  # sell7, vsell7
    14, 15,  # buy7, vbuy7
    36, 37,  # sell8, vsell8
    16, 17,  # buy8, vbuy8
    38, 39,  # sell9, vsell9
    18, 19,  # buy9, vbuy9
    40, 41,  # sell10, vsell10
    20, 21,  # buy10, vbuy10
]]

# Renombrar columnas (btc.py línea 79-90)
df_reordered.columns = [
    "timestamp",
    "sell1", "vsell1", "buy1", "vbuy1",
    "sell2", "vsell2", "buy2", "vbuy2",
    "sell3", "vsell3", "buy3", "vbuy3",
    "sell4", "vsell4", "buy4", "vbuy4",
    "sell5", "vsell5", "buy5", "vbuy5",
    "sell6", "vsell6", "buy6", "vbuy6",
    "sell7", "vsell7", "buy7", "vbuy7",
    "sell8", "vsell8", "buy8", "vbuy8",
    "sell9", "vsell9", "buy9", "vbuy9",
    "sell10", "vsell10", "buy10", "vbuy10",
]

print(f"   ✓ Columnas reordenadas: {len(df_reordered.columns)} columnas")
print(f"   ✓ Orden: [timestamp] + sell1,vsell1,buy1,vbuy1,sell2,...")

# ============================================================================
# 2. EXTRAER EJEMPLOS DISTRIBUIDOS
# ============================================================================
print(f"\n📊 Extrayendo {NUM_EXAMPLES} ejemplos distribuidos uniformemente...")

max_start_idx = len(df_reordered) - SEQ_SIZE
step = max_start_idx // NUM_EXAMPLES
start_indices = [i * step for i in range(NUM_EXAMPLES)]

samples = []
sample_info = []

for i, start_idx in enumerate(start_indices):
    end_idx = start_idx + SEQ_SIZE
    sample = df_reordered.iloc[start_idx:end_idx].copy()
    samples.append(sample)
    
    # Info de la muestra
    timestamp_first = sample['timestamp'].iloc[0]
    timestamp_last = sample['timestamp'].iloc[-1]
    
    # Convertir a string si es datetime
    if isinstance(timestamp_first, (pd.Timestamp, str)):
        date_str = str(timestamp_first)
    else:
        try:
            date_first = pd.to_datetime(timestamp_first, unit='us')
            date_str = date_first.strftime('%Y-%m-%d %H:%M:%S')
        except:
            date_str = "N/A"
    
    # Estadísticas sin normalizar
    sample_no_ts = sample.drop(columns=['timestamp'])
    mean_val = sample_no_ts.values.mean()
    std_val = sample_no_ts.values.std()
    
    info = {
        'example_id': i + 1,
        'start_row': start_idx,
        'end_row': end_idx,
        'timestamp_first': str(timestamp_first),
        'timestamp_last': str(timestamp_last),
        'date_first': date_str,
        'raw_stats': {
            'mean': float(mean_val),
            'std': float(std_val),
        }
    }
    sample_info.append(info)
    
    print(f"   ✓ Ejemplo {i+1}: filas {start_idx:,} - {end_idx:,}")
    print(f"      Fecha: {date_str}")
    print(f"      Stats RAW: mean={mean_val:.2f}, std={std_val:.2f}")

# ============================================================================
# 3. GUARDAR EJEMPLOS CRUDOS Y NORMALIZADOS
# ============================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n💾 Guardando ejemplos...")

# Variables para guardar estadísticas de normalización
norm_stats = None

for i, sample in enumerate(samples):
    # ========== CSV CRUDO (con timestamp) ==========
    csv_file = OUTPUT_DIR / f"raw_example_{i+1}.csv"
    sample.to_csv(csv_file, index=False)
    print(f"\n   ✓ Ejemplo {i+1} CSV CRUDO: {csv_file}")
    print(f"      Shape: ({len(sample)}, {len(sample.columns)})")
    
    # ========== NPY CRUDO (sin timestamp) ==========
    sample_no_ts = sample.drop(columns=['timestamp']).copy()
    npy_data = sample_no_ts.values
    
    npy_file = OUTPUT_DIR / f"raw_example_{i+1}.npy"
    np.save(npy_file, npy_data)
    print(f"   ✓ Ejemplo {i+1} NPY CRUDO: {npy_file}")
    print(f"      Shape: {npy_data.shape}")
    print(f"      Stats RAW: mean={npy_data.mean():.2f}, std={npy_data.std():.2f}")
    
    # ========== NPY NORMALIZADO (USANDO z_score_orderbook) ==========
    # Crear DataFrame para normalizar
    sample_df = sample_no_ts.copy()
    
    # Aplicar z_score_orderbook (MISMO QUE BTC.PY)
    # Primera muestra: calcular estadísticas
    # Resto: usar las mismas estadísticas
    if i == 0:
        sample_norm, mean_size, mean_prices, std_size, std_prices = z_score_orderbook(sample_df)
        norm_stats = {
            'mean_prices': float(mean_prices),
            'std_prices': float(std_prices),
            'mean_size': float(mean_size),
            'std_size': float(std_size)
        }
    else:
        sample_norm, _, _, _, _ = z_score_orderbook(
            sample_df, mean_size, mean_prices, std_size, std_prices
        )
    
    npy_norm = sample_norm.values
    
    norm_file = OUTPUT_DIR / f"normalized_example_{i+1}.npy"
    np.save(norm_file, npy_norm)
    print(f"   ✓ Ejemplo {i+1} NPY NORMALIZADO: {norm_file}")
    print(f"      Shape: {npy_norm.shape}")
    print(f"      Stats NORM: mean={npy_norm.mean():.6f}, std={npy_norm.std():.6f}")

# ============================================================================
# 4. GUARDAR METADATA
# ============================================================================

metadata = {
    'num_examples': NUM_EXAMPLES,
    'seq_size': SEQ_SIZE,
    'csv_source': CSV_PATH,
    'total_csv_rows': len(df),
    'processing': {
        'description': 'Usa el MISMO reordenamiento y normalización que btc.py',
        'reordering': '[timestamp, sell1, vsell1, buy1, vbuy1, ...]',
        'normalization': 'z_score_orderbook() - precios y volúmenes por separado',
    },
    'normalization_stats': norm_stats,
    'examples': sample_info
}

metadata_file = OUTPUT_DIR / "metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n   ✓ Metadata: {metadata_file}")

# ============================================================================
# 5. CREAR README
# ============================================================================

readme_content = f"""# Ejemplos con Procesamiento Idéntico a btc.py

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
- **Fuente**: `{CSV_PATH}`
- **Dataset**: Binance Bitcoin Perpetual (BTCUSDT.P)
- **Período**: 9-20 Enero 2023
- **Frecuencia**: 250ms

## Archivos Generados

### Por cada ejemplo (N = 1 a {NUM_EXAMPLES}):
- `raw_example_N.csv` - CSV crudo con timestamp
- `raw_example_N.npy` - NPY crudo sin timestamp
- `normalized_example_N.npy` - NPY normalizado (listo para inferencia)

### Metadata
- `metadata.json` - Información completa
- `README.md` - Este archivo

## Formato de Datos

### CSV Crudo
- **Shape**: ({SEQ_SIZE}, 41)
- **Columnas**: timestamp + 40 features LOB
- **Valores**: Sin normalizar (precios BTC reales)

### NPY Crudo
- **Shape**: ({SEQ_SIZE}, 40)
- **Valores**: Sin normalizar
- **Orden**: sell1, vsell1, buy1, vbuy1, ...

### NPY Normalizado
- **Shape**: ({SEQ_SIZE}, 40)
- **Valores**: Z-score normalizados
- **Mean**: ≈ 0.0
- **Std**: ≈ 1.0

## Ejemplos Extraídos

| Ejemplo | Filas | Fecha | Mean (raw) | Mean (norm) |
|---------|-------|-------|------------|-------------|
"""

for info in sample_info:
    readme_content += f"| {info['example_id']} | {info['start_row']:,} - {info['end_row']:,} | "
    readme_content += f"{info['date_first']} | {info['raw_stats']['mean']:.2f} | "
    readme_content += "Ver archivo |\n"

readme_content += f"""

## Estadísticas de Normalización

Usando las MISMAS estadísticas para todos los ejemplos (como en btc.py):

- **Mean Prices**: {norm_stats['mean_prices']:.6f}
- **Std Prices**: {norm_stats['std_prices']:.6f}
- **Mean Volumes**: {norm_stats['mean_size']:.6f}
- **Std Volumes**: {norm_stats['std_size']:.6f}

## Uso en Streamlit

Los archivos `normalized_example_N.npy` están listos para inferencia:

```python
import numpy as np

# Cargar ejemplo normalizado
data = np.load('normalized_example_1.npy')
print(f"Shape: {{data.shape}}")  # (128, 40)
print(f"Mean: {{data.mean():.6f}}")  # ≈ 0.0
print(f"Std: {{data.std():.6f}}")    # ≈ 1.0
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

print(f"Train stats: mean={{train[:, :40].mean():.6f}}, std={{train[:, :40].std():.6f}}")
print(f"Example stats: mean={{example.mean():.6f}}, std={{example.std():.6f}}")
# Deberían ser similares
```

---
Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

readme_file = OUTPUT_DIR / "README.md"
with open(readme_file, 'w') as f:
    f.write(readme_content)

print(f"   ✓ README: {readme_file}")

# ============================================================================
# 6. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("✅ EJEMPLOS CREADOS EXITOSAMENTE")
print("=" * 80)

print(f"\n📁 Archivos generados en: {OUTPUT_DIR}/")

print(f"\n📊 Resumen:")
print(f"   • {NUM_EXAMPLES} ejemplos CSV crudos")
print(f"   • {NUM_EXAMPLES} ejemplos NPY crudos")
print(f"   • {NUM_EXAMPLES} ejemplos NPY normalizados")
print(f"   • Total: {NUM_EXAMPLES * 3} archivos de datos")

print(f"\n✅ Ventajas:")
print(f"   • CSV crudo: Ver valores reales de BTC")
print(f"   • NPY crudo: Sin normalizar, formato numpy")
print(f"   • NPY normalizado: LISTO PARA INFERENCIA (mismo procesamiento que btc.py)")

print(f"\n📊 Estadísticas de Normalización:")
print(f"   • Mean Prices: {norm_stats['mean_prices']:.6f}")
print(f"   • Std Prices: {norm_stats['std_prices']:.6f}")
print(f"   • Mean Volumes: {norm_stats['mean_size']:.6f}")
print(f"   • Std Volumes: {norm_stats['std_size']:.6f}")

print(f"\n🎯 Uso recomendado:")
print(f"   1. Para ver datos reales: raw_example_N.csv")
print(f"   2. Para inferencia: normalized_example_N.npy")
print(f"   3. Para procesamiento custom: raw_example_N.npy")

print(f"\n🚀 Siguiente paso:")
print(f"   python3 inference_single_file.py {OUTPUT_DIR}/normalized_example_1.npy")

print("\n" + "=" * 80)
