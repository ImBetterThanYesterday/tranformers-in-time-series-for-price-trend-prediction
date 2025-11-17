#!/usr/bin/env python3
"""
Script para Crear Ejemplos desde el CSV Original Local
======================================================
Toma 7 muestras directamente del CSV en data/BTC/original_source/
y las prepara para inferencia sin necesidad de preprocesar todo el dataset.

Uso:
    python3 create_examples_from_csv.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

print("=" * 80)
print("CREAR 7 EJEMPLOS DESDE CSV ORIGINAL LOCAL")
print("=" * 80)

# Configuración
CSV_PATH = "data/BTC/original_source/1-09-1-20.csv"
OUTPUT_DIR = Path("data/BTC/csv_examples")
NUM_EXAMPLES = 7
SEQ_SIZE = 128

print(f"\n⚙️ Configuración:")
print(f"   CSV Original: {CSV_PATH}")
print(f"   Número de ejemplos: {NUM_EXAMPLES}")
print(f"   Tamaño de ventana: {SEQ_SIZE} timesteps")
print(f"   Directorio de salida: {OUTPUT_DIR}")

# ============================================================================
# 1. FUNCIÓN DE NORMALIZACIÓN Z-SCORE
# ============================================================================
def z_score_normalize(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """
    Aplica Z-score normalization a los datos del orderbook
    """
    # Calcular estadísticas si no se proporcionan
    if mean_size is None or std_size is None:
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    if mean_prices is None or std_prices is None:
        mean_prices = data.iloc[:, 0::2].stack().mean()
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

    return data, mean_size, mean_prices, std_size, std_prices


# ============================================================================
# 2. CARGAR CSV
# ============================================================================
print(f"\n📂 Cargando CSV: {CSV_PATH}")
print("   (Este proceso puede tardar ~30 segundos con 3.7M filas...)")

df = pd.read_csv(CSV_PATH, index_col='Unnamed: 0')
print(f"   ✓ Cargado: {len(df):,} filas × {len(df.columns)} columnas")

# ============================================================================
# 3. REORDENAR COLUMNAS AL FORMATO DEL MODELO
# ============================================================================
print("\n🔄 Reordenando columnas al formato del modelo...")

# Renombrar columnas a números
df.columns = np.arange(len(df.columns))

# Reordenar: [Timestamp, ASK_P1, ASK_V1, BID_P1, BID_V1, ASK_P2, ASK_V2, ...]
# CSV Original: [0=Timestamp, 1=Datetime, 2-11=BID_P, 12-21=BID_V, 22-31=ASK_P, 32-41=ASK_V]

new_order = [0]  # Timestamp primero

for level in range(10):
    ask_price_col = 22 + level
    ask_vol_col = 32 + level
    bid_price_col = 2 + level
    bid_vol_col = 12 + level
    
    new_order.extend([ask_price_col, ask_vol_col, bid_price_col, bid_vol_col])

df_reordered = df.iloc[:, new_order]

# Renombrar columnas
column_names = ["timestamp"]
for i in range(1, 11):
    column_names.extend([
        f"sell{i}", f"vsell{i}", f"buy{i}", f"vbuy{i}"
    ])

df_reordered.columns = column_names

print(f"   ✓ Columnas reordenadas: {len(df_reordered.columns)} columnas")
print(f"   ✓ Formato: [timestamp] + 10 niveles × 4 (ASK_P, ASK_V, BID_P, BID_V)")

# ============================================================================
# 4. EXTRAER 7 EJEMPLOS DISTRIBUIDOS
# ============================================================================
print(f"\n📊 Extrayendo {NUM_EXAMPLES} ejemplos distribuidos uniformemente...")

max_start_idx = len(df_reordered) - SEQ_SIZE

# Distribuir los ejemplos uniformemente a lo largo del dataset
# para capturar diferentes momentos del mercado
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
    
    # Convertir timestamp a datetime para mostrar fecha legible
    try:
        date_first = pd.to_datetime(timestamp_first, unit='us')
        date_last = pd.to_datetime(timestamp_last, unit='us')
        date_str = date_first.strftime('%Y-%m-%d %H:%M:%S')
    except:
        date_str = "N/A"
    
    info = {
        'example_id': i + 1,
        'start_row': start_idx,
        'end_row': end_idx,
        'timestamp_first': int(timestamp_first),
        'timestamp_last': int(timestamp_last),
        'date_first': date_str
    }
    sample_info.append(info)
    
    print(f"   ✓ Ejemplo {i+1}: filas {start_idx:,} - {end_idx:,}")
    print(f"      Fecha: {date_str}")
    print(f"      Timestamp: {timestamp_first} → {timestamp_last}")

# ============================================================================
# 5. NORMALIZAR Y GUARDAR
# ============================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n💾 Normalizando y guardando ejemplos...")

mean_size, mean_prices, std_size, std_prices = None, None, None, None
processed_samples = []

for i, sample in enumerate(samples):
    # Eliminar timestamp
    sample_no_ts = sample.drop(columns=['timestamp'])
    
    # Normalizar (usa estadísticas de la primera muestra para todas)
    normalized, mean_size, mean_prices, std_size, std_prices = z_score_normalize(
        sample_no_ts, mean_size, mean_prices, std_size, std_prices
    )
    
    # Convertir a numpy array
    processed = normalized.values
    processed_samples.append(processed)
    
    # Guardar individualmente
    output_file = OUTPUT_DIR / f"csv_example_{i+1}.npy"
    np.save(output_file, processed)
    
    print(f"   ✓ Ejemplo {i+1}: shape {processed.shape} → {output_file}")
    print(f"      Stats: mean={processed.mean():.4f}, std={processed.std():.4f}, "
          f"min={processed.min():.4f}, max={processed.max():.4f}")

# ============================================================================
# 6. GUARDAR BATCH Y METADATA
# ============================================================================

# Batch file
all_samples = np.array(processed_samples)
batch_file = OUTPUT_DIR / "csv_examples_batch.npy"
np.save(batch_file, all_samples)
print(f"\n   ✓ Batch file: shape {all_samples.shape} → {batch_file}")

# Metadata
metadata = {
    'num_examples': NUM_EXAMPLES,
    'seq_size': SEQ_SIZE,
    'csv_source': CSV_PATH,
    'total_csv_rows': len(df),
    'normalization_stats': {
        'mean_prices': float(mean_prices),
        'std_prices': float(std_prices),
        'mean_size': float(mean_size),
        'std_size': float(std_size),
    },
    'examples': sample_info
}

metadata_file = OUTPUT_DIR / "metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✓ Metadata: {metadata_file}")

# ============================================================================
# 7. CREAR README
# ============================================================================

readme_content = f"""# Ejemplos Creados desde CSV Original

## Información General
- **Fuente**: `{CSV_PATH}`
- **Dataset**: Binance Bitcoin Perpetual (BTCUSDT.P)
- **Período**: 9-20 Enero 2023
- **Frecuencia**: 250ms

## Archivos Generados
- `csv_example_N.npy`: Ejemplos individuales (N = 1 a {NUM_EXAMPLES})
- `csv_examples_batch.npy`: Todos los ejemplos en un batch
- `metadata.json`: Metadatos completos
- `README.md`: Este archivo

## Formato de Datos
- **Shape por ejemplo**: ({SEQ_SIZE}, 40)
- **Features**: 40 (10 niveles del LOB × 4: ASK Price, ASK Vol, BID Price, BID Vol)
- **Normalización**: Z-score

## Ejemplos Extraídos

| Ejemplo | Filas | Fecha | Timestamp Inicial |
|---------|-------|-------|-------------------|
"""

for info in sample_info:
    readme_content += f"| {info['example_id']} | {info['start_row']:,} - {info['end_row']:,} | {info['date_first']} | {info['timestamp_first']} |\n"

readme_content += f"""

## Estadísticas de Normalización
- **Mean Prices**: {mean_prices:.6f}
- **Std Prices**: {std_prices:.6f}
- **Mean Volumes**: {mean_size:.6f}
- **Std Volumes**: {std_size:.6f}

## Uso

### Cargar Ejemplo Individual
```python
import numpy as np

# Cargar un ejemplo
example = np.load('csv_example_1.npy')
print(f"Shape: {{example.shape}}")  # (128, 40)
```

### Inferencia
```bash
# Ejemplo individual
python3 inference_single_file.py data/BTC/csv_examples/csv_example_1.npy

# Todos los ejemplos
python3 inference_pytorch.py --examples_path data/BTC/csv_examples/csv_examples_batch.npy
```

### En Streamlit
Los ejemplos aparecen automáticamente en el dropdown de "Ejemplos desde CSV".

---
Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

readme_file = OUTPUT_DIR / "README.md"
with open(readme_file, 'w') as f:
    f.write(readme_content)

print(f"   ✓ README: {readme_file}")

# ============================================================================
# 8. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("✅ EJEMPLOS CREADOS EXITOSAMENTE")
print("=" * 80)

print(f"\n📁 Archivos generados en: {OUTPUT_DIR}/")
print(f"\n📊 Resumen:")
print(f"   • {NUM_EXAMPLES} ejemplos individuales")
print(f"   • 1 archivo batch con todos los ejemplos")
print(f"   • Shape de cada ejemplo: ({SEQ_SIZE}, 40)")
print(f"   • Normalizados con Z-score")

print(f"\n🎯 Distribución de ejemplos:")
for info in sample_info:
    print(f"   • Ejemplo {info['example_id']}: Fila {info['start_row']:,} ({info['date_first']})")

print(f"\n🚀 Siguiente paso:")
print(f"   1. Probar inferencia:")
print(f"      python3 inference_single_file.py {OUTPUT_DIR}/csv_example_1.npy")
print(f"\n   2. Ver en Streamlit:")
print(f"      streamlit run app.py")
print(f"      (Los ejemplos aparecerán en el dropdown)")

print("\n" + "=" * 80)

