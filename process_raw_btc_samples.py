#!/usr/bin/env python3
"""
Script para Procesar Muestras Crudas de BTC desde el CSV Original
==================================================================
Toma muestras del archivo CSV original de Kaggle, aplica el mismo
preprocesamiento que usa el modelo TLOB, y genera archivos .npy
listos para inferencia.

Dataset Original:
- Fuente: Binance Bitcoin Perpetual (BTCUSDT.P)
- Período: 9-20 Enero 2023 (12 días)
- Frecuencia: 250ms
- Estructura: 42 columnas
  * Col 0: Index
  * Col 1: Timestamp (microsegundos UTC)
  * Col 2: Datetime
  * Col 3-22: BID data (10 niveles de precio + volumen)
  * Col 23-42: ASK data (10 niveles de precio + volumen)

Uso:
    python3 process_raw_btc_samples.py --num_samples 10 --seq_size 128
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def z_score_normalize(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """
    Aplica Z-score normalization a los datos del orderbook
    
    Args:
        data: DataFrame con columnas alternadas (precio, volumen, precio, volumen, ...)
        mean_size, std_size: Estadísticas para normalizar volúmenes
        mean_prices, std_prices: Estadísticas para normalizar precios
    
    Returns:
        data normalizado, estadísticas calculadas
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


def load_raw_csv(csv_path, nrows=None):
    """
    Carga el CSV original de Kaggle
    
    Estructura del CSV:
    - Col 0: Index (Unnamed: 0)
    - Col 1: Timestamp (microsegundos)
    - Col 2: Datetime
    - Col 3-12: BID Price L1-L10
    - Col 13-22: BID Volume L1-L10
    - Col 23-32: ASK Price L1-L10
    - Col 33-42: ASK Volume L1-L10
    """
    print(f"\n📂 Cargando CSV: {csv_path}")
    if nrows:
        print(f"   ⚠ Limitando a primeras {nrows:,} filas para testing")
        df = pd.read_csv(csv_path, index_col='Unnamed: 0', nrows=nrows)
    else:
        df = pd.read_csv(csv_path, index_col='Unnamed: 0')
    print(f"   ✓ Cargado: {len(df):,} filas × {len(df.columns)} columnas")
    return df


def reorder_columns(df):
    """
    Reordena las columnas para que sigan el formato esperado por el modelo:
    ASK Price, ASK Vol, BID Price, BID Vol (alternando por nivel)
    
    Formato Original del CSV:
    [timestamp, datetime, BID_P1-P10, BID_V1-V10, ASK_P1-P10, ASK_V1-V10]
    
    Formato Esperado por el Modelo:
    [ASK_P1, ASK_V1, BID_P1, BID_V1, ASK_P2, ASK_V2, BID_P2, BID_V2, ...]
    """
    print("\n🔄 Reordenando columnas al formato del modelo...")
    
    # Las columnas en el CSV son [0, 1, 2, 3, ..., 41]
    df.columns = np.arange(len(df.columns))
    
    # Seleccionar y reordenar:
    # Col 0: Timestamp
    # Col 1: Datetime
    # Col 2-11: BID Price L1-L10
    # Col 12-21: BID Volume L1-L10
    # Col 22-31: ASK Price L1-L10
    # Col 32-41: ASK Volume L1-L10
    
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
    print(f"   ✓ Formato: timestamp + 10 niveles × 4 (ASK_P, ASK_V, BID_P, BID_V)")
    
    return df_reordered


def extract_samples(df, num_samples, seq_size):
    """
    Extrae muestras aleatorias del DataFrame
    
    Args:
        df: DataFrame con datos ya reordenados
        num_samples: Número de muestras a extraer
        seq_size: Longitud de cada ventana (default 128)
    
    Returns:
        List de DataFrames, cada uno con seq_size filas consecutivas
    """
    print(f"\n📊 Extrayendo {num_samples} muestras aleatorias...")
    print(f"   Tamaño de ventana: {seq_size} timesteps")
    
    max_start_idx = len(df) - seq_size
    if max_start_idx < num_samples:
        print(f"   ⚠ Advertencia: Solo hay espacio para {max_start_idx} ventanas")
        num_samples = min(num_samples, max_start_idx)
    
    # Generar índices aleatorios sin repetición
    np.random.seed(42)  # Para reproducibilidad
    start_indices = np.random.choice(max_start_idx, size=num_samples, replace=False)
    start_indices = sorted(start_indices)
    
    samples = []
    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + seq_size
        sample = df.iloc[start_idx:end_idx].copy()
        samples.append(sample)
        
        # Info de la muestra
        timestamp_first = sample['timestamp'].iloc[0]
        timestamp_last = sample['timestamp'].iloc[-1]
        print(f"   ✓ Muestra {i+1}: filas {start_idx:,} - {end_idx:,} "
              f"(ts: {timestamp_first} → {timestamp_last})")
    
    return samples


def preprocess_sample(sample, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """
    Aplica el mismo preprocesamiento que el modelo:
    1. Elimina timestamp
    2. Aplica Z-score normalization
    
    Args:
        sample: DataFrame con una muestra (seq_size × 41)
        mean_size, mean_prices, std_size, std_prices: Estadísticas para normalización
    
    Returns:
        numpy array normalizado (seq_size × 40)
    """
    # Eliminar timestamp
    sample_no_ts = sample.drop(columns=['timestamp'])
    
    # Aplicar z-score
    normalized, mean_size, mean_prices, std_size, std_prices = z_score_normalize(
        sample_no_ts, mean_size, mean_prices, std_size, std_prices
    )
    
    return normalized.values, mean_size, mean_prices, std_size, std_prices


def main():
    parser = argparse.ArgumentParser(
        description="Procesar muestras crudas de BTC desde CSV original"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='data/BTC/original_source/1-09-1-20.csv',
        help='Ruta al CSV original'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Número de muestras a extraer (default: 10)'
    )
    parser.add_argument(
        '--seq_size',
        type=int,
        default=128,
        help='Tamaño de la ventana temporal (default: 128)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/BTC/raw_samples',
        help='Directorio donde guardar las muestras procesadas'
    )
    parser.add_argument(
        '--nrows',
        type=int,
        default=None,
        help='Limitar número de filas a cargar (para testing rápido)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PROCESAMIENTO DE MUESTRAS CRUDAS DE BTC")
    print("=" * 80)
    print(f"\n⚙️ Configuración:")
    print(f"   CSV Original: {args.csv_path}")
    print(f"   Número de muestras: {args.num_samples}")
    print(f"   Tamaño de ventana: {args.seq_size} timesteps")
    print(f"   Directorio de salida: {args.output_dir}")
    
    # 1. Cargar CSV
    df = load_raw_csv(args.csv_path, nrows=args.nrows)
    
    # 2. Reordenar columnas
    df_reordered = reorder_columns(df)
    
    # 3. Extraer muestras
    samples = extract_samples(df_reordered, args.num_samples, args.seq_size)
    
    # 4. Preprocesar y guardar
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Procesando y guardando muestras...")
    
    mean_size, mean_prices, std_size, std_prices = None, None, None, None
    processed_samples = []
    
    for i, sample in enumerate(samples):
        # Preprocesar (usa estadísticas de la primera muestra para todas)
        processed, mean_size, mean_prices, std_size, std_prices = preprocess_sample(
            sample, mean_size, mean_prices, std_size, std_prices
        )
        
        processed_samples.append(processed)
        
        # Guardar individualmente
        output_file = output_dir / f"raw_sample_{i+1}.npy"
        np.save(output_file, processed)
        
        print(f"   ✓ Sample {i+1}: shape {processed.shape} → {output_file}")
        print(f"      Stats: mean={processed.mean():.4f}, std={processed.std():.4f}, "
              f"min={processed.min():.4f}, max={processed.max():.4f}")
    
    # 5. Guardar todas las muestras en un solo archivo (para batch inference)
    all_samples = np.array(processed_samples)
    batch_file = output_dir / "raw_samples_batch.npy"
    np.save(batch_file, all_samples)
    
    print(f"\n   ✓ Batch file: shape {all_samples.shape} → {batch_file}")
    
    # 6. Guardar metadata
    metadata = {
        'num_samples': args.num_samples,
        'seq_size': args.seq_size,
        'csv_source': args.csv_path,
        'normalization_stats': {
            'mean_prices': float(mean_prices),
            'std_prices': float(std_prices),
            'mean_size': float(mean_size),
            'std_size': float(std_size),
        }
    }
    
    import json
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Metadata: {metadata_file}")
    
    # 7. Crear README
    readme_content = f"""# Muestras Crudas Procesadas de BTC

## Información General
- **Fuente**: {args.csv_path}
- **Dataset**: Binance Bitcoin Perpetual (BTCUSDT.P)
- **Período**: 9-20 Enero 2023
- **Frecuencia de muestreo**: 250ms

## Archivos Generados
- `raw_sample_N.npy`: Muestras individuales procesadas
- `raw_samples_batch.npy`: Todas las muestras en un solo archivo (shape: {all_samples.shape})
- `metadata.json`: Metadatos del procesamiento

## Formato de Datos
- **Shape por muestra**: ({args.seq_size}, 40)
- **Timesteps**: {args.seq_size}
- **Features**: 40 (10 niveles del LOB × 4: ASK Price, ASK Vol, BID Price, BID Vol)
- **Normalización**: Z-score aplicado

## Estructura de Features
| Index | Descripción |
|-------|-------------|
| 0, 2, 4, ..., 18  | ASK Price Levels 1-10 (sell orders) |
| 1, 3, 5, ..., 19  | ASK Volume Levels 1-10 |
| 20, 22, 24, ..., 38 | BID Price Levels 1-10 (buy orders) |
| 21, 23, 25, ..., 39 | BID Volume Levels 1-10 |

## Uso para Inferencia

### Muestra Individual
```python
import numpy as np

# Cargar una muestra
sample = np.load('raw_sample_1.npy')
print(f"Shape: {{sample.shape}}")  # (128, 40)

# Agregar dimensión de batch para el modelo
sample_batch = sample[np.newaxis, :, :]  # (1, 128, 40)
```

### Batch Completo
```python
# Cargar todas las muestras
samples = np.load('raw_samples_batch.npy')
print(f"Shape: {{samples.shape}}")  # ({args.num_samples}, 128, 40)
```

### Inferencia con PyTorch
```bash
python3 inference_single_file.py {output_dir}/raw_sample_1.npy
```

## Estadísticas de Normalización
- Mean Prices: {mean_prices:.6f}
- Std Prices: {std_prices:.6f}
- Mean Volumes: {mean_size:.6f}
- Std Volumes: {std_size:.6f}

---
Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_file = output_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"   ✓ README: {readme_file}")
    
    print("\n" + "=" * 80)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"\n📁 Archivos generados en: {output_dir}/")
    print(f"\n🚀 Para ejecutar inferencia:")
    print(f"   python3 inference_single_file.py {output_dir}/raw_sample_1.npy")


if __name__ == "__main__":
    main()
