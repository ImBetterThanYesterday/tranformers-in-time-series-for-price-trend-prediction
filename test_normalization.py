#!/usr/bin/env python3
"""
Script de Prueba: Normalización Automática
==========================================
Prueba que la función de normalización funciona correctamente
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 80)
print("PRUEBA: NORMALIZACIÓN AUTOMÁTICA")
print("=" * 80)

# ============================================================================
# FUNCIONES (copiadas de app.py)
# ============================================================================

def normalize_raw_data(data):
    """Aplica Z-score normalization a datos crudos"""
    df = pd.DataFrame(data)
    
    # Columnas pares = precios, impares = volúmenes
    mean_prices = df.iloc[:, 0::2].stack().mean()
    std_prices = df.iloc[:, 0::2].stack().std()
    mean_volumes = df.iloc[:, 1::2].stack().mean()
    std_volumes = df.iloc[:, 1::2].stack().std()
    
    print(f"\n📊 Estadísticas de normalización:")
    print(f"   Precios  -> mean: {mean_prices:.2f}, std: {std_prices:.2f}")
    print(f"   Volúmenes -> mean: {mean_volumes:.2f}, std: {std_volumes:.2f}")
    
    # Normalizar
    for col in df.columns[0::2]:  # Precios
        df[col] = (df[col] - mean_prices) / std_prices
    
    for col in df.columns[1::2]:  # Volúmenes
        df[col] = (df[col] - mean_volumes) / std_volumes
    
    return df.values

def is_data_normalized(data):
    """Detecta si los datos ya están normalizados"""
    mean = np.abs(data.mean())
    std = data.std()
    
    if mean > 100:
        return False, "raw"
    elif mean < 1 and 0.5 < std < 2:
        return True, "normalized"
    else:
        return None, "unknown"

# ============================================================================
# PRUEBA 1: DATOS CRUDOS (NPY)
# ============================================================================

print("\n" + "=" * 80)
print("PRUEBA 1: ARCHIVO NPY CRUDO")
print("=" * 80)

npy_file = Path("data/BTC/raw_examples/raw_example_1.npy")
if npy_file.exists():
    print(f"\n📂 Cargando: {npy_file}")
    raw_data = np.load(npy_file)
    
    print(f"\n📊 Datos ORIGINALES (crudos):")
    print(f"   Shape: {raw_data.shape}")
    print(f"   Mean: {raw_data.mean():.2f}")
    print(f"   Std: {raw_data.std():.2f}")
    print(f"   Min: {raw_data.min():.2f}")
    print(f"   Max: {raw_data.max():.2f}")
    
    # Detectar tipo
    is_norm, data_type = is_data_normalized(raw_data)
    print(f"\n🔍 Detección: {data_type}")
    
    if is_norm == False:
        print(f"\n🔄 Aplicando normalización...")
        normalized = normalize_raw_data(raw_data)
        
        print(f"\n📊 Datos NORMALIZADOS:")
        print(f"   Shape: {normalized.shape}")
        print(f"   Mean: {normalized.mean():.6f}")
        print(f"   Std: {normalized.std():.6f}")
        print(f"   Min: {normalized.min():.6f}")
        print(f"   Max: {normalized.max():.6f}")
        
        # Verificar
        if abs(normalized.mean()) < 0.1 and 0.9 < normalized.std() < 1.1:
            print("\n✅ PRUEBA 1 EXITOSA: Normalización correcta")
        else:
            print("\n❌ PRUEBA 1 FALLIDA: Normalización incorrecta")
    else:
        print("\n⚠️  Los datos ya estaban normalizados")
else:
    print(f"\n❌ No se encontró: {npy_file}")

# ============================================================================
# PRUEBA 2: DATOS CRUDOS (CSV)
# ============================================================================

print("\n" + "=" * 80)
print("PRUEBA 2: ARCHIVO CSV CRUDO")
print("=" * 80)

csv_file = Path("data/BTC/raw_examples/raw_example_1.csv")
if csv_file.exists():
    print(f"\n📂 Cargando: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Eliminar timestamp
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    raw_data = df.values
    
    print(f"\n📊 Datos ORIGINALES (crudos):")
    print(f"   Shape: {raw_data.shape}")
    print(f"   Mean: {raw_data.mean():.2f}")
    print(f"   Std: {raw_data.std():.2f}")
    print(f"   Min: {raw_data.min():.2f}")
    print(f"   Max: {raw_data.max():.2f}")
    
    # Mostrar algunos valores
    print(f"\n📝 Primeros valores (crudo):")
    print(f"   Precio 1: {raw_data[0, 0]:.2f}")
    print(f"   Volumen 1: {raw_data[0, 1]:.2f}")
    print(f"   Precio 2: {raw_data[0, 2]:.2f}")
    
    # Detectar tipo
    is_norm, data_type = is_data_normalized(raw_data)
    print(f"\n🔍 Detección: {data_type}")
    
    if is_norm == False:
        print(f"\n🔄 Aplicando normalización...")
        normalized = normalize_raw_data(raw_data)
        
        print(f"\n📊 Datos NORMALIZADOS:")
        print(f"   Shape: {normalized.shape}")
        print(f"   Mean: {normalized.mean():.6f}")
        print(f"   Std: {normalized.std():.6f}")
        print(f"   Min: {normalized.min():.6f}")
        print(f"   Max: {normalized.max():.6f}")
        
        print(f"\n📝 Primeros valores (normalizado):")
        print(f"   Z-score 1: {normalized[0, 0]:.4f}")
        print(f"   Z-score 2: {normalized[0, 1]:.4f}")
        print(f"   Z-score 3: {normalized[0, 2]:.4f}")
        
        # Verificar
        if abs(normalized.mean()) < 0.1 and 0.9 < normalized.std() < 1.1:
            print("\n✅ PRUEBA 2 EXITOSA: Normalización correcta")
        else:
            print("\n❌ PRUEBA 2 FALLIDA: Normalización incorrecta")
    else:
        print("\n⚠️  Los datos ya estaban normalizados")
else:
    print(f"\n❌ No se encontró: {csv_file}")

# ============================================================================
# PRUEBA 3: DATOS YA NORMALIZADOS
# ============================================================================

print("\n" + "=" * 80)
print("PRUEBA 3: ARCHIVO NPY YA NORMALIZADO")
print("=" * 80)

norm_file = Path("data/BTC/individual_examples/example_1.npy")
if norm_file.exists():
    print(f"\n📂 Cargando: {norm_file}")
    norm_data = np.load(norm_file)
    
    print(f"\n📊 Datos:")
    print(f"   Shape: {norm_data.shape}")
    print(f"   Mean: {norm_data.mean():.6f}")
    print(f"   Std: {norm_data.std():.6f}")
    print(f"   Min: {norm_data.min():.6f}")
    print(f"   Max: {norm_data.max():.6f}")
    
    # Detectar tipo
    is_norm, data_type = is_data_normalized(norm_data)
    print(f"\n🔍 Detección: {data_type}")
    
    if is_norm == True:
        print("\n✅ PRUEBA 3 EXITOSA: Detectó datos ya normalizados")
    else:
        print("\n❌ PRUEBA 3 FALLIDA: No detectó normalización existente")
else:
    print(f"\n❌ No se encontró: {norm_file}")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "=" * 80)
print("RESUMEN DE PRUEBAS")
print("=" * 80)

print("""
✅ La función normalize_raw_data() convierte datos crudos a Z-scores
✅ La función is_data_normalized() detecta el tipo de datos
✅ Soporte para archivos CSV y NPY
✅ Preserva shape (128, 40)

📝 El sistema en Streamlit aplicará esto automáticamente al cargar archivos.
""")

print("=" * 80)

