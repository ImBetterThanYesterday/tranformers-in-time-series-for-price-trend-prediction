#!/usr/bin/env python3
"""
Crear Ejemplos Individuales para Inferencia
============================================
Extrae ventanas del dataset BTC y las guarda como archivos individuales.
Cada archivo representa UNA inferencia completa.

Uso:
    python3 create_individual_examples.py
"""

import numpy as np
from pathlib import Path

# Configuración
TRAIN_PATH = "data/BTC/train.npy"
OUTPUT_DIR = "data/BTC/individual_examples"
NUM_EXAMPLES = 5
SEQ_SIZE = 128
NUM_FEATURES = 40

print("=" * 80)
print("CREANDO EJEMPLOS INDIVIDUALES PARA INFERENCIA")
print("=" * 80)

# Crear directorio de salida
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# Cargar datos
print(f"\n📂 Cargando {TRAIN_PATH}...")
data = np.load(TRAIN_PATH)
print(f"✓ Datos cargados: {data.shape}")

# Extraer solo las primeras 40 features (LOB)
data_lob = data[:, :40]
print(f"✓ Usando solo 40 features LOB: {data_lob.shape}")

# Definir índices de inicio (equidistantes)
max_start_idx = len(data_lob) - SEQ_SIZE
step = max_start_idx // (NUM_EXAMPLES + 1)
start_indices = [i * step for i in range(1, NUM_EXAMPLES + 1)]

print(f"\n📊 Extrayendo {NUM_EXAMPLES} ejemplos:")
print(f"   → Ventanas de {SEQ_SIZE} timesteps × {NUM_FEATURES} features")
print(f"   → Índices: {start_indices}")

# Extraer y guardar cada ejemplo
for i, start_idx in enumerate(start_indices, 1):
    # Extraer ventana
    window = data_lob[start_idx:start_idx + SEQ_SIZE]
    
    # Verificar shape
    assert window.shape == (SEQ_SIZE, NUM_FEATURES), f"Shape incorrecto: {window.shape}"
    
    # Guardar
    output_path = output_dir / f"example_{i}.npy"
    np.save(output_path, window)
    
    # Mostrar info
    print(f"\n✓ Ejemplo {i} guardado:")
    print(f"   Archivo:  {output_path}")
    print(f"   Índices:  {start_idx} → {start_idx + SEQ_SIZE - 1}")
    print(f"   Shape:    {window.shape}")
    print(f"   Mean:     {window.mean():.4f}")
    print(f"   Std:      {window.std():.4f}")
    print(f"   Min/Max:  {window.min():.4f} / {window.max():.4f}")
    print(f"   Size:     {window.nbytes / 1024:.1f} KB")

# Resumen
print("\n" + "=" * 80)
print("✅ EJEMPLOS INDIVIDUALES CREADOS")
print("=" * 80)
print(f"\n📁 Directorio: {OUTPUT_DIR}/")
print(f"📊 {NUM_EXAMPLES} archivos creados:")

for i in range(1, NUM_EXAMPLES + 1):
    filename = f"example_{i}.npy"
    filepath = output_dir / filename
    size_kb = filepath.stat().st_size / 1024
    print(f"   {i}. {filename:<20} ({size_kb:.1f} KB)")

print(f"\n💡 Cada archivo representa una inferencia individual:")
print(f"   Shape por archivo: ({SEQ_SIZE}, {NUM_FEATURES})")
print(f"   → {SEQ_SIZE} timesteps consecutivos del LOB")
print(f"   → {NUM_FEATURES} features por timestep")

print(f"\n🎯 Ahora puedes hacer inferencia sobre cada archivo:")
print(f"   python3 inference_single_file.py data/BTC/individual_examples/example_1.npy")

print("\n" + "=" * 80)


