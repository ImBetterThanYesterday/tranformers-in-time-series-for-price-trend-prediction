#!/usr/bin/env python3
"""
Script de Inspección de Datos BTC
==================================
Explora y visualiza los datos de entrada (.npy) del dataset BTC.

Uso:
    python3 inspect_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("INSPECCIÓN DE DATOS BTC (.npy)")
print("=" * 80)

# ============================================================================
# 1. CARGAR Y EXPLORAR ESTRUCTURA
# ============================================================================
print("\n1. ESTRUCTURA GENERAL")
print("-" * 80)

for split in ['train', 'val', 'test']:
    data = np.load(f'data/BTC/{split}.npy')
    print(f"\n{split.upper()}.npy:")
    print(f"  Shape: {data.shape} (timesteps, features)")
    print(f"  Dtype: {data.dtype}")
    print(f"  Size: {data.nbytes / 1024**2:.1f} MB")
    print(f"  Min/Max: {data.min():.4f} / {data.max():.4f}")
    print(f"  Mean/Std: {data.mean():.4f} / {data.std():.4f}")
    print(f"  NaN count: {np.isnan(data).sum()}")
    print(f"  Inf count: {np.isinf(data).sum()}")

# ============================================================================
# 2. ANALIZAR FEATURES
# ============================================================================
print("\n\n2. ANÁLISIS DE FEATURES")
print("-" * 80)

train = np.load('data/BTC/train.npy')
num_features = train.shape[1]

print(f"\nTotal features: {num_features}")
print("\nEstadísticas por feature (primeras 10):")
print(f"{'Feature':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-" * 58)

for i in range(min(10, num_features)):
    feat = train[:, i]
    print(f"{i:<10} {feat.mean():>11.4f} {feat.std():>11.4f} {feat.min():>11.4f} {feat.max():>11.4f}")

# ============================================================================
# 3. VISUALIZAR VENTANAS DE EJEMPLO
# ============================================================================
print("\n\n3. VENTANAS DE ENTRADA AL MODELO")
print("-" * 80)

seq_size = 128
print(f"\nSeq size (longitud de ventana): {seq_size}")
print(f"Número total de ventanas posibles: {len(train) - seq_size + 1:,}")

# Extraer una ventana de ejemplo
window = train[:seq_size, :]
print(f"\nEjemplo de ventana:")
print(f"  Shape: {window.shape} (timesteps, features)")
print(f"  Interpretación: {seq_size} snapshots consecutivos del LOB × {num_features} features")
print(f"  Mean: {window.mean():.4f}")
print(f"  Std: {window.std():.4f}")

print("\n  Primeras 3 timesteps × 6 features:")
print(window[:3, :6])

# ============================================================================
# 4. EXAMINAR EJEMPLOS DE INFERENCIA
# ============================================================================
print("\n\n4. EJEMPLOS PARA INFERENCIA")
print("-" * 80)

examples_path = Path("data/BTC/inference_examples.npy")
if examples_path.exists():
    examples = np.load(examples_path)
    print(f"\n✓ Ejemplos cargados: {examples_path}")
    print(f"  Shape: {examples.shape} (num_ejemplos, seq_size, features)")
    
    for i in range(len(examples)):
        ex = examples[i]
        print(f"\n  Ejemplo {i+1}:")
        print(f"    Mean: {ex.mean():.4f}")
        print(f"    Std:  {ex.std():.4f}")
        print(f"    Min:  {ex.min():.4f}")
        print(f"    Max:  {ex.max():.4f}")
else:
    print(f"\n⚠ No se encontró archivo de ejemplos: {examples_path}")
    print("  Ejecuta primero el script de exploración para generarlo.")

# ============================================================================
# 5. DISTRIBUCIÓN DE VALORES
# ============================================================================
print("\n\n5. DISTRIBUCIÓN DE VALORES")
print("-" * 80)

# Tomar muestra para visualización (para no sobrecargar memoria)
sample = train[::100, :]  # cada 100 timesteps
print(f"\nMuestra tomada: {sample.shape[0]} timesteps")

print("\nHistograma de valores (todas las features combinadas):")
flat_sample = sample.flatten()
print(f"  Percentiles: p5={np.percentile(flat_sample, 5):.4f}, "
      f"p50={np.percentile(flat_sample, 50):.4f}, "
      f"p95={np.percentile(flat_sample, 95):.4f}")

# ============================================================================
# 6. ESTRUCTURA DEL LOB
# ============================================================================
print("\n\n6. ESTRUCTURA DEL LIMIT ORDER BOOK (LOB)")
print("-" * 80)

print("\nCada snapshot del LOB contiene 40 features principales:")
print("  → Primeros 10 niveles de precios ASK (sell orders)")
print("  → Primeros 10 niveles de volúmenes ASK")
print("  → Primeros 10 niveles de precios BID (buy orders)")
print("  → Primeros 10 niveles de volúmenes BID")
print(f"\nEn este dataset hay {num_features} features totales:")
print("  → 40 features del LOB (precios + volúmenes)")
print("  → 4 features adicionales (posiblemente labels/metadata)")

# Visualizar la estructura de un snapshot
snapshot = train[0, :40]  # Primeros 40 features (LOB puro)
print("\nEjemplo de snapshot del LOB (índice 0):")
print("\nASK prices (primeros 5 niveles):", snapshot[:5])
print("ASK volumes (primeros 5 niveles):", snapshot[10:15])
print("BID prices (primeros 5 niveles):", snapshot[20:25])
print("BID volumes (primeros 5 niveles):", snapshot[30:35])

# ============================================================================
# 7. GUARDAR VISUALIZACIONES
# ============================================================================
print("\n\n7. GENERANDO VISUALIZACIONES")
print("-" * 80)

output_dir = Path("inspection_results")
output_dir.mkdir(exist_ok=True)

# Plot 1: Distribución de valores por feature (primeras 10)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Distribución de valores - Primeras 10 Features", fontsize=14, fontweight='bold')

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.hist(train[::100, i], bins=50, alpha=0.7, edgecolor='black')
    ax.set_title(f'Feature {i}', fontsize=10)
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "feature_distributions.png", dpi=150)
print(f"✓ Guardado: {output_dir}/feature_distributions.png")

# Plot 2: Evolución temporal de 4 features
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Evolución Temporal - 4 Features de Ejemplo", fontsize=14, fontweight='bold')

features_to_plot = [0, 10, 20, 30]  # ASK price, ASK vol, BID price, BID vol
feature_names = ["ASK Price (lvl 1)", "ASK Volume (lvl 1)", "BID Price (lvl 1)", "BID Volume (lvl 1)"]

window_size = 2000  # Primeros 2000 timesteps
for idx, (feat_idx, name) in enumerate(zip(features_to_plot, feature_names)):
    ax = axes[idx // 2, idx % 2]
    ax.plot(train[:window_size, feat_idx], linewidth=0.5, alpha=0.8)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Valor (normalizado)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "temporal_evolution.png", dpi=150)
print(f"✓ Guardado: {output_dir}/temporal_evolution.png")

# Plot 3: Heatmap de una ventana
window = train[:seq_size, :40]  # Solo LOB features
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(window.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
ax.set_title(f"Heatmap de Ventana de Entrada ({seq_size} timesteps × 40 LOB features)", 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Timestep')
ax.set_ylabel('Feature Index')
plt.colorbar(im, ax=ax, label='Valor normalizado')
plt.tight_layout()
plt.savefig(output_dir / "window_heatmap.png", dpi=150)
print(f"✓ Guardado: {output_dir}/window_heatmap.png")

print("\n" + "=" * 80)
print("✓ INSPECCIÓN COMPLETADA")
print("=" * 80)
print(f"\nResultados guardados en: {output_dir}/")

