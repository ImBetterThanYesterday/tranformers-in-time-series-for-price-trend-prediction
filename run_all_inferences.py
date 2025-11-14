#!/usr/bin/env python3
"""
Ejecutar Inferencia sobre Todos los Ejemplos Individuales
==========================================================
Procesa todos los archivos example_*.npy y genera un resumen.

Uso:
    python3 run_all_inferences.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from models.tlob import TLOB
from pathlib import Path
import subprocess

# Configuración
EXAMPLES_DIR = "data/BTC/individual_examples"
CHECKPOINT_PATH = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/val_loss=0.623_epoch=2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "hidden_dim": 40,
    "num_layers": 4,
    "seq_size": 128,
    "num_features": 40,
    "num_heads": 1,
    "is_sin_emb": True,
    "dataset_type": "BTC",
}

CLASS_LABELS = {0: "DOWN", 1: "STATIONARY", 2: "UP"}
CLASS_EMOJIS = {0: "📉", 1: "➡️", 2: "📈"}

print("=" * 80)
print("INFERENCIA EN LOTE - TODOS LOS EJEMPLOS INDIVIDUALES")
print("=" * 80)

# Cargar modelo una sola vez
print(f"\n🤖 Cargando modelo...")
model = TLOB(**MODEL_CONFIG)
model.to(DEVICE)
model.eval()

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
state_dict = checkpoint["state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("model."):
        new_key = key[6:]
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value
model.load_state_dict(new_state_dict)

print(f"✅ Modelo cargado: {sum(p.numel() for p in model.parameters()):,} parámetros")

# Buscar solo los archivos example_N.npy (no los _result.npy)
examples_dir = Path(EXAMPLES_DIR)
all_files = sorted(examples_dir.glob("example_*.npy"))
example_files = [f for f in all_files if not f.stem.endswith("_result")]

if not example_files:
    print(f"❌ No se encontraron archivos en {EXAMPLES_DIR}")
    print(f"💡 Ejecuta primero: python3 create_individual_examples.py")
    exit(1)

print(f"\n📁 Encontrados {len(example_files)} archivos")

# Procesar cada archivo
results = []

for i, file_path in enumerate(example_files, 1):
    print(f"\n{'='*80}")
    print(f"PROCESANDO: {file_path.name} ({i}/{len(example_files)})")
    print(f"{'='*80}")
    
    # Cargar ejemplo
    example = np.load(file_path)
    
    # Inferencia
    X = np.expand_dims(example, axis=0)
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
    
    logits_np = logits[0].cpu().numpy()
    probs_np = probs[0].cpu().numpy()
    pred_np = pred[0].item()
    
    # Guardar resultado
    result = {
        'logits': logits_np,
        'probabilities': probs_np,
        'prediction': pred_np,
        'prediction_label': CLASS_LABELS[pred_np],
        'confidence': float(probs_np[pred_np])
    }
    
    result_path = file_path.parent / f"{file_path.stem}_result.npy"
    np.save(result_path, result)
    
    # Mostrar info
    print(f"\n📊 Estadísticas de entrada:")
    print(f"   Mean: {example.mean():.4f} | Std: {example.std():.4f}")
    print(f"   Min:  {example.min():.4f} | Max: {example.max():.4f}")
    
    print(f"\n🎯 Predicción:")
    print(f"   {CLASS_EMOJIS[pred_np]} {CLASS_LABELS[pred_np]} (confianza: {probs_np[pred_np]:.2%})")
    
    print(f"\n🎲 Probabilidades:")
    for j, label in CLASS_LABELS.items():
        emoji = CLASS_EMOJIS[j]
        bar_length = int(probs_np[j] * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {emoji} {label:12s} {bar} {probs_np[j]:>6.2%}")
    
    print(f"\n💾 Resultado guardado: {result_path.name}")
    
    # Guardar para resumen
    results.append({
        'file': file_path.name,
        'prediction': CLASS_LABELS[pred_np],
        'confidence': probs_np[pred_np],
        'logits': logits_np,
        'probs': probs_np,
        'stats': {
            'mean': example.mean(),
            'std': example.std(),
            'min': example.min(),
            'max': example.max()
        }
    })

# Resumen final
print("\n" + "=" * 80)
print("📊 RESUMEN DE TODAS LAS INFERENCIAS")
print("=" * 80)

print(f"\nTotal de archivos procesados: {len(results)}")

# Tabla resumen
print(f"\n{'Archivo':<20} {'Predicción':<12} {'Confianza':<10} {'Logits [D, S, U]'}")
print("-" * 80)

for r in results:
    emoji = CLASS_EMOJIS[[k for k, v in CLASS_LABELS.items() if v == r['prediction']][0]]
    logits_str = f"[{r['logits'][0]:>5.2f}, {r['logits'][1]:>5.2f}, {r['logits'][2]:>5.2f}]"
    print(f"{r['file']:<20} {emoji} {r['prediction']:<10} {r['confidence']:<9.2%} {logits_str}")

# Distribución de predicciones
print(f"\n📈 Distribución de Predicciones:")
pred_counts = {}
for r in results:
    pred = r['prediction']
    pred_counts[pred] = pred_counts.get(pred, 0) + 1

for label in CLASS_LABELS.values():
    count = pred_counts.get(label, 0)
    pct = count / len(results) * 100
    emoji = CLASS_EMOJIS[[k for k, v in CLASS_LABELS.items() if v == label][0]]
    bar_length = int(pct / 100 * 40)
    bar = "█" * bar_length + "░" * (40 - bar_length)
    print(f"   {emoji} {label:12s} {bar} {count}/{len(results)} ({pct:.1f}%)")

# Estadísticas de confianza
confidences = [r['confidence'] for r in results]
print(f"\n💪 Confianza:")
print(f"   Promedio: {np.mean(confidences):.2%}")
print(f"   Mínimo:   {np.min(confidences):.2%}")
print(f"   Máximo:   {np.max(confidences):.2%}")

# Guardar resumen
summary_path = examples_dir / "summary_all_inferences.txt"
with open(summary_path, 'w') as f:
    f.write("RESUMEN DE INFERENCIAS - EJEMPLOS INDIVIDUALES\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total de archivos: {len(results)}\n")
    f.write(f"Modelo: TLOB (BTC, horizon=10, seed=1)\n")
    f.write(f"Checkpoint: val_loss=0.623_epoch=2\n\n")
    
    f.write(f"{'Archivo':<20} {'Predicción':<12} {'Confianza':<10} {'Logits [D, S, U]'}\n")
    f.write("-" * 80 + "\n")
    
    for r in results:
        emoji = CLASS_EMOJIS[[k for k, v in CLASS_LABELS.items() if v == r['prediction']][0]]
        logits_str = f"[{r['logits'][0]:>5.2f}, {r['logits'][1]:>5.2f}, {r['logits'][2]:>5.2f}]"
        f.write(f"{r['file']:<20} {emoji} {r['prediction']:<10} {r['confidence']:<9.2%} {logits_str}\n")
    
    f.write(f"\nDistribución de Predicciones:\n")
    for label in CLASS_LABELS.values():
        count = pred_counts.get(label, 0)
        pct = count / len(results) * 100
        emoji = CLASS_EMOJIS[[k for k, v in CLASS_LABELS.items() if v == label][0]]
        f.write(f"  {emoji} {label}: {count}/{len(results)} ({pct:.1f}%)\n")
    
    f.write(f"\nConfianza:\n")
    f.write(f"  Promedio: {np.mean(confidences):.2%}\n")
    f.write(f"  Rango: {np.min(confidences):.2%} - {np.max(confidences):.2%}\n")

print(f"\n💾 Resumen guardado en: {summary_path}")

print("\n" + "=" * 80)
print("✅ TODAS LAS INFERENCIAS COMPLETADAS")
print("=" * 80)

print(f"\n📁 Archivos generados en: {EXAMPLES_DIR}/")
print(f"   • {len(results)} archivos example_N.npy (entradas)")
print(f"   • {len(results)} archivos example_N_result.npy (resultados)")
print(f"   • 1 archivo summary_all_inferences.txt (resumen)")

