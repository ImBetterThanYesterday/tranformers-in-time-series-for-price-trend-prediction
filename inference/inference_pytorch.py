#!/usr/bin/env python3
"""
Script de Inferencia con PyTorch - TLOB Model
==============================================
Carga el checkpoint .pt del modelo TLOB y realiza predicciones sobre ejemplos de entrada.

Uso:
    python3 inference_pytorch.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from src.models.tlob import TLOB
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CHECKPOINT_PATH = "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/val_loss=0.623_epoch=2.pt"
EXAMPLES_PATH = "src/data/BTC/inference_examples.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparámetros del modelo (deben coincidir con los del entrenamiento)
MODEL_CONFIG = {
    "hidden_dim": 40,
    "num_layers": 4,
    "seq_size": 128,
    "num_features": 40,  # El modelo fue entrenado con 40 features (solo LOB, sin las 4 adicionales)
    "num_heads": 1,
    "is_sin_emb": True,
    "dataset_type": "BTC",
}

print("=" * 80)
print("INFERENCIA CON PYTORCH - MODELO TLOB")
print("=" * 80)
print(f"\n📌 Device: {DEVICE}")
print(f"📌 Checkpoint: {CHECKPOINT_PATH}")
print(f"📌 Ejemplos: {EXAMPLES_PATH}")

# ============================================================================
# 1. CARGAR MODELO
# ============================================================================
print("\n" + "=" * 80)
print("1. CARGANDO MODELO")
print("=" * 80)

# Instanciar el modelo
model = TLOB(**MODEL_CONFIG)
model.to(DEVICE)
model.eval()

# Cargar los pesos entrenados
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

# El checkpoint tiene un prefijo "model." que necesitamos remover
state_dict = checkpoint["state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("model."):
        new_key = key[6:]  # Remover "model." prefix
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict)

print(f"✓ Modelo cargado exitosamente")
print(f"  → Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"  → Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# 2. CARGAR DATOS DE ENTRADA
# ============================================================================
print("\n" + "=" * 80)
print("2. CARGANDO DATOS DE ENTRADA")
print("=" * 80)

examples = np.load(EXAMPLES_PATH)
print(f"✓ Ejemplos cargados: {examples.shape}")
print(f"  → Num ejemplos: {examples.shape[0]}")
print(f"  → Seq size: {examples.shape[1]}")
print(f"  → Features: {examples.shape[2]}")

# Tomar solo las primeras 40 features (LOB puro, sin metadata)
examples = examples[:, :, :40]
print(f"\n✓ Usando solo 40 features LOB: {examples.shape}")

# Convertir a tensor PyTorch
X = torch.from_numpy(examples).float().to(DEVICE)
print(f"✓ Tensor creado: {X.shape} ({X.dtype})")

# ============================================================================
# 3. REALIZAR INFERENCIA
# ============================================================================
print("\n" + "=" * 80)
print("3. REALIZANDO INFERENCIA")
print("=" * 80)

with torch.no_grad():
    # Forward pass
    logits = model(X)
    print(f"✓ Logits obtenidos: {logits.shape}")
    print(f"  → (batch_size={logits.shape[0]}, num_classes={logits.shape[1]})")
    
    # Aplicar softmax para obtener probabilidades
    probs = F.softmax(logits, dim=1)
    
    # Obtener la clase predicha
    preds = torch.argmax(probs, dim=1)

# ============================================================================
# 4. MOSTRAR RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("4. RESULTADOS DE LA PREDICCIÓN")
print("=" * 80)

class_labels = {0: "DOWN", 1: "STATIONARY", 2: "UP"}

for i in range(len(examples)):
    print(f"\n--- Ejemplo {i+1} ---")
    print(f"  Logits: [{logits[i,0]:.4f}, {logits[i,1]:.4f}, {logits[i,2]:.4f}]")
    print(f"  Probabilidades:")
    print(f"    • DOWN:       {probs[i,0]:.2%}")
    print(f"    • STATIONARY: {probs[i,1]:.2%}")
    print(f"    • UP:         {probs[i,2]:.2%}")
    print(f"  Predicción: {class_labels[preds[i].item()]} (clase {preds[i].item()})")
    
    # Confianza de la predicción
    confidence = probs[i, preds[i]].item()
    print(f"  Confianza: {confidence:.2%}")

# ============================================================================
# 5. GUARDAR PREDICCIONES
# ============================================================================
print("\n" + "=" * 80)
print("5. GUARDANDO PREDICCIONES")
print("=" * 80)

output_dir = Path("inference_results")
output_dir.mkdir(exist_ok=True)

# Guardar predicciones y probabilidades
np.save(output_dir / "predictions_pytorch.npy", preds.cpu().numpy())
np.save(output_dir / "probabilities_pytorch.npy", probs.cpu().numpy())
np.save(output_dir / "logits_pytorch.npy", logits.cpu().numpy())

print(f"✓ Resultados guardados en: {output_dir}/")
print(f"  → predictions_pytorch.npy")
print(f"  → probabilities_pytorch.npy")
print(f"  → logits_pytorch.npy")

# ============================================================================
# 6. ANÁLISIS ADICIONAL
# ============================================================================
print("\n" + "=" * 80)
print("6. ANÁLISIS DE ENTRADA")
print("=" * 80)

print("\nEstadísticas de cada ejemplo:")
for i in range(len(examples)):
    ex = examples[i]
    print(f"\nEjemplo {i+1}:")
    print(f"  Mean: {ex.mean():.4f}")
    print(f"  Std:  {ex.std():.4f}")
    print(f"  Min:  {ex.min():.4f}")
    print(f"  Max:  {ex.max():.4f}")
    print(f"  → Predicción: {class_labels[preds[i].item()]}")

print("\n" + "=" * 80)
print("✓ INFERENCIA COMPLETADA")
print("=" * 80)

