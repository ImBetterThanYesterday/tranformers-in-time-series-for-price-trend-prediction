#!/usr/bin/env python3
"""
Script de Inferencia con ONNX Runtime - TLOB Model
===================================================
Carga el modelo ONNX exportado y realiza predicciones optimizadas.
ONNX Runtime es más rápido y portátil que PyTorch para inferencia en producción.

Uso:
    python3 inference_onnx.py
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
ONNX_PATH = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/onnx/val_loss=0.623_epoch=2.onnx"
EXAMPLES_PATH = "data/BTC/inference_examples.npy"

print("=" * 80)
print("INFERENCIA CON ONNX RUNTIME - MODELO TLOB")
print("=" * 80)
print(f"\n📌 ONNX Model: {ONNX_PATH}")
print(f"📌 Ejemplos: {EXAMPLES_PATH}")

# ============================================================================
# 1. CARGAR SESIÓN ONNX
# ============================================================================
print("\n" + "=" * 80)
print("1. CARGANDO SESIÓN ONNX")
print("=" * 80)

# Crear sesión ONNX con optimizaciones
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Crear sesión (usa CPU por defecto, pero puede acelerarse con GPU si está disponible)
providers = ['CPUExecutionProvider']
if ort.get_device() == 'GPU':
    providers.insert(0, 'CUDAExecutionProvider')

session = ort.InferenceSession(ONNX_PATH, session_options, providers=providers)

print(f"✓ Sesión ONNX creada")
print(f"  → Providers: {session.get_providers()}")

# Obtener información de entradas y salidas del modelo
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print(f"\n📥 INPUT del modelo:")
print(f"  Nombre: {input_info.name}")
print(f"  Shape: {input_info.shape}")
print(f"  Type: {input_info.type}")

print(f"\n📤 OUTPUT del modelo:")
print(f"  Nombre: {output_info.name}")
print(f"  Shape: {output_info.shape}")
print(f"  Type: {output_info.type}")

# ============================================================================
# 2. CARGAR DATOS DE ENTRADA
# ============================================================================
print("\n" + "=" * 80)
print("2. CARGANDO DATOS DE ENTRADA")
print("=" * 80)

examples = np.load(EXAMPLES_PATH).astype(np.float32)
print(f"✓ Ejemplos cargados: {examples.shape}")
print(f"  → Num ejemplos: {examples.shape[0]}")
print(f"  → Seq size: {examples.shape[1]}")
print(f"  → Features: {examples.shape[2]}")
print(f"  → Dtype: {examples.dtype}")

# Tomar solo las primeras 40 features (LOB puro, sin metadata)
examples = examples[:, :, :40]
print(f"\n✓ Usando solo 40 features LOB: {examples.shape}")

# ============================================================================
# 3. REALIZAR INFERENCIA
# ============================================================================
print("\n" + "=" * 80)
print("3. REALIZANDO INFERENCIA")
print("=" * 80)

# Preparar entrada (ONNX necesita un diccionario con el nombre de entrada)
input_name = input_info.name
onnx_input = {input_name: examples}

# Ejecutar inferencia
outputs = session.run(None, onnx_input)
logits = outputs[0]

print(f"✓ Logits obtenidos: {logits.shape}")
print(f"  → (batch_size={logits.shape[0]}, num_classes={logits.shape[1]})")

# Aplicar softmax manualmente para obtener probabilidades
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Obtener la clase predicha
preds = np.argmax(probs, axis=1)

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
    print(f"  Predicción: {class_labels[preds[i]]} (clase {preds[i]})")
    
    # Confianza de la predicción
    confidence = probs[i, preds[i]]
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
np.save(output_dir / "predictions_onnx.npy", preds)
np.save(output_dir / "probabilities_onnx.npy", probs)
np.save(output_dir / "logits_onnx.npy", logits)

print(f"✓ Resultados guardados en: {output_dir}/")
print(f"  → predictions_onnx.npy")
print(f"  → probabilities_onnx.npy")
print(f"  → logits_onnx.npy")

# ============================================================================
# 6. BENCHMARK DE VELOCIDAD
# ============================================================================
print("\n" + "=" * 80)
print("6. BENCHMARK DE VELOCIDAD")
print("=" * 80)

import time

num_iterations = 100
times = []

for _ in range(num_iterations):
    start = time.perf_counter()
    session.run(None, onnx_input)
    end = time.perf_counter()
    times.append(end - start)

mean_time = np.mean(times) * 1000  # en ms
std_time = np.std(times) * 1000
throughput = len(examples) / (mean_time / 1000)  # ejemplos por segundo

print(f"✓ Benchmark completado ({num_iterations} iteraciones):")
print(f"  → Tiempo promedio: {mean_time:.2f} ± {std_time:.2f} ms")
print(f"  → Throughput: {throughput:.1f} ejemplos/segundo")
print(f"  → Latencia por ejemplo: {mean_time / len(examples):.2f} ms")

# ============================================================================
# 7. ANÁLISIS ADICIONAL
# ============================================================================
print("\n" + "=" * 80)
print("7. ANÁLISIS DE ENTRADA")
print("=" * 80)

print("\nEstadísticas de cada ejemplo:")
for i in range(len(examples)):
    ex = examples[i]
    print(f"\nEjemplo {i+1}:")
    print(f"  Mean: {ex.mean():.4f}")
    print(f"  Std:  {ex.std():.4f}")
    print(f"  Min:  {ex.min():.4f}")
    print(f"  Max:  {ex.max():.4f}")
    print(f"  → Predicción: {class_labels[preds[i]]}")

print("\n" + "=" * 80)
print("✓ INFERENCIA COMPLETADA")
print("=" * 80)

