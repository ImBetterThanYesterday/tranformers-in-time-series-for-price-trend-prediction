#!/usr/bin/env python3
"""
Demo Completo de Inferencia - TLOB Model
=========================================
Script de demostración que muestra todo el proceso:
1. Cargar datos
2. Visualizar estructura
3. Realizar predicciones
4. Interpretar resultados

Uso:
    python3 demo_inference.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from models.tlob import TLOB
from pathlib import Path
import sys

# Configuración
CHECKPOINT_PATH = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/val_loss=0.623_epoch=2.pt"
EXAMPLES_PATH = "data/BTC/inference_examples.npy"
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

def print_section(title, width=80):
    """Imprime un separador bonito"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

def print_subsection(title, width=80):
    """Imprime un subseparador"""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)

def load_model():
    """Carga el modelo TLOB desde el checkpoint"""
    print_section("🤖 CARGANDO MODELO TLOB")
    
    # Verificar que existe el checkpoint
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ Error: No se encuentra el checkpoint en {CHECKPOINT_PATH}")
        print(f"\n💡 Solución: Entrena el modelo primero con:")
        print(f"   python3 main.py +model=tlob +dataset=btc experiment.is_wandb=False")
        sys.exit(1)
    
    print(f"📂 Checkpoint: {CHECKPOINT_PATH}")
    print(f"🖥️  Device: {DEVICE}")
    
    # Crear modelo
    model = TLOB(**MODEL_CONFIG)
    model.to(DEVICE)
    model.eval()
    
    # Cargar pesos
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Remover prefijo "model." del state_dict
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[6:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    
    # Info del modelo
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Modelo cargado exitosamente")
    print(f"   → Parámetros: {num_params:,}")
    print(f"   → Layers: {MODEL_CONFIG['num_layers']}")
    print(f"   → Hidden dim: {MODEL_CONFIG['hidden_dim']}")
    
    # Info del checkpoint
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
        print(f"\n📊 Hiperparámetros del entrenamiento:")
        print(f"   → Horizon: {hparams.get('horizon', 'N/A')} timesteps")
        print(f"   → Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   → Learning rate: {hparams.get('lr', 'N/A')}")
    
    return model

def load_examples():
    """Carga los ejemplos para inferencia"""
    print_section("📦 CARGANDO EJEMPLOS DE ENTRADA")
    
    if not Path(EXAMPLES_PATH).exists():
        print(f"❌ Error: No se encuentran ejemplos en {EXAMPLES_PATH}")
        print(f"\n💡 Solución: Extrae ejemplos con:")
        print(f"   python3 extract_examples.py --split train --num 5 --random")
        sys.exit(1)
    
    # Cargar
    examples = np.load(EXAMPLES_PATH)
    print(f"📂 Archivo: {EXAMPLES_PATH}")
    print(f"📏 Shape original: {examples.shape}")
    print(f"   → {examples.shape[0]} ejemplos")
    print(f"   → {examples.shape[1]} timesteps por ejemplo")
    print(f"   → {examples.shape[2]} features por timestep")
    
    # Usar solo 40 features LOB
    examples = examples[:, :, :40]
    print(f"\n✂️  Usando solo 40 features LOB")
    print(f"📏 Shape procesado: {examples.shape}")
    
    # Estadísticas
    print(f"\n📊 Estadísticas globales:")
    print(f"   → Mean: {examples.mean():.4f}")
    print(f"   → Std:  {examples.std():.4f}")
    print(f"   → Min:  {examples.min():.4f}")
    print(f"   → Max:  {examples.max():.4f}")
    
    return examples

def visualize_example(examples, idx=0):
    """Visualiza la estructura de un ejemplo"""
    print_section(f"🔍 ESTRUCTURA DE UN EJEMPLO (#{idx+1})")
    
    example = examples[idx]
    print(f"Shape: {example.shape} (timesteps × features)")
    
    print_subsection("Primeros 3 timesteps completos:")
    for t in range(3):
        print(f"\nTimestep {t}:")
        print(f"  ASK Prices  (0-9):   {example[t, 0:10].tolist()[:5]} ...")
        print(f"  ASK Volumes (10-19): {example[t, 10:20].tolist()[:5]} ...")
        print(f"  BID Prices  (20-29): {example[t, 20:30].tolist()[:5]} ...")
        print(f"  BID Volumes (30-39): {example[t, 30:40].tolist()[:5]} ...")
    
    print_subsection("Resumen del ejemplo:")
    print(f"Número de snapshots del LOB: {len(example)}")
    print(f"Ventana temporal: timestep 0 → {len(example)-1}")
    print(f"Features por snapshot: {example.shape[1]}")
    print(f"  → 10 niveles ASK prices + 10 niveles ASK volumes")
    print(f"  → 10 niveles BID prices + 10 niveles BID volumes")

def predict(model, examples):
    """Realiza predicciones"""
    print_section("🎯 REALIZANDO PREDICCIONES")
    
    # Convertir a tensor
    X = torch.from_numpy(examples).float().to(DEVICE)
    print(f"📊 Tensor de entrada: {X.shape}")
    
    # Inferencia
    print(f"🔄 Ejecutando forward pass...")
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    print(f"✅ Predicciones completadas")
    print(f"   → Logits shape: {logits.shape}")
    print(f"   → Probabilidades shape: {probs.shape}")
    
    return logits.cpu().numpy(), probs.cpu().numpy(), preds.cpu().numpy()

def display_results(examples, logits, probs, preds):
    """Muestra los resultados de manera amigable"""
    print_section("📊 RESULTADOS DE LAS PREDICCIONES")
    
    for i in range(len(examples)):
        print(f"\n{'='*80}")
        print(f"🔹 EJEMPLO {i+1}/{len(examples)}")
        print(f"{'='*80}")
        
        # Estadísticas de entrada
        ex = examples[i]
        print(f"\n📥 Entrada:")
        print(f"   Mean: {ex.mean():>8.4f}  |  Std:  {ex.std():>8.4f}")
        print(f"   Min:  {ex.min():>8.4f}  |  Max:  {ex.max():>8.4f}")
        
        # Logits
        print(f"\n📊 Logits (salida cruda del modelo):")
        print(f"   DOWN:       {logits[i, 0]:>8.4f}")
        print(f"   STATIONARY: {logits[i, 1]:>8.4f}")
        print(f"   UP:         {logits[i, 2]:>8.4f}")
        
        # Probabilidades
        print(f"\n🎲 Probabilidades (después de softmax):")
        print(f"   📉 DOWN:       {probs[i, 0]:>7.2%}")
        print(f"   ➡️  STATIONARY: {probs[i, 1]:>7.2%}")
        print(f"   📈 UP:         {probs[i, 2]:>7.2%}")
        
        # Predicción final
        pred_class = preds[i]
        pred_label = CLASS_LABELS[pred_class]
        pred_emoji = CLASS_EMOJIS[pred_class]
        confidence = probs[i, pred_class]
        
        print(f"\n{'*'*80}")
        print(f"🎯 PREDICCIÓN: {pred_emoji} {pred_label} (clase {pred_class})")
        print(f"💪 CONFIANZA:  {confidence:.2%}")
        print(f"{'*'*80}")
        
        # Interpretación
        if confidence > 0.95:
            conf_text = "MUY ALTA"
        elif confidence > 0.85:
            conf_text = "ALTA"
        elif confidence > 0.70:
            conf_text = "MODERADA"
        else:
            conf_text = "BAJA"
        
        print(f"\n💡 Interpretación:")
        print(f"   El modelo predice que el precio estará {pred_label}")
        print(f"   en los próximos 10 timesteps con confianza {conf_text}.")
        
        if pred_label == "DOWN":
            print(f"   → Tendencia bajista esperada 📉")
        elif pred_label == "UP":
            print(f"   → Tendencia alcista esperada 📈")
        else:
            print(f"   → Precio se mantendrá estable ➡️")

def save_results(logits, probs, preds):
    """Guarda los resultados"""
    print_section("💾 GUARDANDO RESULTADOS")
    
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "demo_logits.npy", logits)
    np.save(output_dir / "demo_probabilities.npy", probs)
    np.save(output_dir / "demo_predictions.npy", preds)
    
    print(f"✅ Resultados guardados en: {output_dir}/")
    print(f"   → demo_logits.npy")
    print(f"   → demo_probabilities.npy")
    print(f"   → demo_predictions.npy")
    
    print(f"\n📖 Puedes cargarlos con:")
    print(f"   probs = np.load('inference_results/demo_probabilities.npy')")

def main():
    """Función principal"""
    print("\n" + "🚀" * 40)
    print("  DEMO DE INFERENCIA - TLOB MODEL")
    print("🚀" * 40)
    
    try:
        # 1. Cargar modelo
        model = load_model()
        
        # 2. Cargar ejemplos
        examples = load_examples()
        
        # 3. Visualizar un ejemplo
        visualize_example(examples, idx=0)
        
        # 4. Realizar predicciones
        logits, probs, preds = predict(model, examples)
        
        # 5. Mostrar resultados
        display_results(examples, logits, probs, preds)
        
        # 6. Guardar resultados
        save_results(logits, probs, preds)
        
        # Finalizar
        print_section("✅ DEMO COMPLETADA CON ÉXITO")
        print("\n💡 Próximos pasos:")
        print("   1. Explora los resultados guardados en inference_results/")
        print("   2. Lee docs/inference_guide.md para más detalles")
        print("   3. Extrae nuevos ejemplos con extract_examples.py")
        print("   4. Experimenta con diferentes checkpoints (horizontes 10, 20, 50, 100)")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

