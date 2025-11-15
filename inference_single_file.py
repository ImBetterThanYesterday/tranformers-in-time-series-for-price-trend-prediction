#!/usr/bin/env python3
"""
Inferencia sobre un Archivo Individual
========================================
Realiza inferencia sobre UN SOLO archivo .npy que contiene una ventana.

Uso:
    python3 inference_single_file.py <path_to_example.npy>
    
Ejemplo:
    python3 inference_single_file.py data/BTC/individual_examples/example_1.npy
"""

import numpy as np
import torch
import torch.nn.functional as F
from models.tlob import TLOB
from pathlib import Path
import sys

# Configuración del modelo
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

def print_section(title, width=80):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

def load_model():
    """Carga el modelo TLOB"""
    model = TLOB(**MODEL_CONFIG)
    model.to(DEVICE)
    model.eval()
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Remover prefijo "model."
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[6:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    
    return model

def load_example(file_path):
    """Carga un ejemplo individual desde archivo"""
    if not Path(file_path).exists():
        print(f"❌ Error: Archivo no encontrado: {file_path}")
        sys.exit(1)
    
    example = np.load(file_path)
    
    # Verificar shape
    expected_shape = (128, 40)
    if example.shape != expected_shape:
        print(f"❌ Error: Shape incorrecto")
        print(f"   Esperado: {expected_shape}")
        print(f"   Recibido: {example.shape}")
        sys.exit(1)
    
    return example

def predict(model, example):
    """Realiza predicción sobre un ejemplo"""
    # Añadir dimensión de batch
    X = np.expand_dims(example, axis=0)  # (128, 40) → (1, 128, 40)
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    # Inferencia
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
    
    return logits[0].cpu().numpy(), probs[0].cpu().numpy(), pred[0].item()

def display_results(file_path, example, logits, probs, pred):
    """Muestra los resultados de manera amigable"""
    print_section(f"🎯 RESULTADO DE INFERENCIA")
    
    print(f"\n📂 Archivo: {file_path}")
    print(f"📊 Shape: {example.shape} (timesteps × features)")
    
    # Estadísticas de entrada
    print(f"\n📥 Estadísticas de Entrada:")
    print(f"   Mean: {example.mean():>8.4f}")
    print(f"   Std:  {example.std():>8.4f}")
    print(f"   Min:  {example.min():>8.4f}")
    print(f"   Max:  {example.max():>8.4f}")
    
    # Logits
    print(f"\n📊 Logits (salida cruda):")
    print(f"   DOWN:       {logits[0]:>8.4f}")
    print(f"   STATIONARY: {logits[1]:>8.4f}")
    print(f"   UP:         {logits[2]:>8.4f}")
    
    # Probabilidades
    print(f"\n🎲 Probabilidades:")
    print(f"   📉 DOWN:       {probs[0]:>7.2%}")
    print(f"   ➡️  STATIONARY: {probs[1]:>7.2%}")
    print(f"   📈 UP:         {probs[2]:>7.2%}")
    
    # Predicción final
    pred_label = CLASS_LABELS[pred]
    pred_emoji = CLASS_EMOJIS[pred]
    confidence = probs[pred]
    
    print("\n" + "*" * 80)
    print(f"🎯 PREDICCIÓN: {pred_emoji} {pred_label} (clase {pred})".center(80))
    print(f"💪 CONFIANZA:  {confidence:.2%}".center(80))
    print("*" * 80)
    
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

def save_result(file_path, logits, probs, pred):
    """Guarda el resultado junto al archivo de entrada"""
    input_path = Path(file_path)
    output_dir = input_path.parent
    base_name = input_path.stem
    
    # Guardar resultado
    result = {
        'logits': logits,
        'probabilities': probs,
        'prediction': pred,
        'prediction_label': CLASS_LABELS[pred],
        'confidence': float(probs[pred])
    }
    
    output_path = output_dir / f"{base_name}_result.npy"
    np.save(output_path, result)
    
    print(f"\n💾 Resultado guardado en: {output_path}")
    
    # También guardar en formato texto legible
    txt_path = output_dir / f"{base_name}_result.txt"
    with open(txt_path, 'w') as f:
        f.write(f"RESULTADO DE INFERENCIA\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Archivo de entrada: {file_path}\n\n")
        f.write(f"Logits:\n")
        f.write(f"  DOWN:       {logits[0]:.4f}\n")
        f.write(f"  STATIONARY: {logits[1]:.4f}\n")
        f.write(f"  UP:         {logits[2]:.4f}\n\n")
        f.write(f"Probabilidades:\n")
        f.write(f"  DOWN:       {probs[0]:.2%}\n")
        f.write(f"  STATIONARY: {probs[1]:.2%}\n")
        f.write(f"  UP:         {probs[2]:.2%}\n\n")
        f.write(f"Predicción: {CLASS_LABELS[pred]} (clase {pred})\n")
        f.write(f"Confianza: {probs[pred]:.2%}\n")
    
    print(f"💾 Resultado en texto: {txt_path}")

def main():
    if len(sys.argv) != 2:
        print("❌ Error: Debes proporcionar el path al archivo de ejemplo")
        print(f"\nUso:")
        print(f"  python3 {sys.argv[0]} <path_to_example.npy>")
        print(f"\nEjemplo:")
        print(f"  python3 {sys.argv[0]} data/BTC/individual_examples/example_1.npy")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("\n" + "🚀" * 40)
    print("  INFERENCIA SOBRE ARCHIVO INDIVIDUAL")
    print("🚀" * 40)
    
    try:
        # 1. Cargar modelo
        print_section("🤖 CARGANDO MODELO")
        print(f"📂 Checkpoint: {CHECKPOINT_PATH}")
        print(f"🖥️  Device: {DEVICE}")
        model = load_model()
        print(f"✅ Modelo cargado: {sum(p.numel() for p in model.parameters()):,} parámetros")
        
        # 2. Cargar ejemplo
        print_section("📦 CARGANDO EJEMPLO")
        example = load_example(file_path)
        print(f"✅ Ejemplo cargado correctamente")
        print(f"   Shape: {example.shape}")
        
        # 3. Realizar predicción
        print_section("🎯 REALIZANDO PREDICCIÓN")
        logits, probs, pred = predict(model, example)
        print(f"✅ Predicción completada")
        
        # 4. Mostrar resultados
        display_results(file_path, example, logits, probs, pred)
        
        # 5. Guardar resultado
        save_result(file_path, logits, probs, pred)
        
        print_section("✅ INFERENCIA COMPLETADA CON ÉXITO")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


