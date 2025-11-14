#!/usr/bin/env python3
"""
Script para Extraer Ejemplos de Entrada desde los Datos BTC
============================================================
Permite extraer ventanas de ejemplo personalizadas desde train/val/test.npy

Uso:
    # Extraer 5 ejemplos aleatorios del train set
    python3 extract_examples.py --split train --num 5 --random

    # Extraer ejemplos específicos por índice
    python3 extract_examples.py --split train --indices 0 1000 2000 3000 4000

    # Extraer ventanas consecutivas
    python3 extract_examples.py --split test --num 10 --consecutive --start 5000
"""

import numpy as np
import argparse
from pathlib import Path

def extract_examples(
    split='train',
    num_examples=5,
    indices=None,
    random=False,
    consecutive=False,
    start_idx=0,
    seq_size=128,
    output_path='data/BTC/inference_examples.npy'
):
    """
    Extrae ventanas de ejemplo desde los datos BTC.
    
    Args:
        split: 'train', 'val' o 'test'
        num_examples: Número de ejemplos a extraer
        indices: Lista de índices específicos (opcional)
        random: Si True, extrae ejemplos aleatorios
        consecutive: Si True, extrae ventanas consecutivas
        start_idx: Índice de inicio
        seq_size: Longitud de cada ventana (default 128)
        output_path: Ruta donde guardar los ejemplos
    """
    
    print("=" * 80)
    print("EXTRACCIÓN DE EJEMPLOS PARA INFERENCIA")
    print("=" * 80)
    
    # Cargar datos
    data_path = f'data/BTC/{split}.npy'
    print(f"\n📂 Cargando {data_path}...")
    data = np.load(data_path)
    print(f"✓ Datos cargados: {data.shape}")
    
    max_start_idx = len(data) - seq_size
    print(f"  → Índices válidos: 0 a {max_start_idx:,}")
    
    # Determinar índices de inicio de cada ventana
    if indices is not None:
        # Usar índices especificados
        start_indices = indices
        num_examples = len(indices)
        print(f"\n📍 Usando índices especificados: {indices}")
        
    elif random:
        # Índices aleatorios
        rng = np.random.RandomState(42)
        start_indices = rng.choice(max_start_idx, size=num_examples, replace=False)
        start_indices = sorted(start_indices)
        print(f"\n🎲 Extrayendo {num_examples} ventanas aleatorias")
        print(f"  Seed: 42")
        
    elif consecutive:
        # Ventanas consecutivas (sin solapamiento)
        start_indices = []
        idx = start_idx
        for i in range(num_examples):
            if idx + seq_size > len(data):
                print(f"  ⚠ Solo se pueden extraer {i} ventanas consecutivas desde {start_idx}")
                break
            start_indices.append(idx)
            idx += seq_size  # No overlap
        num_examples = len(start_indices)
        print(f"\n📊 Extrayendo {num_examples} ventanas consecutivas (sin solapamiento)")
        print(f"  Inicio: {start_idx}")
        
    else:
        # Ventanas equidistantes
        step = max_start_idx // (num_examples + 1)
        start_indices = [start_idx + i * step for i in range(1, num_examples + 1)]
        print(f"\n📏 Extrayendo {num_examples} ventanas equidistantes")
        print(f"  Step: {step:,}")
    
    # Validar índices
    for idx in start_indices:
        if idx < 0 or idx + seq_size > len(data):
            raise ValueError(f"Índice {idx} inválido (rango: 0-{max_start_idx})")
    
    # Extraer ventanas
    print(f"\n" + "=" * 80)
    print("EXTRAYENDO VENTANAS")
    print("=" * 80)
    
    examples = []
    for i, start in enumerate(start_indices):
        window = data[start:start + seq_size]
        examples.append(window)
        
        print(f"\nVentana {i+1}/{num_examples}:")
        print(f"  Índices: {start} → {start + seq_size - 1}")
        print(f"  Shape: {window.shape}")
        print(f"  Mean: {window.mean():.4f}")
        print(f"  Std:  {window.std():.4f}")
        print(f"  Min:  {window.min():.4f}")
        print(f"  Max:  {window.max():.4f}")
        
        # Mostrar primeras 2 filas × 5 features
        print(f"  Primeras 2 timesteps × 5 features:")
        print(window[:2, :5])
    
    # Convertir a array
    examples_array = np.array(examples)
    
    print(f"\n" + "=" * 80)
    print("GUARDANDO EJEMPLOS")
    print("=" * 80)
    
    # Crear directorio si no existe
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar
    np.save(output_path, examples_array)
    
    print(f"\n✓ Ejemplos guardados en: {output_path}")
    print(f"  Shape: {examples_array.shape}")
    print(f"  Interpretación: ({num_examples} ejemplos, {seq_size} timesteps, {data.shape[1]} features)")
    print(f"  Size: {examples_array.nbytes / 1024:.1f} KB")
    
    # Información adicional
    print(f"\n" + "=" * 80)
    print("INFORMACIÓN PARA INFERENCIA")
    print("=" * 80)
    
    print(f"\n✅ Ahora puedes usar estos ejemplos para inferencia:")
    print(f"   python3 inference_pytorch.py")
    print(f"   python3 inference_onnx.py")
    
    print(f"\n📝 Nota: El modelo TLOB usa solo las primeras 40 features (LOB).")
    print(f"   Los scripts de inferencia extraerán [:, :, :40] automáticamente.")
    
    print(f"\n" + "=" * 80)
    print("✓ EXTRACCIÓN COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae ejemplos de ventanas LOB para inferencia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Extraer 5 ejemplos aleatorios del train set
  python3 extract_examples.py --split train --num 5 --random

  # Extraer ejemplos específicos por índice
  python3 extract_examples.py --split train --indices 0 1000 2000 3000 4000

  # Extraer 10 ventanas consecutivas desde el índice 5000
  python3 extract_examples.py --split test --num 10 --consecutive --start 5000

  # Extraer 20 ejemplos equidistantes del validation set
  python3 extract_examples.py --split val --num 20
        """
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split a usar (default: train)'
    )
    
    parser.add_argument(
        '--num',
        type=int,
        default=5,
        help='Número de ejemplos a extraer (default: 5)'
    )
    
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        help='Índices específicos de inicio de ventanas'
    )
    
    parser.add_argument(
        '--random',
        action='store_true',
        help='Extraer ejemplos aleatorios'
    )
    
    parser.add_argument(
        '--consecutive',
        action='store_true',
        help='Extraer ventanas consecutivas (sin solapamiento)'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Índice de inicio (default: 0)'
    )
    
    parser.add_argument(
        '--seq-size',
        type=int,
        default=128,
        help='Longitud de cada ventana (default: 128)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/BTC/inference_examples.npy',
        help='Ruta de salida (default: data/BTC/inference_examples.npy)'
    )
    
    args = parser.parse_args()
    
    extract_examples(
        split=args.split,
        num_examples=args.num,
        indices=args.indices,
        random=args.random,
        consecutive=args.consecutive,
        start_idx=args.start,
        seq_size=args.seq_size,
        output_path=args.output
    )

