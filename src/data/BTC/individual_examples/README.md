# 📦 Ejemplos Individuales para Inferencia

> **5 archivos independientes, cada uno representa UNA inferencia completa**

---

## 📁 Contenido del Directorio

```
individual_examples/
├── example_1.npy             # Entrada: ventana LOB (128×40)
├── example_1_result.npy      # Salida: predicción + probabilidades
├── example_1_result.txt      # Resultado legible en texto
│
├── example_2.npy
├── example_2_result.npy
├── example_2_result.txt
│
├── example_3.npy
├── example_3_result.npy
├── example_3_result.txt
│
├── example_4.npy
├── example_4_result.npy
├── example_4_result.txt
│
├── example_5.npy
├── example_5_result.npy
├── example_5_result.txt
│
├── summary_all_inferences.txt  # Resumen de todos los resultados
└── README.md                    # Este archivo
```

---

## 🎯 ¿Qué representa cada archivo?

### Archivos de Entrada (`example_N.npy`)

Cada archivo `example_N.npy` contiene:
- **Shape:** `(128, 40)`
- **Contenido:** Una ventana temporal del Limit Order Book
  - 128 timesteps consecutivos
  - 40 features LOB por timestep
- **Tipo:** `numpy.ndarray` (float64)
- **Tamaño:** ~40 KB cada uno

**Estructura de los 40 features:**
```
Features 0-9:   ASK Prices  (10 niveles)
Features 10-19: ASK Volumes (10 niveles)
Features 20-29: BID Prices  (10 niveles)
Features 30-39: BID Volumes (10 niveles)
```

### Archivos de Resultado (`example_N_result.npy`)

Cada archivo `example_N_result.npy` contiene un diccionario con:
```python
{
    'logits': array([logit_down, logit_stat, logit_up]),
    'probabilities': array([prob_down, prob_stat, prob_up]),
    'prediction': int,  # 0=DOWN, 1=STATIONARY, 2=UP
    'prediction_label': str,  # "DOWN", "STATIONARY", "UP"
    'confidence': float  # Probabilidad de la clase predicha
}
```

### Archivos de Texto (`example_N_result.txt`)

Versión legible de los resultados para revisión rápida.

---

## 📊 Resumen de Resultados

| Archivo | Predicción | Confianza | Logits [D, S, U] |
|---------|------------|-----------|------------------|
| `example_1.npy` | ➡️ **STATIONARY** | **92.06%** | [-3.67, 3.06, 0.60] |
| `example_2.npy` | 📈 **UP** | **55.15%** | [-0.05, -0.42, 0.68] |
| `example_3.npy` | 📈 **UP** | **93.81%** | [-1.30, -0.57, 2.54] |
| `example_4.npy` | ➡️ **STATIONARY** | **77.45%** | [-0.11, 1.37, -1.39] |
| `example_5.npy` | 📉 **DOWN** | **86.90%** | [1.71, -0.69, -1.11] |

### Distribución de Predicciones

```
📉 DOWN:       1/5 (20%)  ████████
➡️ STATIONARY: 2/5 (40%)  ████████████████
📈 UP:         2/5 (40%)  ████████████████
```

### Confianza Promedio

- **Promedio:** 81.07%
- **Rango:** 55.15% - 93.81%

---

## 🚀 Cómo Usar los Archivos

### 1. Cargar un ejemplo para inferencia

```python
import numpy as np

# Cargar ventana de entrada
example = np.load('example_1.npy')
print(f"Shape: {example.shape}")  # (128, 40)

# Ver estadísticas
print(f"Mean: {example.mean():.4f}")
print(f"Std: {example.std():.4f}")
```

### 2. Ejecutar inferencia sobre un archivo

```bash
# Inferencia individual
python3 ../../inference_single_file.py example_1.npy

# O desde el directorio raíz:
cd ../../../
python3 inference_single_file.py data/BTC/individual_examples/example_1.npy
```

### 3. Cargar el resultado de una predicción

```python
import numpy as np

# Cargar resultado
result = np.load('example_1_result.npy', allow_pickle=True).item()

print(f"Predicción: {result['prediction_label']}")
print(f"Confianza: {result['confidence']:.2%}")
print(f"Probabilidades:")
print(f"  DOWN: {result['probabilities'][0]:.2%}")
print(f"  STATIONARY: {result['probabilities'][1]:.2%}")
print(f"  UP: {result['probabilities'][2]:.2%}")
```

### 4. Procesar todos los archivos en lote

```bash
cd ../../../
python3 run_all_inferences.py
```

---

## 📝 Características de los Datos

### Example 1
```
Índices: 463,472 → 463,599
Mean: -0.5904 | Std: 1.0431
Predicción: STATIONARY (92.06%)
Interpretación: Precio estable con alta confianza
```

### Example 2
```
Índices: 926,944 → 927,071
Mean: -0.5133 | Std: 0.5461
Predicción: UP (55.15%)
Interpretación: Tendencia alcista moderada
```

### Example 3
```
Índices: 1,390,416 → 1,390,543
Mean: 0.1325 | Std: 0.6961
Predicción: UP (93.81%)
Interpretación: Tendencia alcista muy fuerte
```

### Example 4
```
Índices: 1,853,888 → 1,854,015
Mean: 0.3223 | Std: 0.5676
Predicción: STATIONARY (77.45%)
Interpretación: Precio estable con confianza alta
```

### Example 5
```
Índices: 2,317,360 → 2,317,487
Mean: 0.3665 | Std: 0.6190
Predicción: DOWN (86.90%)
Interpretación: Tendencia bajista fuerte
```

---

## 🎓 Conceptos Clave

### Ventana de Entrada (Input Window)

Una **ventana** es una secuencia de 128 snapshots consecutivos del LOB:
```
Timestep 0:  [ask_prices[10], ask_vols[10], bid_prices[10], bid_vols[10]]
Timestep 1:  [ask_prices[10], ask_vols[10], bid_prices[10], bid_vols[10]]
...
Timestep 127: [ask_prices[10], ask_vols[10], bid_prices[10], bid_vols[10]]
```

### Horizonte de Predicción

El modelo predice la tendencia en los **próximos 10 timesteps** (horizon=10).

### Clases de Predicción

- **DOWN (0):** Precio bajará 📉
- **STATIONARY (1):** Precio estable ➡️
- **UP (2):** Precio subirá 📈

### Logits vs Probabilidades

```python
# Logits (salida cruda del modelo)
logits = [-3.67, 3.06, 0.60]

# Probabilidades (después de softmax)
# Sum = 1.0, representa confianza
probs = [0.0011, 0.9206, 0.0783]  # [0.11%, 92.06%, 7.83%]
```

---

## 🔧 Scripts Relacionados

### Crear nuevos ejemplos individuales

```bash
cd ../../../
python3 create_individual_examples.py
```

Este script:
- Lee `data/BTC/train.npy`
- Extrae 5 ventanas equidistantes
- Guarda cada una como `example_N.npy`

### Inferencia sobre archivo individual

```bash
cd ../../../
python3 inference_single_file.py data/BTC/individual_examples/example_1.npy
```

Este script:
- Carga el modelo TLOB
- Ejecuta inferencia sobre el archivo
- Muestra resultados detallados
- Guarda `example_1_result.npy` y `example_1_result.txt`

### Procesar todos los ejemplos

```bash
cd ../../../
python3 run_all_inferences.py
```

Este script:
- Procesa todos los `example_N.npy`
- Genera resultados para cada uno
- Crea resumen consolidado
- Muestra distribución de predicciones

---

## 📈 Análisis de Resultados

### Observaciones Interesantes

1. **Diversidad de predicciones:** A diferencia de los ejemplos anteriores (todos STATIONARY), estos 5 archivos muestran las 3 clases:
   - 1 DOWN
   - 2 STATIONARY
   - 2 UP

2. **Confianza variable:** Rangos de 55% a 94%, mostrando que el modelo tiene diferentes grados de certeza según el patrón de entrada.

3. **Relación Mean vs Predicción:**
   - Mean negativo (-0.59, -0.51): STATIONARY o UP
   - Mean positivo (0.13, 0.32, 0.37): UP, STATIONARY, DOWN
   - No hay correlación directa → el modelo captura patrones temporales complejos

4. **Logits más informativos que probabilidades:**
   - Example 3: logit UP = 2.54 (muy alto) → 93.81% confianza
   - Example 5: logit DOWN = 1.71 (alto) → 86.90% confianza
   - Example 2: logits cercanos a 0 → predicción menos confiante (55%)

---

## 💡 Uso Recomendado

### Para Aprendizaje

```python
# 1. Cargar y explorar
import numpy as np
example = np.load('example_1.npy')

# 2. Visualizar heatmap
import matplotlib.pyplot as plt
plt.imshow(example.T, aspect='auto', cmap='RdYlBu_r')
plt.xlabel('Timestep')
plt.ylabel('Feature')
plt.title('LOB Window - Example 1')
plt.colorbar()
plt.show()

# 3. Examinar evolución temporal
plt.plot(example[:, 0], label='ASK Price Level 1')
plt.plot(example[:, 20], label='BID Price Level 1')
plt.legend()
plt.show()
```

### Para Testing

Usa estos archivos para:
- ✅ Probar pipelines de inferencia
- ✅ Validar resultados esperados
- ✅ Benchmark de velocidad
- ✅ Debugging del modelo

### Para Demostración

Perfectos para mostrar:
- ✅ Diversidad de predicciones
- ✅ Formato de entrada/salida
- ✅ Interpretación de resultados
- ✅ Uso real del modelo

---

## 📚 Referencias

- **Modelo:** TLOB (Transformer with Dual Attention)
- **Dataset:** Bitcoin LOB (2023-01-09 to 2023-01-20)
- **Checkpoint:** `val_loss=0.623_epoch=2.pt` (horizon=10)
- **Paper:** "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction"

---

## 📞 Archivos de Soporte

```
TLOB-main/
├── create_individual_examples.py     # Crear ejemplos individuales
├── inference_single_file.py          # Inferencia sobre 1 archivo
├── run_all_inferences.py             # Procesar todos en lote
│
├── docs/
│   ├── inference_guide.md            # Guía detallada
│   └── RESUMEN_EJECUTIVO.md          # Resumen del proyecto
│
└── data/BTC/individual_examples/     # Este directorio
    ├── example_*.npy                 # 5 ejemplos de entrada
    ├── example_*_result.npy          # 5 resultados
    ├── example_*_result.txt          # 5 resultados (texto)
    ├── summary_all_inferences.txt    # Resumen consolidado
    └── README.md                      # Este documento
```

---

**Última actualización:** 14 Noviembre 2025  
**Autor:** Documentación del proyecto TLOB


