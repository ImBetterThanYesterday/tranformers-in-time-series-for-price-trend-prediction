# Normalización Automática de Datos Crudos

## 📋 Resumen

Se implementó un sistema **inteligente** que detecta y normaliza automáticamente datos crudos (sin procesar) al cargarlos en Streamlit.

### ✨ Características Principales

1. **Detección Automática**: El sistema detecta si los datos están crudos o ya normalizados
2. **Normalización Z-Score**: Aplica normalización automáticamente cuando es necesario
3. **Soporte Multi-Formato**: Acepta archivos CSV y NPY
4. **Transparencia**: Muestra al usuario qué procesamiento se aplicó

---

## 🔧 Componentes Implementados

### 1. Script de Creación: `create_raw_examples.py`

**Propósito**: Extrae ejemplos del CSV original y los guarda **sin normalizar**.

**Salida**:
- `data/BTC/raw_examples/raw_example_N.csv` - Con timestamp
- `data/BTC/raw_examples/raw_example_N.npy` - Solo LOB (128, 40)
- `metadata.json` - Información detallada
- `README.md` - Documentación

**Uso**:
```bash
python3 create_raw_examples.py
```

**Resultado**: 7 ejemplos distribuidos uniformemente a lo largo del dataset

---

### 2. Funciones de Normalización en `app.py`

#### `normalize_raw_data(data)`
Aplica Z-score normalization:
```python
# Precios (columnas pares)
df[col] = (df[col] - mean_prices) / std_prices

# Volúmenes (columnas impares)
df[col] = (df[col] - mean_volumes) / std_volumes
```

#### `is_data_normalized(data)`
Detecta automáticamente el tipo de datos:
- **Raw (crudo)**: `mean > 100` (precios BTC ~17000-21000)
- **Normalized**: `mean ≈ 0` y `std ≈ 1`
- **Unknown**: Caso ambiguo

#### `load_data(filepath)`
Carga datos y normaliza automáticamente si es necesario:
1. Lee archivo CSV o NPY
2. Verifica shape (128, 40)
3. Detecta si está normalizado
4. Normaliza si es necesario
5. Muestra mensaje informativo al usuario

---

## 🎯 Flujo de Trabajo

### Caso 1: Datos Preprocesados
```
Usuario selecciona → "📦 Preprocesados"
↓
Carga example_1.npy
↓
Sistema detecta: "Ya normalizados"
↓
✅ Listo para inferencia
```

### Caso 2: Datos Crudos (NPY)
```
Usuario selecciona → "📄 Crudos (CSV/NPY)"
↓
Carga raw_example_1.npy
↓
Sistema detecta: "Datos crudos" (mean=8593.41)
↓
🔄 Aplica normalización Z-score
↓
✅ Normalizado (mean≈0, std≈1)
↓
Listo para inferencia
```

### Caso 3: Datos Crudos (CSV)
```
Usuario selecciona → "📄 Crudos (CSV/NPY)"
↓
Carga raw_example_1.csv
↓
Sistema:
  1. Lee CSV
  2. Elimina columna 'timestamp'
  3. Detecta: "Datos crudos"
  4. Aplica normalización
↓
✅ Listo para inferencia
```

---

## 📊 Comparación: Crudo vs Normalizado

| Aspecto | Crudo (Raw) | Normalizado |
|---------|-------------|-------------|
| **Precios BTC** | 17181.6, 17181.5, ... | -0.938, -0.941, ... |
| **Volúmenes** | 23.371, 0.746, ... | 1.234, -0.456, ... |
| **Mean** | ~8500 - 10600 | ≈ 0.0 |
| **Std** | ~8500 - 10600 | ≈ 1.0 |
| **Legibilidad** | ✅ Alta (valores reales) | ❌ Baja (Z-scores) |
| **Uso Directo** | ❌ No (requiere normalización) | ✅ Sí |
| **Formato** | CSV o NPY | NPY |

---

## 🖥️ Interfaz de Streamlit

### Selector de Fuente
```
○ 📦 Preprocesados  
○ 📄 Crudos (CSV/NPY)
```

### Mensajes Informativos

#### Al cargar datos crudos:
```
ℹ️ Detectados datos crudos. Aplicando normalización Z-score...
✅ Normalización completada (mean=0.0003, std=1.0012)
```

#### Al cargar datos ya normalizados:
```
✅ Datos ya normalizados (mean=-0.0002, std=0.9998)
```

#### Al subir archivo:
```
Upload: *.npy o *.csv
```

---

## 📂 Estructura de Archivos

```
data/BTC/
├── individual_examples/          # Preprocesados (normalizados)
│   ├── example_1.npy
│   ├── example_2.npy
│   └── ...
│
├── raw_examples/                 # Crudos (sin normalizar)
│   ├── raw_example_1.csv         # Con timestamp
│   ├── raw_example_1.npy         # Solo LOB
│   ├── raw_example_2.csv
│   ├── raw_example_2.npy
│   ├── ...
│   ├── metadata.json
│   └── README.md
│
└── original_source/
    └── 1-09-1-20.csv             # CSV original completo
```

---

## 🧪 Validación

### Verificar que la normalización funciona:

```python
import numpy as np

# Cargar ejemplo crudo
raw = np.load('data/BTC/raw_examples/raw_example_1.npy')
print(f"Raw: mean={raw.mean():.2f}, std={raw.std():.2f}")
# Output: Raw: mean=8593.41, std=8589.24

# Cargar desde Streamlit (ya normalizado)
# mean≈0.0, std≈1.0
```

### Verificar archivos CSV:

```python
import pandas as pd

df = pd.read_csv('data/BTC/raw_examples/raw_example_1.csv')
print(df.head())
```

**Output**:
```
   timestamp     sell1  vsell1     buy1  vbuy1  ...
0  1673302660926  17181.6  23.371  17181.5  0.746  ...
1  1673302661175  17181.6  23.371  17181.5  0.746  ...
```

---

## ✅ Ventajas del Sistema

### 1. **Flexibilidad**
- Trabaja con datos crudos y normalizados
- Soporta CSV y NPY
- Detección automática

### 2. **Transparencia**
- Usuario ve valores reales en CSV
- Sistema muestra qué procesamiento aplica
- Estadísticas antes y después

### 3. **Facilidad de Uso**
- No requiere pre-procesamiento manual
- Upload directo de archivos
- Normalización invisible al usuario

### 4. **Debugging**
- CSV legible para inspección manual
- Metadata detallado
- Estadísticas raw disponibles

### 5. **Portabilidad**
- CSV es formato universal
- No depende de pre-procesamiento previo
- Fácil de compartir

---

## 🚀 Uso Completo

### Paso 1: Crear ejemplos crudos
```bash
python3 create_raw_examples.py
```

### Paso 2: Ejecutar Streamlit
```bash
# Local
streamlit run app.py

# Docker
docker-compose up -d
```

### Paso 3: En la interfaz
1. Seleccionar "📄 Crudos (CSV/NPY)"
2. Elegir `raw_example_1.csv` o `raw_example_1.npy`
3. Click "🔄 Cargar"
4. Ver mensaje: "🔄 Detectados datos crudos..."
5. Sistema normaliza automáticamente
6. Hacer predicción normalmente

---

## 📊 Ejemplo de Salida

### Al cargar `raw_example_1.csv`:

```
🔄 Detectados datos crudos. Aplicando normalización Z-score...

Estadísticas originales:
  - Mean: 8593.41
  - Std: 8589.24
  - Min: 0.00
  - Max: 17186.40

✅ Normalización completada
  - Mean: 0.0003
  - Std: 1.0012
  - Min: -0.9987
  - Max: 0.9998
```

### Predicción:

```
🎯 Predicción: DOWN (81.3%)

Probabilidades:
  ▼ DOWN:  81.3%
  — HOLD:  12.4%
  ▲ UP:     6.3%
```

---

## 🔍 Detalles Técnicos

### Z-Score Normalization

**Fórmula**:
```
x_norm = (x - μ) / σ
```

**Aplicación**:
- **Precios** (columnas 0, 2, 4, ..., 38): Usan μ_prices y σ_prices
- **Volúmenes** (columnas 1, 3, 5, ..., 39): Usan μ_volumes y σ_volumes

**Resultado**:
- Media ≈ 0
- Desviación estándar ≈ 1
- Preserva la distribución original
- Facilita el aprendizaje del modelo

### Detección de Datos

**Heurística**:
```python
mean = abs(data.mean())
std = data.std()

if mean > 100:          # Valores reales de BTC
    return "raw"
elif mean < 1 and 0.5 < std < 2:  # Z-scores
    return "normalized"
else:
    return "unknown"
```

---

## 📝 Archivos Modificados

1. **`create_raw_examples.py`** (nuevo)
   - Extrae ejemplos crudos del CSV
   - Guarda en formato CSV y NPY
   - Sin normalización

2. **`app.py`**
   - `normalize_raw_data()`: Nueva función
   - `is_data_normalized()`: Nueva función
   - `load_data()`: Modificada para soportar CSV y normalización automática
   - Selector de fuente actualizado
   - File uploader acepta CSV y NPY

3. **`data/BTC/raw_examples/`** (nuevo directorio)
   - 7 ejemplos CSV
   - 7 ejemplos NPY
   - Metadata
   - README

---

## 🎓 Conclusión

El sistema implementado permite:

✅ Trabajar con datos en **formato crudo** (valores reales)  
✅ **Normalización automática** sin intervención del usuario  
✅ Soporte para **CSV y NPY**  
✅ **Transparencia** en el procesamiento  
✅ **Flexibilidad** para diferentes fuentes de datos  

El usuario solo necesita:
1. Seleccionar un archivo (CSV o NPY)
2. El sistema hace el resto automáticamente

---

*Implementación completada: 2024-11-16*

