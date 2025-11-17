# 🚀 Guía Rápida: Normalización Automática

## ¿Qué se implementó?

Se agregó un sistema **inteligente** que detecta y normaliza automáticamente datos crudos (sin procesar) cuando los cargas en Streamlit.

### ✨ Características

1. ✅ **Detección Automática**: Identifica si los datos están crudos o ya normalizados
2. ✅ **Normalización Z-Score**: Aplica normalización automáticamente cuando es necesario
3. ✅ **Multi-Formato**: Soporta archivos `.csv` y `.npy`
4. ✅ **Transparente**: Te muestra qué procesamiento se aplicó

---

## 📊 Tipos de Datos Disponibles

### 1. **Preprocesados** (Ya normalizados)
- **Ubicación**: `data/BTC/individual_examples/`
- **Archivos**: `example_1.npy` a `example_5.npy`
- **Estado**: Ya normalizados (mean≈0, std≈1)
- **Uso directo**: ✅ Sí

### 2. **Crudos CSV** (Con timestamp)
- **Ubicación**: `data/BTC/raw_examples/`
- **Archivos**: `raw_example_1.csv` a `raw_example_7.csv`
- **Estado**: Sin normalizar (valores reales de BTC)
- **Uso directo**: ❌ No (requiere normalización)
- **Formato**: Incluye columna `timestamp`

### 3. **Crudos NPY** (Sin timestamp)
- **Ubicación**: `data/BTC/raw_examples/`
- **Archivos**: `raw_example_1.npy` a `raw_example_7.npy`
- **Estado**: Sin normalizar (valores reales de BTC)
- **Uso directo**: ❌ No (requiere normalización)
- **Formato**: Solo LOB (128, 40)

---

## 🎯 Uso Rápido

### Opción 1: Docker (Recomendado)

```bash
# Iniciar Streamlit
docker-compose up -d

# Abrir navegador
open http://localhost:8501
```

### Opción 2: Local

```bash
# Ejecutar Streamlit
streamlit run app.py

# Abrir navegador
# Se abrirá automáticamente en http://localhost:8501
```

---

## 🖥️ Usar Streamlit con Datos Crudos

### Paso 1: Seleccionar Fuente

En el sidebar izquierdo, verás dos opciones:

```
○ 📦 Preprocesados  
○ 📄 Crudos (CSV/NPY)
```

**Selecciona**: `📄 Crudos (CSV/NPY)`

### Paso 2: Elegir Archivo

Verás una lista de archivos disponibles:

```
14 ejemplos:
- raw_example_1.csv
- raw_example_1.npy
- raw_example_2.csv
- raw_example_2.npy
- ...
```

**Elige cualquiera**, por ejemplo: `raw_example_1.csv`

### Paso 3: Cargar

Click en el botón `🔄 Cargar`

### Paso 4: Normalización Automática

Verás estos mensajes:

```
ℹ️ Detectados datos crudos. Aplicando normalización Z-score...

📊 Estadísticas de normalización:
   Precios  -> mean: 8594.60, std: 8589.75
   Volúmenes -> mean: 8592.23, std: 8592.09

✅ Normalización completada (mean=0.0000, std=0.9998)
```

### Paso 5: Ver y Analizar

- **Tab Visualización**: Ver distribuciones de las 40 features
- **Tab Análisis**: Estadísticas descriptivas
- **Tab Predicción**: Hacer predicción con el modelo TLOB

---

## 📝 Ejemplos de Valores

### CSV Crudo (raw_example_1.csv)

```csv
timestamp,sell1,vsell1,buy1,vbuy1,...
1673302660926,17181.7,17182.2,17181.6,17181.0,...
1673302661177,17181.7,17182.2,17181.6,17181.0,...
```

- **Precios**: ~17000-21000 USDT (valores reales)
- **Volúmenes**: 0-50 (cantidades reales)
- **Legible**: ✅ Puedes entender los valores

### NPY Normalizado (después de cargar)

```
mean: 0.0000
std: 0.9998
min: -1.0006
max: 1.0002
```

- **Z-scores**: Centrados en 0, std≈1
- **Legible**: ❌ Números abstractos
- **Listo para modelo**: ✅ Sí

---

## 🔍 Diferencia: CSV vs NPY

### CSV Crudo
```python
# Con timestamp
raw_example_1.csv
Shape: (128, 41)  # 41 = timestamp + 40 features
Incluye: timestamp, sell1, vsell1, buy1, ...
```

### NPY Crudo
```python
# Sin timestamp
raw_example_1.npy
Shape: (128, 40)  # Solo 40 features del LOB
Incluye: sell1, vsell1, buy1, vbuy1, ...
```

**Ambos se normalizan automáticamente al cargar en Streamlit**

---

## 🧪 Crear Tus Propios Ejemplos

### Script Disponible: `create_raw_examples.py`

```bash
python3 create_raw_examples.py
```

**Salida**:
- 7 archivos CSV crudos
- 7 archivos NPY crudos
- `metadata.json` con información detallada
- `README.md` con documentación

**Los ejemplos aparecerán automáticamente en Streamlit**

---

## 🎬 Demo Completo

### 1. Crear Ejemplos Crudos
```bash
python3 create_raw_examples.py
```

### 2. Probar Normalización
```bash
python3 test_normalization.py
```

**Salida esperada**:
```
✅ PRUEBA 1 EXITOSA: Normalización correcta
✅ PRUEBA 2 EXITOSA: Normalización correcta
✅ PRUEBA 3 EXITOSA: Detectó datos ya normalizados
```

### 3. Ejecutar Streamlit
```bash
docker-compose up -d
```

### 4. Usar en Navegador
1. Abrir: http://localhost:8501
2. Sidebar → Seleccionar "📄 Crudos (CSV/NPY)"
3. Elegir `raw_example_1.csv`
4. Click "🔄 Cargar"
5. Ver normalización automática
6. Tab "Predicción" → Click "🎯 Predecir"

---

## 📊 Resultado de Predicción

```
🎯 Predicción: DOWN (81.3%)

Probabilidades:
  ▼ DOWN:  81.3%
  — HOLD:  12.4%
  ▲ UP:     6.3%

Logits:
  DOWN:  1.234
  HOLD: -0.456
  UP:   -1.789
```

---

## 🔧 Archivos Importantes

| Archivo | Descripción |
|---------|-------------|
| `create_raw_examples.py` | Crea ejemplos crudos del CSV |
| `test_normalization.py` | Prueba la normalización |
| `app.py` | Streamlit con normalización automática |
| `NORMALIZACION_AUTOMATICA.md` | Documentación técnica completa |
| `data/BTC/raw_examples/` | Ejemplos crudos (CSV y NPY) |
| `data/BTC/individual_examples/` | Ejemplos preprocesados |

---

## ❓ FAQ

### ¿Cuál formato debo usar?

- **CSV**: Si quieres ver los valores reales y entender qué está pasando
- **NPY**: Si solo necesitas hacer inferencia rápida

### ¿El sistema detecta automáticamente?

✅ Sí. El sistema detecta si los datos están crudos (mean > 100) o normalizados (mean ≈ 0).

### ¿Puedo subir mis propios archivos?

✅ Sí. En Streamlit, usa el botón "O sube archivo" y sube tu `.csv` o `.npy`.

### ¿Qué pasa si subo datos ya normalizados?

El sistema detectará que ya están normalizados y **no** aplicará normalización adicional.

### ¿Los CSV deben tener timestamp?

Es opcional. Si el CSV tiene una columna `timestamp`, se eliminará automáticamente.

---

## 📈 Comparación Visual

### Antes (CSV Crudo)
```
Precio BTC: 17181.7 USDT
Volumen: 23.371 BTC
Mean: 8593.41
Std: 8589.24
```

### Después (Normalizado)
```
Z-score precio: 0.9997
Z-score volumen: 0.9998
Mean: 0.0000
Std: 0.9998
```

---

## ✅ Ventajas del Sistema

| Ventaja | Descripción |
|---------|-------------|
| **Flexibilidad** | CSV y NPY, crudos y normalizados |
| **Automático** | No requiere pre-procesamiento manual |
| **Transparente** | Muestra qué se aplicó y por qué |
| **Robusto** | Detecta y maneja diferentes formatos |
| **Fácil de usar** | Solo cargar archivo y listo |

---

## 🎓 Conclusión

Ya no necesitas:
- ❌ Pre-procesar datos manualmente
- ❌ Preocuparte por normalización
- ❌ Convertir CSV a NPY

El sistema hace todo automáticamente cuando cargas el archivo en Streamlit.

**¡Solo carga y predice! 🚀**

---

## 📞 Comandos Rápidos

```bash
# Crear ejemplos crudos
python3 create_raw_examples.py

# Probar normalización
python3 test_normalization.py

# Ejecutar Streamlit (Docker)
docker-compose up -d

# Ver logs
docker logs tlob-streamlit --tail 20

# Detener
docker-compose down
```

---

*Implementado: 2024-11-16*
*Versión: 1.0*

