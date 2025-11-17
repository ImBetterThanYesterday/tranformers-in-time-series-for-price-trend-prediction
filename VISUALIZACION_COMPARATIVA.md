# 📊 Visualización Comparativa: Datos Crudos vs Normalizados

**Fecha**: 16 de Noviembre, 2024  
**Feature**: Comparación visual del preprocesamiento  
**Estado**: ✅ Implementado

---

## 🎯 Objetivo

Mostrar al usuario **cómo se transforman los datos** durante el preprocesamiento, comparando:
- **📥 Datos Originales (Crudos)**: Valores reales del mercado BTC
- **✅ Datos Normalizados**: Z-scores listos para el modelo

---

## ✨ Nueva Funcionalidad

### Cuando se Carga un Archivo Crudo

Al cargar un archivo CSV o NPY crudo (por ejemplo `raw_example_1.csv`), Streamlit ahora:

1. **Detecta** que los datos están crudos
2. **Guarda** una copia de los datos originales
3. **Normaliza** los datos automáticamente
4. **Muestra** ambas versiones lado a lado

---

## 📐 Diseño de la Interfaz

### Estructura del TAB "📊 Datos"

```
┌─────────────────────────────────────────────────────────────────┐
│ 🔄 Preprocesamiento Aplicado                                    │
│ Este archivo fue cargado con datos crudos y normalizado         │
│ automáticamente                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────┐  ┌──────────────────────────┐      │
│  │ 📥 Datos Originales    │  │ ✅ Datos Normalizados     │      │
│  │ (Crudos)               │  │                           │      │
│  ├────────────────────────┤  ├──────────────────────────┤      │
│  │ Valores reales BTC     │  │ Z-score: mean≈0, std≈1   │      │
│  │                        │  │                           │      │
│  │ Mean: 8593.41          │  │ Mean: 0.000000            │      │
│  │ Std:  8589.24          │  │ Std:  0.999805            │      │
│  │ Range: 0 ~ 17186       │  │ Range: -1.00 ~ 1.00       │      │
│  │                        │  │                           │      │
│  │ 🔢 Ver primeras 10     │  │ 🔢 Ver primeras 10        │      │
│  │    filas ▼             │  │    filas ▼                │      │
│  │                        │  │                           │      │
│  │ T0  17181.70 17182.20  │  │ T0  0.999716 0.999768     │      │
│  │ T1  17181.70 17182.20  │  │ T1  0.999716 0.999768     │      │
│  │ ...                    │  │ ...                       │      │
│  │                        │  │                           │      │
│  │ Precios en USDT,       │  │ Z-scores normalizados     │      │
│  │ volúmenes en BTC       │  │                           │      │
│  └────────────────────────┘  └──────────────────────────┘      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Métricas generales, heatmap, series temporales, etc...         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Implementación Técnica

### 1. Modificación de `load_data()`

**Antes**:
```python
def load_data(filepath):
    # ...
    return data  # Solo retorna datos normalizados
```

**Después**:
```python
def load_data(filepath):
    # ...
    if is_normalized == False:  # Datos crudos
        data_raw = data.copy()  # Guardar copia
        data_normalized = normalize_raw_data(data)
        return data_normalized, data_raw  # Retornar AMBOS
    else:
        return data, None  # Solo normalizados
```

### 2. Almacenamiento en Session State

```python
st.session_state['data'] = data_normalized       # Para inferencia
st.session_state['data_raw'] = data_raw          # Para visualización
st.session_state['filename'] = selected_name
st.session_state['source'] = example_source
```

### 3. Visualización Comparativa

```python
data_raw = st.session_state.get('data_raw', None)

if data_raw is not None:  # Solo si hay datos crudos
    st.info("🔄 Preprocesamiento Aplicado...")
    
    col_raw, col_norm = st.columns(2)
    
    with col_raw:
        st.markdown("### 📥 Datos Originales (Crudos)")
        st.metric("Mean", f"{data_raw.mean():.2f}")
        # Mostrar primeras filas...
    
    with col_norm:
        st.markdown("### ✅ Datos Normalizados")
        st.metric("Mean", f"{data.mean():.6f}")
        # Mostrar primeras filas...
```

---

## 📊 Casos de Uso

### Caso 1: Cargar Archivo Crudo (CSV)

```
Usuario selecciona: raw_example_1.csv
↓
Sistema detecta: "Datos crudos" (mean=8593)
↓
Sistema guarda: data_raw (copia original)
↓
Sistema normaliza: data (z-score)
↓
Streamlit muestra: AMBAS versiones lado a lado
```

**Visualización**:
- **Izquierda**: Precios reales (17181.70 USDT, 23.37 BTC)
- **Derecha**: Z-scores (0.999716, 0.999768)

---

### Caso 2: Cargar Archivo NPY Crudo

```
Usuario selecciona: raw_example_1.npy
↓
Sistema detecta: "Datos crudos" (mean=8593)
↓
Sistema guarda: data_raw (copia original)
↓
Sistema normaliza: data (z-score)
↓
Streamlit muestra: AMBAS versiones lado a lado
```

**Visualización**: Igual que CSV

---

### Caso 3: Cargar Archivo Ya Normalizado

```
Usuario selecciona: normalized_example_1.npy
↓
Sistema detecta: "Ya normalizado" (mean≈0)
↓
Sistema NO guarda data_raw (None)
↓
Streamlit muestra: Solo datos normalizados (sin comparación)
```

**Visualización**: 
- NO muestra la sección de comparación
- Continúa directo a métricas generales

---

## 🎨 Elementos Visuales

### Métricas Mostradas

#### Datos Crudos
```
Mean:  8593.41    ← Promedio de todos los valores
Std:   8589.24    ← Desviación estándar alta
Range: 0 ~ 17186  ← Rango muy amplio
```

#### Datos Normalizados
```
Mean:  0.000000   ← Centrado en cero
Std:   0.999805   ← Desviación ≈ 1
Range: -1.00 ~ 1.00 ← Rango normalizado
```

### Tablas Expandibles

Cada lado tiene un expander con las primeras 10 filas:

**Datos Crudos**:
```
      F0        F1        F2       F3       ...
T0    17181.70  17182.20  17181.60  17181.00
T1    17181.70  17182.20  17181.60  17181.00
...
```
Caption: "Precios en USDT, volúmenes en BTC"

**Datos Normalizados**:
```
      F0        F1        F2        F3       ...
T0    0.999716  0.999768  0.999716  0.999716
T1    0.999716  0.999768  0.999716  0.999716
...
```
Caption: "Z-scores normalizados"

---

## 🎯 Beneficios para el Usuario

### 1. **Transparencia**
- Ve exactamente qué valores tenía el archivo original
- Entiende qué transformación se aplicó

### 2. **Educativo**
- Aprende qué es la normalización Z-score
- Ve el antes y después en tiempo real

### 3. **Verificación**
- Puede verificar que los valores originales son correctos
- Puede verificar que la normalización fue exitosa (mean≈0, std≈1)

### 4. **Debugging**
- Si algo falla, puede ver los valores crudos
- Puede identificar problemas en los datos originales

---

## 📝 Mensajes al Usuario

### Cuando Hay Preprocesamiento
```
ℹ️ Preprocesamiento Aplicado
Este archivo fue cargado con datos crudos y normalizado automáticamente
```

### En Datos Crudos
```
📥 Datos Originales (Crudos)
Valores reales del mercado BTC
```
- Precios en USDT
- Volúmenes en BTC

### En Datos Normalizados
```
✅ Datos Normalizados
Z-score: mean≈0, std≈1
```
- Z-scores normalizados
- Listos para inferencia

---

## 🔍 Ejemplo Visual

### Al Cargar `raw_example_1.csv`

**Datos Crudos (primera fila)**:
```
sell1:  17181.70 USDT  (ASK price nivel 1)
vsell1: 17182.20 BTC   (ASK volume nivel 1)
buy1:   17181.60 USDT  (BID price nivel 1)
vbuy1:  17181.00 BTC   (BID volume nivel 1)
```

**Datos Normalizados (primera fila)**:
```
sell1:  0.999716  (z-score del ASK price)
vsell1: 0.999768  (z-score del ASK volume)
buy1:   0.999716  (z-score del BID price)
vbuy1:  0.999716  (z-score del BID volume)
```

**Interpretación**:
- Precios ~17181 USDT → z-score ≈ 0.9997
- Volúmenes ~17182 BTC → z-score ≈ 0.9998
- ✅ Normalización correcta: valores originales muy similares → z-scores muy similares

---

## 🚀 Flujo Completo

### 1. Usuario Selecciona Archivo
```
Sidebar → "📄 Crudos (CSV/NPY)" → raw_example_1.csv → 🔄 Cargar
```

### 2. Sistema Procesa
```
1. Lee CSV
2. Elimina timestamp
3. Detecta: datos crudos (mean=8593)
4. Guarda copia: data_raw
5. Normaliza: data_normalized
6. Almacena ambos en session_state
```

### 3. Usuario Ve
```
TAB "📊 Datos":
├─ 🔄 Banner: "Preprocesamiento Aplicado"
├─ Columna Izquierda: Datos Crudos
│  ├─ Mean: 8593.41
│  ├─ Std: 8589.24
│  └─ Tabla con valores reales
├─ Columna Derecha: Datos Normalizados
│  ├─ Mean: 0.000000
│  ├─ Std: 0.999805
│  └─ Tabla con z-scores
└─ Resto de visualizaciones (heatmap, series, etc.)
```

### 4. Usuario Predice
```
TAB "🎯 Predicción":
- Usa data (normalizado) para inferencia
- data_raw solo es para visualización
```

---

## ✅ Checklist de Implementación

- [x] Modificar `load_data()` para retornar tupla
- [x] Guardar `data_raw` en session_state
- [x] Crear visualización comparativa lado a lado
- [x] Mostrar métricas de ambos datasets
- [x] Agregar expanders con primeras filas
- [x] Agregar captions explicativos
- [x] Solo mostrar comparación si hay datos crudos
- [x] Actualizar Docker
- [x] Documentar funcionalidad

---

## 🎓 Para el Usuario Final

### ¿Qué veo cuando cargo un CSV crudo?

1. **Banner azul**: Te avisa que se aplicó preprocesamiento
2. **Dos columnas**:
   - Izquierda: Tus datos originales (precios reales, volúmenes reales)
   - Derecha: Datos transformados (z-scores para el modelo)
3. **Expanders**: Click para ver las primeras 10 filas de cada versión
4. **Resto normal**: Heatmaps, gráficos, predicción usan los datos normalizados

### ¿Por qué es útil?

- **Entiendes** qué está pasando con tus datos
- **Verificas** que los valores originales son correctos
- **Aprendes** cómo funciona la normalización
- **Confías** en el sistema porque ves todo el proceso

---

## 📚 Referencias

- Normalización Z-score: `normalize_raw_data()` en `app.py`
- Detección automática: `is_data_normalized()` en `app.py`
- Procesamiento original: `z_score_orderbook()` en `utils/utils_data.py`

---

**Implementado**: 16 de Noviembre, 2024  
**Versión**: 1.0  
**Estado**: ✅ Funcionando  

