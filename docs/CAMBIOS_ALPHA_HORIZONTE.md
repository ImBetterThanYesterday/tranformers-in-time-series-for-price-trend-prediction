# Cambios: Horizonte Dinámico y Selector de Umbral Alpha

## 🎯 Resumen de Cambios

### ✅ Problema 1: Orden de Clases CORREGIDO

**Problema anterior:**
```python
# ❌ INCORRECTO
CLASSES = {0: "DOWN 📉", 1: "STATIONARY ➡️", 2: "UP 📈"}
```

**Solución:**
```python
# ✅ CORRECTO (según utils_data.py línea 158)
CLASSES = {0: "UP 📈", 1: "STATIONARY ➡️", 2: "DOWN 📉"}
COLORS = {0: "#10b981", 1: "#3b82f6", 2: "#ef4444"}
```

**Explicación:**

El etiquetado en `utils_data.py`:
```python
labels = np.where(
    percentage_change < -alpha, 2,  # Baja mucho → DOWN (clase 2)
    np.where(percentage_change > alpha, 0, 1)  # Sube mucho → UP (clase 0), resto → STATIONARY (clase 1)
)
```

Por lo tanto:
- **Clase 0**: UP 📈 (cambio > +alpha)
- **Clase 1**: STATIONARY ➡️ (cambio dentro de ±alpha)
- **Clase 2**: DOWN 📉 (cambio < -alpha)

El modelo de PyTorch da salidas softmax en el **mismo orden**: `[prob_up, prob_stationary, prob_down]`

---

### ✅ Problema 2: TypeError 'NoneType' object is not subscriptable - RESUELTO

**Problema:**
```python
# ❌ Error cuando data no está cargado
data_for_alpha = st.session_state.get('data_raw', data)
alpha = calculate_alpha(data_for_alpha, ...)  # TypeError si data_for_alpha es None
```

**Solución:**
```python
# ✅ Validación antes de calcular alpha
if 'data' not in st.session_state:
    st.error("⚠️ Primero debes cargar datos en la pestaña 'Datos'")
    st.stop()

data = st.session_state['data']
data_for_alpha = st.session_state.get('data_raw', data)

if data_for_alpha is None:
    st.error("❌ Error: No se pudieron cargar los datos para calcular alpha")
    st.stop()
```

---

### ✅ Problema 3: Selectores movidos a Tab de Predicción

**Antes:** Los selectores estaban en el sidebar (configuración global)

**Ahora:** Los selectores están en la **Tab de Predicción** (configuración por predicción)

```python
# TAB 3: Predicción
with tab3:
    st.header("🎯 Realizar Predicción")
    
    # ============ CONFIGURACIÓN DE PREDICCIÓN ============
    st.subheader("⚙️ Parámetros de Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selector de horizonte
        horizon = st.selectbox(
            "**Horizonte de Predicción:**",
            options=[10, 20, 50, 100],
            index=0
        )
    
    with col2:
        # Selector de tipo de umbral
        threshold_type = st.radio(
            "**Tipo de Umbral (Alpha):**",
            options=["📊 Normal", "💹 Spread"],
            index=0
        )
```

---

## 📊 Funcionalidad: Selector de Horizonte

### Checkpoints Disponibles

```python
CHECKPOINTS = {
    10: "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_42/pt/val_loss=0.624_epoch=2.pt",
    20: "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_20_seed_42/pt/val_loss=0.822_epoch=1.pt",
    50: "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_50_seed_42/pt/val_loss=0.962_epoch=0.pt",
    100: "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_100_seed_42/pt/val_loss=1.013_epoch=0.pt"
}
```

### Carga Dinámica de Modelos

```python
def get_model(horizon=10):
    model_key = f'tlob_model_h{horizon}'
    
    if model_key not in st.session_state or st.session_state.get('current_horizon') != horizon:
        # Cargar checkpoint correspondiente al horizonte
        checkpoint_path = CHECKPOINTS[horizon]
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        # ... cargar modelo ...
        st.session_state[model_key] = model
        st.session_state['current_horizon'] = horizon
```

**Ventaja:** Cada horizonte tiene su propio modelo entrenado específicamente para esa ventana temporal.

---

## 💹 Funcionalidad: Selector de Umbral Alpha

### Tipos de Umbral

#### 📊 Normal (Por defecto)
```python
# Basado en cambio porcentual promedio
alpha = np.abs(percentage_change).mean() / 2
```

**Uso:** 
- Volatilidad natural del activo
- **Usado durante el entrenamiento del modelo**
- Predicción teórica de tendencias

**Ejemplo:** Si alpha = 0.002 (0.2%)
- Cambio > +0.2% → UP
- Cambio < -0.2% → DOWN
- Cambio dentro de ±0.2% → STATIONARY

---

#### 💹 Spread
```python
# Basado en spread promedio (costos de transacción)
spread = ask_prices - bid_prices
avg_mid_price = mid_prices.mean()
alpha = spread.mean() / avg_mid_price
```

**Uso:**
- Costos de transacción reales
- Evaluación de rentabilidad en trading real
- Análisis más restrictivo

**Ejemplo:** Si spread = 0.005 (0.5%)
- Un cambio de +0.3% predicho como UP no sería rentable (< spread)
- Solo cambios > 0.5% serían rentables después de costos

---

### Implementación: calculate_alpha()

```python
def calculate_alpha(data, horizon=10, use_spread=False, len_smooth=5):
    """
    Calcula el umbral alpha para clasificación de tendencias
    
    Args:
        data: numpy array con datos LOB (shape: seq_len, num_features)
        horizon: horizonte de predicción
        use_spread: Si True, usa spread; si False, usa cambio porcentual
        len_smooth: longitud de ventana para suavizado
        
    Returns:
        alpha: umbral calculado
    """
    # Extraer precios ask (columna 0) y bid (columna 2)
    ask_prices = data[:, 0]
    bid_prices = data[:, 2]
    
    # Calcular mid-price
    mid_prices = (ask_prices + bid_prices) / 2
    
    if use_spread:
        # Alpha basado en spread promedio (como porcentaje del mid-price)
        spread = ask_prices - bid_prices
        avg_mid_price = mid_prices.mean()
        alpha = (spread.mean() / avg_mid_price) if avg_mid_price != 0 else 0.0
    else:
        # Alpha basado en cambio porcentual promedio
        if len(mid_prices) > horizon + len_smooth:
            previous_prices = mid_prices[:-horizon]
            future_prices = mid_prices[horizon:]
            percentage_change = (future_prices - previous_prices) / previous_prices
            alpha = np.abs(percentage_change).mean() / 2
        else:
            alpha = 0.002  # 0.2% por defecto
    
    return alpha
```

---

## 🧪 Ejemplo de Uso

### Escenario 1: Predicción Teórica (Normal)

1. Usuario selecciona:
   - Horizonte: **20 timesteps**
   - Umbral: **📊 Normal**

2. Sistema calcula:
   ```
   alpha = 0.0018 (0.18%)
   ```

3. Interpretación:
   - Cambios > +0.18% → Predicción UP
   - Cambios < -0.18% → Predicción DOWN
   - Cambios dentro → Predicción STATIONARY

---

### Escenario 2: Análisis de Trading Real (Spread)

1. Usuario selecciona:
   - Horizonte: **20 timesteps**
   - Umbral: **💹 Spread**

2. Sistema calcula:
   ```
   alpha = 0.0045 (0.45%)
   ```

3. Interpretación:
   - Cambios > +0.45% → Potencialmente rentable (UP después de costos)
   - Cambios < -0.45% → Potencialmente rentable (DOWN después de costos)
   - Cambios dentro de ±0.45% → NO rentable (STATIONARY, costos > ganancia)

4. **Resultado mostrado:**
   ```
   Configuración de la predicción:
   - Horizonte: 20 timesteps
   - Tipo de umbral: Spread
   - Alpha calculado: 0.0045 (0.45%)
   
   Los cambios de precio menores a ±0.45% se consideran STATIONARY.
   ```

---

## 📚 Referencia: Paper TLOB

Del paper original ([GitHub](https://github.com/LeonardoBerti00/TLOB)):

> "Predictability must be considered in relation to transaction costs. We experimented with defining trends using an average spread, reflecting the primary transaction cost. The resulting performance deterioration underscores the complexity of translating trend classification into profitable trading strategies."

**Implicación:**
- Los modelos pueden predecir tendencias con alta precisión
- Pero **no todas las tendencias son rentables** después de costos de transacción
- El umbral basado en spread simula condiciones de trading real

---

## 🎨 Cambios en la Interfaz

### Antes:
```
Sidebar:
├── Configuración
│   ├── [Horizonte: 10, 20, 50, 100]  ← Global
│   └── [Umbral: Normal, Spread]      ← Global
└── Cargar Datos
```

### Ahora:
```
Sidebar:
├── Configuración
│   └── Info del modelo
└── Cargar Datos

Tab Predicción:
├── Parámetros de Predicción
│   ├── [Horizonte: 10, 20, 50, 100]  ← Por predicción
│   └── [Umbral: Normal, Spread]      ← Por predicción
├── Explicación del etiquetado
└── Botón "Ejecutar Predicción"
```

**Ventaja:** Cada predicción puede tener su propia configuración sin afectar predicciones anteriores.

---

## ✅ Resumen de Archivos Modificados

1. **`app.py`**:
   - Corregido `CLASSES` y `COLORS`
   - Agregado `calculate_alpha()`
   - Movido selectores a Tab Predicción
   - Agregado validación de datos antes de calcular alpha
   - Actualizado orden de métricas en resultados

---

## 🚀 Testing

### Test 1: Horizonte 10 con Umbral Normal
```bash
docker-compose up -d
# Ir a http://localhost:8501
# Cargar example_1.npy
# Seleccionar Horizonte: 10, Umbral: Normal
# Ejecutar Predicción
# Verificar: Alpha ~0.001-0.003
```

### Test 2: Horizonte 100 con Umbral Spread
```bash
# Seleccionar Horizonte: 100, Umbral: Spread
# Ejecutar Predicción
# Verificar: Alpha ~0.004-0.008 (mayor por spread)
```

### Test 3: Cambio de Horizonte
```bash
# Predicción 1: Horizonte 10
# Predicción 2: Horizonte 50
# Verificar: Mensaje "Modelo cargado (horizonte 50 timesteps)"
# Verificar: Resultados diferentes debido a diferente modelo
```

---

## 📝 Notas Importantes

1. **Modelo fue entrenado con umbral Normal**: El umbral Spread es solo para análisis post-predicción, no afecta los pesos del modelo.

2. **Orden de clases es crítico**: Asegurarse de que `CLASSES = {0: "UP", 1: "STATIONARY", 2: "DOWN"}` coincida con el etiquetado en `utils_data.py`.

3. **Alpha se calcula dinámicamente**: No es un valor fijo, depende de los datos del ejemplo y la configuración elegida.

4. **Checkpoints pre-entrenados**: Cada horizonte tiene su propio checkpoint, optimizado para esa ventana temporal específica.

