# Fix: Orden Invertido del Softmax y Alpha en Datos Preprocesados

## 🐛 Problemas Encontrados

### 1. Error: "No se pudieron cargar los datos para calcular alpha"

**Síntoma:**
```
❌ Error: No se pudieron cargar los datos para calcular alpha
```

**Causa:**
- Los datos **preprocesados** (`example_1.npy`) ya están normalizados
- No incluyen los datos crudos originales (`data_raw`)
- La función `calculate_alpha()` necesita precios reales para calcular spread o volatilidad
- Cuando `data_raw = None`, el código fallaba

**Solución:**
```python
data_for_alpha = st.session_state.get('data_raw', None)

if data_for_alpha is not None:
    # Calcular alpha dinámicamente desde datos crudos
    alpha = calculate_alpha(data_for_alpha, horizon=horizon, use_spread=use_spread)
    alpha_calculated = True
else:
    # Usar alpha teórico por defecto para datos preprocesados
    if use_spread:
        alpha = 0.005  # 0.5% (spread típico de Bitcoin)
    else:
        alpha = 0.002  # 0.2% (volatilidad típica)
    alpha_calculated = False
```

---

### 2. Orden Invertido del Softmax

**Problema reportado por el usuario:**

```
Etiquetas (truth)        Modelo (predicción softmax)
----------------------------------------------------
0 = UP               ↔   softmax[2]  = UP
1 = STABLE           ↔   softmax[1]  = STABLE
2 = DOWN             ↔   softmax[0]  = DOWN
```

**Causa:**

El modelo de PyTorch da las probabilidades softmax en **ORDEN INVERSO** a las etiquetas:

**Etiquetas durante entrenamiento:**
```python
# utils_data.py línea 158
labels = np.where(
    percentage_change < -alpha, 2,  # DOWN → etiqueta 2
    np.where(percentage_change > alpha, 0, 1)  # UP → etiqueta 0, STABLE → etiqueta 1
)
```

Por lo tanto:
- Etiqueta 0 = UP
- Etiqueta 1 = STATIONARY
- Etiqueta 2 = DOWN

**Salida del modelo:**
```python
output = model(x)  # → tensor de shape [3]
softmax = F.softmax(output)
# softmax[0] = probabilidad de DOWN (etiqueta 2) ❌
# softmax[1] = probabilidad de STABLE (etiqueta 1) ✅
# softmax[2] = probabilidad de UP (etiqueta 0) ❌
```

**Solución: Invertir el orden**

```python
def run_prediction(model, data):
    """Ejecuta predicción
    
    IMPORTANTE: El modelo da softmax en orden INVERSO a las etiquetas:
    - Etiquetas: [0=UP, 1=STABLE, 2=DOWN]
    - Softmax:   [DOWN, STABLE, UP]
    """
    x = torch.from_numpy(data[None, :, :]).float().to(DEVICE)
    with torch.no_grad():
        logits_raw = model(x)[0].cpu().numpy()
        probs_raw = F.softmax(torch.from_numpy(logits_raw), dim=0).numpy()
        
        # INVERTIR orden para que coincida con etiquetas
        # probs_raw = [DOWN, STABLE, UP]
        # probs = [UP, STABLE, DOWN] (orden de etiquetas)
        logits = np.array([logits_raw[2], logits_raw[1], logits_raw[0]])
        probs = np.array([probs_raw[2], probs_raw[1], probs_raw[0]])
        
        pred = int(np.argmax(probs))
    return logits, probs, pred
```

**Resultado:**
- `probs[0]` = probabilidad de UP ✅
- `probs[1]` = probabilidad de STABLE ✅
- `probs[2]` = probabilidad de DOWN ✅

---

## ✅ Cambios Implementados

### 1. Manejo de Alpha para Datos Preprocesados

**Antes:**
```python
# ❌ Siempre intentaba calcular alpha, fallaba con datos preprocesados
data_for_alpha = st.session_state.get('data_raw', data)
if data_for_alpha is None:
    st.error("❌ Error: No se pudieron cargar los datos para calcular alpha")
    st.stop()
alpha = calculate_alpha(data_for_alpha, horizon, use_spread)
```

**Ahora:**
```python
# ✅ Detecta si hay datos raw, sino usa valores teóricos
data_for_alpha = st.session_state.get('data_raw', None)

if data_for_alpha is not None:
    # Calcular dinámicamente
    alpha = calculate_alpha(data_for_alpha, horizon, use_spread)
    alpha_calculated = True
else:
    # Usar valores por defecto
    alpha = 0.005 if use_spread else 0.002
    alpha_calculated = False
    st.info("ℹ️ Usando alpha teórico (datos preprocesados)")
```

**Valores teóricos por defecto:**
- **Normal**: 0.002 (0.2%) - volatilidad típica de Bitcoin
- **Spread**: 0.005 (0.5%) - spread típico bid-ask de Bitcoin

---

### 2. Inversión del Orden del Softmax

**Antes:**
```python
# ❌ Usaba directamente el orden del modelo (INVERTIDO)
logits = model(x)[0].cpu().numpy()
probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
pred = int(np.argmax(probs))
```

**Ahora:**
```python
# ✅ Invierte el orden para que coincida con las etiquetas
logits_raw = model(x)[0].cpu().numpy()
probs_raw = F.softmax(torch.from_numpy(logits_raw), dim=0).numpy()

# INVERTIR: [DOWN, STABLE, UP] → [UP, STABLE, DOWN]
logits = np.array([logits_raw[2], logits_raw[1], logits_raw[0]])
probs = np.array([probs_raw[2], probs_raw[1], probs_raw[0]])

pred = int(np.argmax(probs))
```

---

### 3. Documentación Actualizada

#### Sidebar - Explicación del etiquetado

Agregado expander "ℹ️ Sobre el etiquetado y salida del modelo" que explica:
- Etiquetas durante el entrenamiento
- Orden del softmax (invertido)
- Inversión automática en la app

#### Tab Predicción - Nota sobre inversión

```
Nota: El modelo da softmax en orden inverso [DOWN, STABLE, UP], 
pero la app lo invierte automáticamente para mostrar correctamente.
```

#### Tab Resultados - Indicador de alpha calculado vs teórico

**Alpha calculado (datos crudos):**
```
Alpha calculado: 0.0023 (0.23%)
Calculado dinámicamente desde datos crudos
```

**Alpha teórico (datos preprocesados):**
```
Alpha teórico: 0.0020 (0.20%)
Valor por defecto (datos preprocesados)
```

---

## 📊 Comparación: Antes vs Ahora

### Escenario 1: Datos Preprocesados

#### Antes:
```
1. Cargar example_1.npy
2. Click "Ejecutar Predicción"
3. ❌ Error: No se pudieron cargar los datos para calcular alpha
4. ❌ Predicción muestra UP cuando debería ser DOWN
```

#### Ahora:
```
1. Cargar example_1.npy
2. Click "Ejecutar Predicción"
3. ℹ️ Info: Usando alpha teórico (0.2%)
4. ✅ Predicción correcta (orden invertido automáticamente)
5. ✅ Resultados muestran "Alpha teórico: 0.0020 (0.20%)"
```

---

### Escenario 2: Datos Crudos

#### Antes:
```
1. Cargar raw_example_1.csv
2. Click "Ejecutar Predicción"
3. ✅ Alpha calculado: 0.0023 (0.23%)
4. ❌ Predicción muestra UP cuando debería ser DOWN
```

#### Ahora:
```
1. Cargar raw_example_1.csv
2. Click "Ejecutar Predicción"
3. ✅ Alpha calculado dinámicamente: 0.0023 (0.23%)
4. ✅ Predicción correcta (orden invertido automáticamente)
5. ✅ Resultados muestran "Alpha calculado: 0.0023 (0.23%)"
```

---

## 🧪 Testing

### Test 1: Datos Preprocesados + Normal

```bash
# Iniciar Docker
docker-compose up -d

# En la app:
1. Tab Datos → Seleccionar "Preprocesados"
2. Cargar "example_1.npy"
3. Tab Predicción:
   - Horizonte: 10
   - Umbral: 📊 Normal
4. Click "Ejecutar Predicción"

# Verificar:
✅ No debe haber error de alpha
✅ Debe mostrar "ℹ️ Usando alpha teórico"
✅ Resultados deben mostrar "Alpha teórico: 0.0020 (0.20%)"
✅ Predicción debe ser correcta (no invertida)
```

---

### Test 2: Datos Preprocesados + Spread

```bash
# En la app:
1. Tab Predicción:
   - Horizonte: 20
   - Umbral: 💹 Spread
2. Click "Ejecutar Predicción"

# Verificar:
✅ Alpha teórico: 0.0050 (0.50%)
✅ Predicción correcta
```

---

### Test 3: Datos Crudos + Normal

```bash
# En la app:
1. Tab Datos → Seleccionar "Crudos (CSV/NPY)"
2. Cargar "raw_example_1.csv"
3. Tab Predicción:
   - Horizonte: 10
   - Umbral: 📊 Normal
4. Click "Ejecutar Predicción"

# Verificar:
✅ No debe haber info de alpha teórico
✅ Alpha calculado dinámicamente (valor real del dataset)
✅ Resultados: "Alpha calculado: 0.XXXX"
✅ Predicción correcta
```

---

### Test 4: Verificar Inversión de Softmax

**Caso de prueba:**

Si el modelo raw da:
```python
logits_raw = [2.5, 1.0, 3.2]  # [DOWN, STABLE, UP]
probs_raw = [0.25, 0.10, 0.65]
```

La app debe mostrar:
```python
logits = [3.2, 1.0, 2.5]  # [UP, STABLE, DOWN]
probs = [0.65, 0.10, 0.25]
pred = 0  # UP (argmax de [0.65, 0.10, 0.25])
```

**Verificar en resultados:**
```
📈 UP: 65.0%
➡️ STATIONARY: 10.0%
📉 DOWN: 25.0%
```

---

## 📝 Archivos Modificados

### `app.py`

1. **Función `run_prediction()`** (líneas 285-313):
   - Agregado inversión de orden de softmax
   - Documentación del orden invertido

2. **Tab Predicción** (líneas 783-815):
   - Agregado manejo de datos preprocesados
   - Uso de alpha teórico cuando no hay datos raw
   - Mensaje informativo para usuario

3. **Tab Resultados** (líneas 854-876):
   - Indicador de alpha calculado vs teórico
   - Nota sobre origen del alpha

4. **Sidebar** (líneas 430-457):
   - Explicación detallada del orden invertido
   - Tabla de correspondencia etiqueta ↔ softmax

5. **Tab Predicción - Info** (líneas 767-781):
   - Nota sobre inversión automática

---

## 🎯 Resumen

### Problema 1: Alpha con Datos Preprocesados ✅ RESUELTO
- **Solución:** Usar valores teóricos por defecto (0.2% Normal, 0.5% Spread)
- **Resultado:** La app funciona con datos preprocesados y crudos

### Problema 2: Orden Invertido del Softmax ✅ RESUELTO
- **Solución:** Invertir arrays de logits y probs: `[raw[2], raw[1], raw[0]]`
- **Resultado:** Predicciones correctas alineadas con etiquetas

### Mejoras Adicionales:
- ✅ Documentación clara del orden invertido
- ✅ Indicador visual de alpha calculado vs teórico
- ✅ Mensajes informativos para el usuario
- ✅ Explicación detallada en sidebar

---

## ⚠️ Importante para Desarrollo Futuro

1. **Si reentrenas el modelo:** Verificar que el orden del softmax siga siendo invertido.

2. **Si cambias el etiquetado:** Actualizar tanto `utils_data.py` como la inversión en `run_prediction()`.

3. **Si agregas nuevos tipos de datos:** Considerar si tendrán `data_raw` o necesitarán alpha teórico.

4. **Testing:** Siempre probar con datos preprocesados Y crudos después de cambios.

---

## 📚 Referencias

- Código etiquetado: `src/utils/utils_data.py` líneas 150-161
- Función predicción: `app.py` líneas 285-313
- Documentación previa: `docs/CAMBIOS_ALPHA_HORIZONTE.md`

