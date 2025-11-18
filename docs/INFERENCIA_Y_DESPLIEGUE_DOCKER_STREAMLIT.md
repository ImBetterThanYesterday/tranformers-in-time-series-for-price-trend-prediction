# Inferencia y Despliegue: Guía Completa

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Flujo Completo de Inferencia](#2-flujo-completo-de-inferencia)
3. [Preprocesamiento de Datos](#3-preprocesamiento-de-datos)
4. [Despliegue con Docker](#4-despliegue-con-docker)
5. [Aplicación Streamlit](#5-aplicación-streamlit)
6. [Casos de Uso](#6-casos-de-uso)
7. [Troubleshooting](#7-troubleshooting)
8. [Referencias](#8-referencias)

---

## 1. Introducción

Este documento explica el proceso completo de inferencia del modelo TLOB, desde la carga de datos hasta la visualización de predicciones en Streamlit. El despliegue se realiza mediante Docker para garantizar portabilidad y facilidad de uso.

### Componentes del Sistema

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Datos LOB      │ --> │  Modelo TLOB    │ --> │  Streamlit UI   │
│  (CSV/NPY)      │     │  (PyTorch)      │     │  (Visualización)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 2. Flujo Completo de Inferencia

### 2.1 Diagrama de Flujo Detallado

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          INICIO: Cargar Archivo                            │
│                       (CSV o NPY desde Streamlit)                          │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │ Detectar Formato │
                    │ y Tipo de Datos  │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼──────────┐      ┌──────────▼──────────┐
     │  Archivo .CSV     │      │   Archivo .NPY      │
     │  (siempre crudo)  │      │  (crudo o normal.)  │
     └────────┬──────────┘      └──────────┬──────────┘
              │                             │
              │                    ┌────────▼────────┐
              │                    │ Detectar si está│
              │                    │  Normalizado    │
              │                    │  (mean ≈ 0?)    │
              │                    └────────┬────────┘
              │                             │
              │                 ┌───────────┴───────────┐
              │                 │                       │
              │          ┌──────▼──────┐      ┌────────▼────────┐
              │          │  Ya Normal  │      │   Datos Crudos  │
              │          │  (mean≈0,   │      │  (mean>>1000)   │
              │          │   std≈1)    │      │                 │
              │          └──────┬──────┘      └────────┬────────┘
              │                 │                      │
              └─────────────────┴──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  ¿Necesita            │
                    │  Normalización?       │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
           ┌────────▼────────┐     ┌───────▼──────────┐
           │   SÍ (Crudo)    │     │   NO (Normalizado)│
           │                 │     │                   │
           │ Aplicar Z-score │     │ Usar tal cual     │
           │ Normalization   │     │                   │
           └────────┬────────┘     └───────┬──────────┘
                    │                      │
                    │    ┌─────────────────┘
                    │    │
         ┌──────────▼────▼──────────┐
         │ Datos Normalizados        │
         │ Shape: (128, 40)          │
         │ Mean ≈ 0, Std ≈ 1        │
         └──────────┬────────────────┘
                    │
         ┌──────────▼────────────────┐
         │ Convertir a Tensor        │
         │ tensor.float().to(DEVICE) │
         │ Shape: (1, 128, 40)       │
         └──────────┬────────────────┘
                    │
         ┌──────────▼────────────────┐
         │ Cargar Modelo TLOB        │
         │ según horizonte:          │
         │ - 10, 20, 50, 100 steps   │
         └──────────┬────────────────┘
                    │
         ┌──────────▼────────────────┐
         │ Inferencia (Forward Pass) │
         │ with torch.no_grad():     │
         │   logits = model(tensor)  │
         └──────────┬────────────────┘
                    │
         ┌──────────▼────────────────┐
         │ Aplicar Softmax           │
         │ probs = softmax(logits)   │
         │ Shape: (1, 3)             │
         └──────────┬────────────────┘
                    │
         ┌──────────▼────────────────┐
         │ ⚠️ INVERTIR ORDEN         │
         │ probs_inv = [probs[2],    │
         │              probs[1],    │
         │              probs[0]]    │
         │ (ver docs/FIX_ORDEN...)   │
         └──────────┬────────────────┘
                    │
         ┌──────────▼────────────────┐
         │ Obtener Predicción        │
         │ pred = argmax(probs_inv)  │
         │ 0=UP, 1=STAT, 2=DOWN      │
         └──────────┬────────────────┘
                    │
         ┌──────────▼────────────────┐
         │ Visualizar en Streamlit   │
         │ - Gráficos                │
         │ - Métricas                │
         │ - Heatmaps                │
         └───────────────────────────┘
```

### 2.2 Detección Automática de Normalización

El código en `app.py` detecta automáticamente si los datos están normalizados:

```python
def is_data_normalized(data):
    """
    Detecta si los datos ya están normalizados
    
    Heurística:
    - Si mean >> 100: Datos crudos (precios BTC en USDT)
    - Si mean ≈ 0 y std ≈ 1: Ya normalizados (z-scores)
    """
    mean = np.abs(data.mean())
    std = data.std()
    
    if mean > 100:
        return False, "raw"  # Datos crudos
    elif mean < 1 and 0.5 < std < 2:
        return True, "normalized"  # Ya normalizado
    else:
        return None, "unknown"  # No estamos seguros
```

**Ejemplo con Datos Reales**:

```python
# Datos crudos BTC
raw_data = np.array([[42150.5, 0.524, 42148.2, 0.631, ...]])  # Precios en USDT
mean = 21075.0  # >> 100
is_normalized(raw_data)  # --> (False, "raw")

# Datos normalizados
norm_data = np.array([[0.523, 0.145, -0.412, 0.223, ...]])  # Z-scores
mean = 0.0001  # ≈ 0
is_normalized(norm_data)  # --> (True, "normalized")
```

---

## 3. Preprocesamiento de Datos

### 3.1 Z-Score Normalization

**Fórmula**:

$$
x_{norm} = \frac{x - \mu}{\sigma}
$$

Donde:
- $x$: Valor original
- $\mu$: Media
- $\sigma$: Desviación estándar

**Implementación en `app.py`**:

```python
def normalize_raw_data(data):
    """
    Normaliza datos crudos separando precios y volúmenes
    
    Input: (128, 40) - valores crudos
    Output: (128, 40) - z-scores
    """
    df = pd.DataFrame(data)
    
    # Columnas pares = precios, impares = volúmenes
    mean_prices = df.iloc[:, 0::2].stack().mean()
    std_prices = df.iloc[:, 0::2].stack().std()
    mean_volumes = df.iloc[:, 1::2].stack().mean()
    std_volumes = df.iloc[:, 1::2].stack().std()
    
    # Normalizar por tipo
    for col in df.columns[0::2]:  # Precios
        df[col] = (df[col] - mean_prices) / std_prices
    
    for col in df.columns[1::2]:  # Volúmenes
        df[col] = (df[col] - mean_volumes) / std_volumes
    
    return df.values
```

### 3.2 Ejemplo Numérico Completo

#### Entrada (Datos Crudos):

```
Timestep  ASK_P1     ASK_V1   BID_P1     BID_V1   ...
0         42150.50   0.524    42148.20   0.631    ...
1         42151.20   0.489    42148.50   0.702    ...
...       ...        ...      ...        ...      ...
127       42155.80   0.512    42152.10   0.598    ...
```

**Estadísticas**:
- `mean_prices = 42152.35 USDT`
- `std_prices = 2.45 USDT`
- `mean_volumes = 0.567 BTC`
- `std_volumes = 0.089 BTC`

#### Salida (Datos Normalizados):

```
Timestep  ASK_P1   ASK_V1   BID_P1   BID_V1   ...
0         -0.755   -0.483   -1.691   0.719    ...
1         -0.469   -0.876   -1.569   1.517    ...
...       ...      ...      ...      ...      ...
127        1.410   -0.618   -0.102   0.348    ...
```

**Estadísticas**:
- `mean_normalized = 0.0001` ✓
- `std_normalized = 0.998` ✓

### 3.3 Validación de Shape

```python
# Verificar shape correcto
assert data.shape == (128, 40), f"Shape incorrecto: {data.shape}"

# 128 timesteps × 40 features
# Features: [ASK_P1, ASK_V1, BID_P1, BID_V1, ..., ASK_P10, ASK_V10, BID_P10, BID_V10]
```

---

## 4. Despliegue con Docker

### 4.1 Método Principal: Docker Compose

**Opción Recomendada - Un Solo Comando**:

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/tlob-prediction.git
cd tlob-prediction

# 2. Levantar aplicación con un solo comando
docker-compose up -d

# ✅ Listo! La app estará disponible en http://localhost:8501
```

**Verificar que está corriendo**:

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Verificar estado
docker-compose ps

# Output esperado:
# NAME              STATUS        PORTS
# tlob-streamlit    Up 2 minutes  0.0.0.0:8501->8501/tcp
```

### 4.2 Método Alternativo: Docker Build Manual

Si prefieres más control:

```bash
# 1. Construir imagen
docker build -t tlob-app:latest .

# 2. Ejecutar contenedor
docker run -d \
  --name tlob-container \
  -p 8501:8501 \
  -v $(pwd)/src/data:/app/src/data:ro \
  tlob-app:latest

# 3. Verificar logs
docker logs -f tlob-container
```

### 4.3 Dockerfile Explicado

```dockerfile
# Imagen base ligera
FROM python:3.12-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y gcc g++ git

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Configurar PYTHONPATH para imports
ENV PYTHONPATH=/app:${PYTHONPATH}

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Puntos Clave**:
1. `PYTHONPATH=/app`: Permite que Python encuentre el módulo `src`
2. `--server.address=0.0.0.0`: Permite acceso desde fuera del contenedor
3. `--no-cache-dir`: Reduce tamaño de imagen
4. `python:3.12-slim`: Imagen base ligera (~150MB vs ~1GB de python:3.12)

### 4.4 Docker Compose Explicado

```yaml
services:
  tlob-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: tlob-streamlit
    ports:
      - "8501:8501"  # Puerto host:contenedor
    volumes:
      - ./src/data:/app/src/data:ro  # Solo lectura
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    restart: unless-stopped  # Reiniciar automáticamente
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - tlob-network

networks:
  tlob-network:
    driver: bridge
```

**Ventajas de Docker Compose**:
- ✅ Configuración declarativa (YAML)
- ✅ Un solo comando para levantar todo
- ✅ Gestión automática de redes
- ✅ Healthchecks integrados
- ✅ Fácil escalar a múltiples servicios

### 4.5 Comandos Útiles de Docker

```bash
# Detener aplicación
docker-compose down

# Reconstruir imagen (después de cambios en código)
docker-compose up -d --build

# Ver logs de errores
docker-compose logs --tail=50

# Entrar al contenedor para debugging
docker-compose exec tlob-app /bin/bash

# Limpiar todo (contenedores, imágenes, volúmenes)
docker-compose down -v
docker system prune -af
```

---

## 5. Aplicación Streamlit

### 5.1 Arquitectura de la App

```
app.py
├── Sidebar (Configuración)
│   ├── Cargar Datos
│   │   ├── Selector de Fuente (Preprocesados / Crudos)
│   │   ├── Lista de Ejemplos
│   │   └── File Uploader
│   └── Info del Modelo
│
├── Tab 1: 📊 Datos
│   ├── Métricas (Shape, Mean, Std)
│   ├── Heatmap (128×40)
│   ├── Series Temporales
│   └── Comparación Raw vs Normalized
│
├── Tab 2: 🔍 Análisis
│   ├── 40 Histogramas
│   └── Tabla de Estadísticas
│
├── Tab 3: 🎯 Predicción
│   ├── Selector de Horizonte (10/20/50/100)
│   ├── Selector de Umbral (Normal/Spread)
│   ├── Botón "Ejecutar Predicción"
│   └── Info sobre Etiquetado
│
└── Tab 4: 📈 Resultados
    ├── Predicción Principal (grande)
    ├── Gráfico de Probabilidades
    ├── Métricas por Clase
    └── Detalles Técnicos
```

### 5.2 Screenshots Principales

#### Screenshot 1: Carga de Datos

**[PLACEHOLDER: Screenshot mostrando el sidebar con selector de fuente y botón de carga]**

**Descripción**: 
- Radio buttons para elegir fuente (Preprocesados / Crudos)
- Dropdown con lista de archivos disponibles
- Botón "Cargar" para confirmar
- File uploader para archivos personalizados

---

#### Screenshot 2: Visualización de Datos

**[PLACEHOLDER: Screenshot del Tab "Datos" mostrando heatmap y series temporales]**

**Descripción**:
- Heatmap interactivo de 128×40 (Plotly)
- Series temporales de ASK/BID prices y volumes
- Comparación lado a lado de datos raw vs normalizados (si aplicable)

---

#### Screenshot 3: Configuración de Predicción

**[PLACEHOLDER: Screenshot del Tab "Predicción" con selectores de horizonte y umbral]**

**Descripción**:
- Selector de horizonte (10, 20, 50, 100 timesteps)
- Radio buttons para tipo de umbral (Normal / Spread)
- Info box explicando el etiquetado
- Botón grande "Ejecutar Predicción"

---

#### Screenshot 4: Resultado de Predicción

**[PLACEHOLDER: Screenshot del Tab "Resultados" mostrando predicción UP con 85% de confianza]**

**Descripción**:
- Card grande central con predicción y emoji
- Color de fondo según clase (verde/azul/rojo)
- Porcentaje de confianza
- Métricas de probabilidades por clase

---

#### Screenshot 5: Gráfico de Probabilidades

**[PLACEHOLDER: Screenshot del gráfico de barras con probabilidades de las 3 clases]**

**Descripción**:
- Barra chart interactivo (Plotly)
- 3 barras: UP (verde), STATIONARY (azul), DOWN (rojo)
- Valores en porcentaje
- Etiquetas claras

---

### 5.3 Gestión de Estado con Session State

```python
# Variables clave en st.session_state
st.session_state = {
    # Datos
    'data': np.array,              # (128, 40) normalizado
    'data_raw': np.array,          # (128, 40) crudo (opcional)
    'filename': str,               # Nombre del archivo
    'source': str,                 # "Preprocesados" o "Crudos"
    
    # Modelos (caché)
    'tlob_model_h10': TLOB,        # Modelo horizonte 10
    'tlob_model_h20': TLOB,        # Modelo horizonte 20
    'tlob_model_h50': TLOB,        # Modelo horizonte 50
    'tlob_model_h100': TLOB,       # Modelo horizonte 100
    'current_horizon': int,        # Horizonte actual
    
    # Resultados de predicción
    'pred_result': dict,           # {'logits': [...], 'probs': [...], 'pred': int}
    'horizon': int,                # Horizonte usado
    'use_spread': bool,            # Tipo de umbral
    'alpha': float,                # Alpha calculado
    'alpha_type': str,             # "Normal" o "Spread"
}
```

**Ventajas**:
- No recargar modelos en cada interacción
- Mantener datos cargados entre tabs
- Preservar resultados de predicciones
- UX fluida sin pérdida de estado

### 5.4 Flujo de Usuario Típico

```
1. Usuario abre http://localhost:8501
   ↓
2. Ve pantalla inicial con info y botón de carga
   ↓
3. Selecciona fuente de datos en sidebar
   ↓
4. Elige archivo y hace click en "Cargar"
   ↓
5. Si es crudo: app detecta y normaliza automáticamente
   ↓
6. TAB 1: Visualiza heatmap y series temporales
   ↓
7. TAB 2: Explora distribuciones de features
   ↓
8. TAB 3: Configura horizonte (ej: 10) y umbral (Normal)
   ↓
9. Click en "Ejecutar Predicción"
   ↓
10. App carga modelo de horizonte 10 (o usa caché)
    ↓
11. Ejecuta forward pass del modelo
    ↓
12. Invierte orden de softmax (crítico!)
    ↓
13. TAB 4: Muestra resultado con gráficos y métricas
    ↓
14. Usuario puede probar otro horizonte o cargar otro archivo
```

---

## 6. Casos de Uso

### 6.1 Caso 1: Predicción con Datos Preprocesados

```python
# Datos ya están normalizados (de src/data/BTC/individual_examples/)
# Shape: (128, 40), mean≈0, std≈1

# 1. Cargar en Streamlit
uploaded_file = "example_1.npy"

# 2. No requiere normalización
is_normalized(data)  # --> True

# 3. Directo a inferencia
logits, probs, pred = run_prediction(model, data)

# 4. Mostrar resultado
st.success(f"Predicción: {CLASSES[pred]}")
```

### 6.2 Caso 2: Predicción con Datos Crudos (CSV)

```python
# Datos crudos desde Binance (precios en USDT, volúmenes en BTC)
# Shape: (128, 40), mean>>1000

# 1. Cargar CSV
uploaded_file = "raw_example_1.csv"
data_raw = pd.read_csv(uploaded_file).values  # (128, 40)

# 2. Detectar que necesita normalización
is_normalized(data_raw)  # --> False

# 3. Aplicar Z-score
data_normalized = normalize_raw_data(data_raw)

# 4. Inferencia
logits, probs, pred = run_prediction(model, data_normalized)

# 5. Mostrar resultado
st.success(f"Predicción: {CLASSES[pred]}")
```

### 6.3 Caso 3: Comparar Múltiples Horizontes

```python
# Probar diferentes horizontes de predicción

horizontes = [10, 20, 50, 100]
resultados = {}

for h in horizontes:
    # Cargar modelo correspondiente
    model = get_model(horizon=h)
    
    # Ejecutar predicción
    logits, probs, pred = run_prediction(model, data)
    
    # Guardar
    resultados[h] = {
        'pred': CLASSES[pred],
        'confianza': probs[pred]
    }

# Visualizar comparación
st.table(pd.DataFrame(resultados).T)
```

**Ejemplo de Output**:

```
Horizonte | Predicción  | Confianza
----------|-------------|----------
10        | UP 📈      | 85.2%
20        | UP 📈      | 78.4%
50        | STATIONARY | 65.1%
100       | DOWN 📉    | 72.3%
```

**Interpretación**: A corto plazo (10-20 steps) predice UP, pero a largo plazo (100 steps) predice DOWN.

---

## 7. Troubleshooting

### 7.1 Problemas Comunes

#### Error: "No module named 'src'"

**Causa**: PYTHONPATH no incluye el directorio raíz.

**Solución**:

```bash
# En Docker
ENV PYTHONPATH=/app:${PYTHONPATH}

# En local
export PYTHONPATH=/path/to/tlob-prediction:$PYTHONPATH
```

#### Error: "Shape incorrecto: (128, 42)"

**Causa**: CSV incluye columnas extra (timestamp, datetime).

**Solución**:

```python
# Eliminar columnas no necesarias
if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp', 'datetime'])

data = df.values  # Ahora shape=(128, 40) ✓
```

#### Error: "AttributeError: 'UploadedFile' object has no attribute 'suffix'"

**Causa**: Streamlit UploadedFile no tiene `.suffix` directamente.

**Solución**:

```python
# Usar Path(uploaded_file.name).suffix
from pathlib import Path

file_extension = Path(uploaded_file.name).suffix  # '.csv' o '.npy'
```

#### Error: Predicción siempre STATIONARY

**Causa**: Modelo no está cargando correctamente o datos no están normalizados.

**Verificar**:

```python
# 1. Verificar mean y std de datos
print(f"Mean: {data.mean()}, Std: {data.std()}")
# Esperado: Mean ≈ 0, Std ≈ 1

# 2. Verificar que modelo cargó
print(f"Parámetros: {sum(p.numel() for p in model.parameters())}")
# Esperado: ~1,100,000

# 3. Verificar que está en modo eval
print(model.training)  # False esperado
```

### 7.2 Performance Issues

#### Streamlit Lento

**Optimizaciones**:

```python
# 1. Usar @st.cache_data para cargar modelo
@st.cache_resource
def load_model(horizon):
    return get_model(horizon)

# 2. Cachear preprocesamiento
@st.cache_data
def preprocess_data(file_bytes):
    return normalize_raw_data(np.load(file_bytes))

# 3. Usar session_state para resultados
if 'pred_result' not in st.session_state:
    st.session_state['pred_result'] = run_prediction(model, data)
```

#### Docker Usa Mucha Memoria

**Optimizar Dockerfile**:

```dockerfile
# Usar multi-stage build
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
```

**Resultado**: Imagen ~400MB en lugar de ~1.2GB

---

## 8. Referencias

### Documentación Relacionada

1. **Mecanismo de Atención**: [`docs/MECANISMO_ATENCION_QKV.md`](MECANISMO_ATENCION_QKV.md)
   - Teoría matemática de Q, K, V
   - Ejemplo paso a paso con datos TLOB

2. **Innovaciones del Modelo**: [`docs/INNOVACIONES_TLOB.md`](INNOVACIONES_TLOB.md)
   - Dual Attention
   - BiN Normalization
   - Comparación vs otros modelos

3. **Arquitectura Completa**: [`docs/ARQUITECTURA_COMPLETA.md`](ARQUITECTURA_COMPLETA.md)
   - Estructura de 4 pares de Transformers
   - Dimensiones en cada capa
   - Forward pass detallado

4. **README Principal**: [`README.md`](../README.md)
   - Introducción al proyecto
   - Instalación rápida
   - Resultados y métricas

### Paper y Repositorio Original

```bibtex
@article{berti2025tlob,
  title={TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data},
  author={Berti, Leonardo and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2502.15757},
  year={2025}
}
```

- **Paper**: https://arxiv.org/pdf/2502.15757
- **Repositorio Original**: https://github.com/LeonardoBerti00/TLOB

### Herramientas Utilizadas

- **PyTorch**: https://pytorch.org/
- **Streamlit**: https://streamlit.io/
- **Docker**: https://www.docker.com/
- **Plotly**: https://plotly.com/python/
- **NumPy**: https://numpy.org/

---

**Última actualización**: Noviembre 2025  
**Versión**: 1.0.0

---

## Apéndice A: Checklist de Despliegue

Usa este checklist para verificar que todo está configurado correctamente:

### Pre-Despliegue

- [ ] Docker y Docker Compose instalados
- [ ] Repositorio clonado
- [ ] Checkpoints del modelo presentes en `src/data/checkpoints/TLOB/`
- [ ] Datos de ejemplo presentes en `src/data/BTC/`

### Despliegue

- [ ] `docker-compose up -d` ejecutado sin errores
- [ ] `docker-compose ps` muestra contenedor "Up"
- [ ] http://localhost:8501 accesible en navegador
- [ ] Sidebar muestra lista de ejemplos
- [ ] Se puede cargar un archivo sin errores

### Post-Despliegue

- [ ] Predicción funciona con datos preprocesados
- [ ] Predicción funciona con datos crudos (CSV)
- [ ] Se pueden cambiar horizontes (10, 20, 50, 100)
- [ ] Gráficos se visualizan correctamente
- [ ] No hay errores en `docker-compose logs`

### Troubleshooting

- [ ] Si falla: revisar logs con `docker-compose logs -f`
- [ ] Si no carga modelo: verificar PYTHONPATH en Dockerfile
- [ ] Si shape incorrecto: verificar que datos son (128, 40)
- [ ] Si siempre predice igual: verificar normalización de datos

---

**Fin del Documento**

