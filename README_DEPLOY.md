# 📈 TLOB: Predicción de Tendencias de Precios con Transformers

> **Despliegue de aplicación interactiva con Streamlit y Docker**

---

## 📄 Artículo Base

**Título:** "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data"

**Autores:** Leonardo Berti (Sapienza University of Rome), Gjergji Kasneci (Technical University of Munich)

**Repositorio Original:** [TLOB GitHub](https://github.com/lorenzoletizia/TLOB) *(enlace del paper)*

**Publicación:** 2024

---

## 🎯 Descripción del Modelo

### Resumen

TLOB es un modelo **Transformer** diseñado específicamente para predecir tendencias de precios usando datos del **Limit Order Book (LOB)**. A diferencia de modelos previos que utilizan arquitecturas complejas (CNNs, RNNs), TLOB demuestra que una arquitectura basada en Transformers con **Dual Attention** supera el estado del arte en múltiples datasets.

###Principales Innovaciones

1. **Dual Attention Mechanism:**
   - **Spatial Attention:** Captura relaciones entre diferentes features del LOB (precios ↔ volúmenes)
   - **Temporal Attention:** Captura evolución temporal del mercado
   - Permite al modelo adaptarse a la microestructura del mercado

2. **BiN (Batch-Instance Normalization):**
   - Normalización a nivel de batch e instancia
   - Estabiliza el entrenamiento con datos financieros volátiles

3. **Nuevo método de etiquetado:**
   - Elimina el sesgo de horizonte presente en trabajos anteriores
   - Mejora la robustez del modelo

4. **Generalización superior:**
   - Funciona en múltiples datasets (FI-2010, LOBSTER, Bitcoin)
   - Supera SoTA en F1-score (+3.7 en FI-2010, +1.1 en BTC)

### Aplicación

El modelo predice la **tendencia del precio** en un horizonte futuro fijo (10, 20, 50 o 100 timesteps) clasificando en 3 clases:
- 📉 **DOWN:** Tendencia bajista
- ➡️ **STATIONARY:** Precio estable
- 📈 **UP:** Tendencia alcista

---

## 🏗️ Arquitectura del Modelo

### Resumen Teórico

```
INPUT (batch, 128, 40)
  ↓
┌─────────────────────────────────┐
│ 1. BiN Normalization            │ ← Estabiliza entrenamiento
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 2. Linear Embedding (40 → 40)   │ ← Proyección a espacio latente
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 3. Positional Encoding          │ ← Encoding sinusoidal
│    (Sinusoidal)                  │
└──────────────┬──────────────────┘
               ↓
        ┌──────┴──────┐
        ↓             ↓
┌──────────────┐ ┌──────────────┐
│  Branch 1    │ │  Branch 2    │  ← DUAL ATTENTION
│  (Spatial)   │ │  (Temporal)  │     (Innovación clave)
│              │ │              │
│ 4 Layers     │ │ 4 Layers     │
│ Transformer  │ │ Transformer  │
└──────┬───────┘ └──────┬───────┘
       │                │
       └────────┬───────┘
                ↓
    ┌───────────────────────┐
    │ 4. Concatenate        │
    └───────────┬───────────┘
                ↓
    ┌───────────────────────┐
    │ 5. MLP Final          │ ← Clasificación
    │    (hidden → 3)       │
    └───────────┬───────────┘
                ↓
          OUTPUT (batch, 3)
      [DOWN, STATIONARY, UP]
```

### Componentes Clave

#### 1. **BiN (Batch-Instance Normalization)**
```python
# Normaliza tanto a nivel de batch como de instancia
# Fórmula: x_norm = (x - μ) / (σ + ε)
# Donde μ y σ se calculan en dos niveles
```

#### 2. **Positional Encoding**
```python
# Encoding sinusoidal para capturar orden temporal
# PE(pos, 2i) = sin(pos / 10000^(2i/d))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

#### 3. **Dual Attention**
```python
# Branch 1 (Spatial): Attention sobre features
Q, K, V = Linear(x)
Attention_spatial = Softmax(Q @ K^T / √d) @ V

# Branch 2 (Temporal): Attention sobre timesteps
# Similar pero con dimensiones transpuestas
```

#### 4. **Transformer Layer**
Cada layer contiene:
- Multi-Head Self-Attention
- Layer Normalization
- Feedforward MLP
- Residual Connections

**Parámetros totales:** 1,135,974 (~1.1M)

---

## 📦 Estructura del Proyecto

```
TLOB-main/
├── app.py                          # Aplicación Streamlit
├── Dockerfile                      # Configuración Docker
├── docker-compose.yml              # Orquestación (opcional)
├── requirements.txt                # Dependencias Python
├── .dockerignore                   # Archivos a excluir de Docker
│
├── models/                         # Arquitecturas de modelos
│   ├── tlob.py                     # Modelo TLOB principal
│   ├── bin.py                      # BiN Normalization
│   ├── mlplob.py                   # MLP auxiliar
│   └── ...
│
├── data/                           # Datos y checkpoints
│   ├── BTC/
│   │   └── individual_examples/    # 5 ejemplos precargados
│   │       ├── example_1.npy       # Ventana LOB (128×40)
│   │       ├── example_2.npy
│   │       ├── example_3.npy
│   │       ├── example_4.npy
│   │       ├── example_5.npy
│   │       └── README.md
│   │
│   └── checkpoints/
│       └── TLOB/
│           └── BTC_seq_size_128_horizon_10_seed_1/
│               ├── pt/
│               │   └── val_loss=0.623_epoch=2.pt  # Pesos del modelo
│               └── onnx/
│                   └── val_loss=0.623_epoch=2.onnx
│
├── preprocessing/                  # Scripts de preprocesamiento
│   ├── btc.py                      # Preprocesamiento Bitcoin
│   ├── dataset.py                  # Dataset PyTorch
│   └── ...
│
└── docs/                           # Documentación completa
    ├── knowledge.md
    ├── inference_guide.md
    └── RESUMEN_EJECUTIVO.md
```

---

## 🚀 Instalación y Ejecución

### Opción 1: Ejecución con Docker (Recomendado)

#### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/tlob-prediction.git
cd tlob-prediction
```

#### Paso 2: Construir la imagen Docker

```bash
docker build -t tlob-app .
```

Este comando:
- Lee el `Dockerfile`
- Instala todas las dependencias
- Copia el código fuente
- Configura la aplicación Streamlit

**Tiempo estimado:** 5-10 minutos (primera vez)

#### Paso 3: Ejecutar el contenedor

```bash
docker run -p 8501:8501 tlob-app
```

Parámetros:
- `-p 8501:8501`: Mapea el puerto 8501 del contenedor al host
- `tlob-app`: Nombre de la imagen

#### Paso 4: Acceder a la aplicación

Abre tu navegador y ve a:
```
http://localhost:8501
```

¡La aplicación debería estar corriendo! 🎉

---

### Opción 2: Ejecución Local (Sin Docker)

#### Paso 1: Instalar dependencias

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
pip install streamlit plotly seaborn
```

#### Paso 2: Ejecutar la aplicación

```bash
streamlit run app.py
```

#### Paso 3: Acceder

La aplicación se abrirá automáticamente en tu navegador, o accede a:
```
http://localhost:8501
```

---

## 🎮 Uso de la Aplicación

### Interfaz Principal

La aplicación tiene 4 pestañas principales:

#### 1. **📊 Datos**
- Visualiza la ventana temporal del LOB en formato heatmap
- Muestra evolución temporal de features clave
- Tabla con valores numéricos
- Estadísticas básicas (mean, std, min, max)

#### 2. **🔍 Análisis**
- Distribución de valores por feature
- Estadísticas detalladas (percentiles, cuartiles)
- Análisis visual de patrones

#### 3. **🎯 Predicción**
- Botón para ejecutar inferencia
- Carga del modelo TLOB
- Forward pass sobre los datos

#### 4. **📈 Resultados**
- Predicción final con emoji visual
- Confianza de la predicción
- Distribución de probabilidades (gráfico de barras)
- Logits y probabilidades detalladas
- Interpretación del resultado

### Flujo de Uso

```
1. Seleccionar ejemplo precargado (o subir archivo .npy)
   ↓
2. Explorar datos en pestaña "Datos"
   ↓
3. Analizar estadísticas en "Análisis"
   ↓
4. Ir a "Predicción" y hacer clic en "Ejecutar Predicción"
   ↓
5. Ver resultados en "Resultados"
```

---

## 🔧 Cómo Funciona la Inferencia

### 1. Carga de Pesos del Modelo

```python
# app.py - Función load_model()

# Paso 1: Instanciar arquitectura
model = TLOB(**MODEL_CONFIG)

# Paso 2: Cargar checkpoint (.pt)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

# Paso 3: Limpiar state_dict (remover prefijo "model.")
# PyTorch Lightning guarda con este prefijo
state_dict = checkpoint["state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("model."):
        new_key = key[6:]  # Remover prefijo
        new_state_dict[new_key] = value

# Paso 4: Cargar pesos en el modelo
model.load_state_dict(new_state_dict)

# Paso 5: Modo evaluación
model.eval()  # Desactiva dropout, batch norm, etc.
```

**Checkpoint usado:** `val_loss=0.623_epoch=2.pt`
- Mejor modelo del entrenamiento (epoch 2)
- Validation loss: 0.623
- Horizonte de predicción: 10 timesteps
- Entrenado en Bitcoin LOB (Enero 2023)

---

### 2. Preprocesamiento de Datos

```python
# app.py - Función load_lob_window()

# Los datos ya vienen preprocesados:
# - Normalización Z-score: (x - μ) / σ
# - Shape: (128, 40)
# - 128 timesteps consecutivos
# - 40 features del LOB

window = np.load(file_path)  # Cargar desde archivo

# Validar shape
assert window.shape == (128, 40), "Shape incorrecto"

# Estructura de features:
# 0-9:   ASK Prices (10 niveles)
# 10-19: ASK Volumes
# 20-29: BID Prices
# 30-39: BID Volumes
```

**Nota importante:** Los datos **ya están normalizados**. No se requiere preprocesamiento adicional.

---

### 3. Generación de la Salida (Inferencia)

```python
# app.py - Función predict()

def predict(model, window):
    # Paso 1: Añadir dimensión de batch
    # (128, 40) → (1, 128, 40)
    X = np.expand_dims(window, axis=0)
    
    # Paso 2: Convertir a tensor PyTorch
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    # Paso 3: Inferencia (sin calcular gradientes)
    with torch.no_grad():
        # Forward pass del modelo
        logits = model(X_tensor)  # Shape: (1, 3)
        
        # Aplicar softmax para obtener probabilidades
        # Softmax: e^x_i / sum(e^x_j)
        probs = F.softmax(logits, dim=1)  # Shape: (1, 3)
        
        # Clase predicha (argmax)
        pred = torch.argmax(probs, dim=1)  # Shape: (1,)
    
    # Paso 4: Convertir a NumPy y extraer valores
    return (
        logits[0].cpu().numpy(),  # [logit_down, logit_stat, logit_up]
        probs[0].cpu().numpy(),   # [p_down, p_stat, p_up]
        pred[0].item()            # 0, 1, o 2
    )
```

**Flujo interno del modelo:**
1. **Input:** `(1, 128, 40)`
2. **BiN Normalize:** Normalización dual
3. **Embed:** Linear (40 → 40)
4. **Add Pos Encoding:** `+ sinusoidal_encoding`
5. **Dual Attention:**
   - Branch 1 (Spatial): 4 layers Transformer
   - Branch 2 (Temporal): 4 layers Transformer
6. **Concatenate:** Unir ambas ramas
7. **MLP Final:** `hidden*2 → hidden → 3`
8. **Output:** `(1, 3)` logits

---

### 4. Integración con Streamlit

```python
# app.py - Visualización de resultados

# Paso 1: Ejecutar predicción
logits, probs, pred = predict(model, window)

# Paso 2: Mapear a etiquetas
pred_label = CLASS_LABELS[pred]  # "DOWN", "STATIONARY", "UP"
pred_emoji = CLASS_EMOJIS[pred]  # 📉, ➡️, 📈
confidence = probs[pred]          # Probabilidad de la clase predicha

# Paso 3: Visualizar
st.markdown(f"""
<div style="...">
    <h1>{pred_emoji}</h1>
    <h2>{pred_label}</h2>
    <h3>Confianza: {confidence:.2%}</h3>
</div>
""", unsafe_allow_html=True)

# Paso 4: Gráfico de probabilidades
fig = go.Figure(data=[
    go.Bar(x=["DOWN", "STATIONARY", "UP"], y=probs*100)
])
st.plotly_chart(fig)

# Paso 5: Detalles técnicos
st.code(f"""
Logits: [{logits[0]:.4f}, {logits[1]:.4f}, {logits[2]:.4f}]
Probabilidades: [{probs[0]:.2%}, {probs[1]:.2%}, {probs[2]:.2%}]
Predicción: {pred_label} (clase {pred})
""")
```

**Componentes de visualización:**
- Resultado principal con color dinámico
- Métricas de las 3 clases
- Gráfico interactivo de probabilidades (Plotly)
- Heatmap de la ventana temporal
- Evolución temporal de features clave
- Estadísticas detalladas

---

## 📊 Ejemplos Precargados

La aplicación incluye **5 ejemplos** listos para usar:

| Archivo | Predicción | Confianza | Interpretación |
|---------|------------|-----------|----------------|
| `example_1.npy` | ➡️ STATIONARY | 92.06% | Precio estable con alta confianza |
| `example_2.npy` | 📈 UP | 55.15% | Tendencia alcista moderada |
| `example_3.npy` | 📈 UP | 93.81% | Tendencia alcista muy fuerte |
| `example_4.npy` | ➡️ STATIONARY | 77.45% | Precio estable |
| `example_5.npy` | 📉 DOWN | 86.90% | Tendencia bajista fuerte |

Cada ejemplo es una ventana de **128 timesteps × 40 features** extraída del dataset de Bitcoin.

---

## 🐳 Comandos Docker Útiles

```bash
# Construir imagen
docker build -t tlob-app .

# Ejecutar contenedor
docker run -p 8501:8501 tlob-app

# Ejecutar en modo detached (background)
docker run -d -p 8501:8501 tlob-app

# Ver logs
docker logs <container_id>

# Detener contenedor
docker stop <container_id>

# Listar contenedores activos
docker ps

# Listar todas las imágenes
docker images

# Eliminar contenedor
docker rm <container_id>

# Eliminar imagen
docker rmi tlob-app

# Acceder al contenedor (debug)
docker exec -it <container_id> /bin/bash
```

---

## 🔬 Detalles Técnicos

### Requisitos de Sistema

- **Python:** 3.9+
- **RAM:** Mínimo 4GB (recomendado 8GB)
- **Disco:** ~2GB para imagen Docker
- **CPU:** Cualquier procesador moderno
- **GPU:** Opcional (el modelo funciona bien en CPU)

### Dependencias Principales

```
torch==2.0.1
pytorch-lightning==2.0.0
streamlit==1.28.0
plotly==5.17.0
numpy==1.24.0
pandas==2.0.0
einops==0.7.0
```

### Performance

- **Latencia de inferencia:** ~50-100ms por predicción (CPU)
- **Throughput:** ~10-20 predicciones/segundo
- **Tamaño del modelo:** ~4.5 MB (.pt) o ~4.3 MB (.onnx)

---

## 📚 Documentación Adicional

Para más detalles, consulta:

- **`docs/knowledge.md`:** Knowledge base completa del proyecto
- **`docs/inference_guide.md`:** Guía detallada de inferencia
- **`docs/RESUMEN_EJECUTIVO.md`:** Resumen ejecutivo
- **`data/BTC/individual_examples/README.md`:** Documentación de ejemplos

---

## 🤝 Contribuciones

Este proyecto es una implementación educativa del paper TLOB para el curso de Analítica.

**Equipo:**
- [Tu nombre]
- [Nombre compañero 1]
- [Nombre compañero 2]

**Instructor:** [Nombre del profesor]  
**Curso:** Analítica Avanzada  
**Universidad:** [Tu universidad]  
**Fecha:** Noviembre 2025

---

## 📄 Licencia

Este proyecto está basado en el repositorio original TLOB, sujeto a su licencia.

---

## 🙏 Agradecimientos

- **Autores del paper TLOB:** Leonardo Berti y Gjergji Kasneci
- **Dataset:** Bitcoin LOB de Kaggle
- **Frameworks:** PyTorch, PyTorch Lightning, Streamlit

---

## 📞 Contacto

Para preguntas o problemas:
- Abrir un issue en GitHub
- Contactar al equipo: [email]

---

**¡Gracias por usar TLOB! 🚀**


