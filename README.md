# 📈 TLOB: Predicción de Tendencias de Precios con Transformers

> **Aplicación interactiva de Streamlit para predicción de tendencias de precios en Bitcoin usando el modelo TLOB**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📄 Artículo Base

**Título:** "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data"

**Autores:** Leonardo Berti (Sapienza University of Rome), Gjergji Kasneci (Technical University of Munich)

**Repositorio Original:** [TLOB GitHub](https://github.com/lorenzoletizia/TLOB)

**Publicación:** 2024

---

## 🎯 Descripción del Modelo

TLOB es un modelo **Transformer** con **Dual Attention** diseñado para predecir tendencias de precios usando datos del **Limit Order Book (LOB)**. El modelo procesa 128 timesteps consecutivos del LOB (40 features) y predice la tendencia en los próximos 10 timesteps.

### Principales Innovaciones

1. **Dual Attention Mechanism:**
   - **Spatial Attention:** Captura relaciones entre features del LOB
   - **Temporal Attention:** Captura evolución temporal del mercado

2. **BiN (Batch-Instance Normalization):**
   - Normalización a nivel de batch e instancia
   - Estabiliza entrenamiento con datos financieros

3. **Nuevo método de etiquetado:**
   - Elimina sesgo de horizonte de trabajos anteriores
   - Mejora robustez del modelo

4. **Generalización superior:**
   - Supera SoTA en múltiples datasets
   - F1-score: +3.7 en FI-2010, +1.1 en BTC

---

## 🏗️ Resumen Teórico de la Arquitectura

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
└──────────────┬──────────────────┘
               ↓
        ┌──────┴──────┐
        ↓             ↓
┌──────────────┐ ┌──────────────┐
│  Branch 1    │ │  Branch 2    │  ← DUAL ATTENTION
│  (Spatial)   │ │  (Temporal)  │     (Innovación clave)
│ 4 Layers     │ │ 4 Layers     │
└──────┬───────┘ └──────┬───────┘
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

**Parámetros totales:** 1,135,974 (~1.1M)

---

## 🚀 Pasos para Ejecutar el Proyecto

### Opción 1: Docker (Recomendado) 🐳

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/tlob-prediction.git
cd tlob-prediction

# 2. Construir imagen
docker build -t tlob-app .

# 3. Ejecutar contenedor
docker run -p 8501:8501 tlob-app

# 4. Abrir navegador
# → http://localhost:8501
```

**O con docker-compose:**

```bash
docker-compose up
```

---

### Opción 2: Instalación Local 💻

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements_streamlit.txt

# 3. Ejecutar aplicación
streamlit run app.py

# 4. La app se abre automáticamente en el navegador
```

---

## 🎮 Cómo Usar la Aplicación

### Flujo de Uso:

1. **Cargar Datos:** Selecciona un ejemplo precargado o sube tu archivo `.npy`
2. **Explorar Datos:** Visualiza el heatmap y series temporales
3. **Realizar Predicción:** Click en "Ejecutar Predicción"
4. **Ver Resultados:** Visualiza la tendencia predicha y confianza

### Ejemplos Precargados:

| Archivo | Predicción | Confianza |
|---------|------------|-----------|
| `example_1.npy` | ➡️ STATIONARY | 92.06% |
| `example_2.npy` | 📈 UP | 55.15% |
| `example_3.npy` | 📈 UP | 93.81% ⭐ |
| `example_4.npy` | ➡️ STATIONARY | 77.45% |
| `example_5.npy` | 📉 DOWN | 86.90% |

---

## 📚 Explicación de Cómo se Cargan los Pesos

```python
# 1. Instanciar arquitectura TLOB
model = TLOB(
    hidden_dim=40,
    num_layers=4,
    seq_size=128,
    num_features=40,
    num_heads=1,
    is_sin_emb=True,
    dataset_type="BTC"
)

# 2. Cargar checkpoint (.pt)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

# 3. Limpiar state_dict (remover prefijo "model." de PyTorch Lightning)
state_dict = checkpoint["state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("model."):
        new_key = key[6:]  # Remover "model."
        new_state_dict[new_key] = value

# 4. Cargar pesos en el modelo
model.load_state_dict(new_state_dict)

# 5. Modo evaluación (desactiva dropout, batch norm)
model.eval()
```

---

## 🔄 Explicación del Preprocesamiento de Datos

Los datos del LOB **ya vienen preprocesados**:

```python
# Los archivos .npy contienen:
# - Shape: (128, 40)
# - 128 timesteps consecutivos
# - 40 features del LOB
# - Normalización Z-score aplicada: (x - μ) / σ

# Estructura de las 40 features:
# Features 0-9:   ASK Prices (10 niveles de profundidad)
# Features 10-19: ASK Volumes
# Features 20-29: BID Prices
# Features 30-39: BID Volumes

# Carga simple:
window = np.load(file_path)  # Shape: (128, 40)

# No requiere preprocesamiento adicional
```

---

## 🎯 Explicación de Cómo se Genera la Salida (Inferencia)

```python
def predict(model, window):
    """
    Proceso de inferencia completo
    """
    # 1. Añadir dimensión de batch: (128, 40) → (1, 128, 40)
    X = np.expand_dims(window, axis=0)
    
    # 2. Convertir a tensor de PyTorch
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    # 3. Forward pass (sin calcular gradientes)
    with torch.no_grad():
        # Inferencia del modelo
        logits = model(X_tensor)  # Shape: (1, 3)
        
        # Aplicar softmax para obtener probabilidades
        # Softmax: e^x_i / sum(e^x_j)
        probs = F.softmax(logits, dim=1)  # Shape: (1, 3)
        
        # Obtener clase predicha (argmax)
        pred = torch.argmax(probs, dim=1)  # Shape: (1,)
    
    # 4. Convertir a NumPy y retornar
    return (
        logits[0].cpu().numpy(),  # [logit_down, logit_stat, logit_up]
        probs[0].cpu().numpy(),   # [p_down, p_stat, p_up]
        pred[0].item()            # 0, 1, o 2
    )

# Interpretación de resultados:
# pred = 0 → DOWN (precio bajará)
# pred = 1 → STATIONARY (precio estable)
# pred = 2 → UP (precio subirá)
```

---

## 🖥️ Explicación de la Integración con Streamlit

### Estructura de la Aplicación:

```python
# 1. Configuración de la página
st.set_page_config(
    page_title="TLOB - Predicción de Tendencias",
    page_icon="📈",
    layout="wide"
)

# 2. Carga del modelo (con session_state para caching)
if 'model' not in st.session_state:
    st.session_state['model'] = load_model()

# 3. Sidebar para cargar datos
with st.sidebar:
    # Selector de ejemplos precargados
    selected_file = st.selectbox("Selecciona un ejemplo:", example_files)
    
    # Botón para cargar
    if st.button("Cargar Ejemplo"):
        st.session_state['window'] = load_lob_window(selected_file)

# 4. Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["Datos", "Análisis", "Predicción", "Resultados"])

# 5. Visualización de datos (tab1)
with tab1:
    # Heatmap interactivo con Plotly
    fig = go.Figure(data=go.Heatmap(z=window.T, ...))
    st.plotly_chart(fig)

# 6. Ejecución de predicción (tab3)
with tab3:
    if st.button("Ejecutar Predicción"):
        logits, probs, pred = predict(model, window)
        st.session_state['pred'] = pred

# 7. Visualización de resultados (tab4)
with tab4:
    # Mostrar resultado con HTML personalizado
    st.markdown(f"""
    <div style="...">
        <h1>{emoji}</h1>
        <h2>{pred_label}</h2>
        <h3>Confianza: {confidence:.2%}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico de probabilidades
    fig = go.Figure(data=[go.Bar(x=labels, y=probs*100)])
    st.plotly_chart(fig)
```

---

## 📂 Estructura del Proyecto

```
TLOB-main/
├── app.py                          # Aplicación Streamlit principal
├── Dockerfile                      # Configuración Docker
├── docker-compose.yml              # Orquestación Docker
├── requirements_streamlit.txt      # Dependencias Python
│
├── models/                         # Arquitecturas de modelos
│   ├── tlob.py                     # Modelo TLOB
│   ├── bin.py                      # BiN Normalization
│   └── ...
│
├── data/
│   ├── BTC/
│   │   └── individual_examples/    # 5 ejemplos precargados
│   │       ├── example_1.npy
│   │       ├── example_2.npy
│   │       ├── example_3.npy
│   │       ├── example_4.npy
│   │       └── example_5.npy
│   │
│   └── checkpoints/
│       └── TLOB/
│           └── BTC_seq_size_128_horizon_10_seed_1/
│               └── pt/
│                   └── val_loss=0.623_epoch=2.pt
│
├── preprocessing/                  # Scripts de preprocesamiento
├── docs/                           # Documentación completa
│
├── README.md                       # Este archivo
├── README_DEPLOY.md                # Documentación detallada (500+ líneas)
├── QUICK_START.md                  # Inicio rápido
├── TROUBLESHOOTING.md              # Solución de problemas
├── TEST_APP.md                     # Guía de testing
└── ENTREGA_FINAL.md                # Resumen ejecutivo
```

---

## 🔧 Requisitos

- **Python:** 3.12+ (recomendado para mejor performance)
- **RAM:** Mínimo 4GB (recomendado 8GB)
- **Disco:** ~2GB para Docker
- **CPU:** Cualquier procesador moderno
- **GPU:** Opcional (funciona perfectamente en CPU)

---

## 📊 Visualizaciones Incluidas

1. **Heatmap Temporal:** Visualización de 128 timesteps × 40 features
2. **Series Temporales:** Evolución de 4 features clave
3. **Distribuciones:** Histogramas de valores por feature
4. **Probabilidades:** Gráfico de barras de las 3 clases
5. **Resultado Principal:** Visualización con emoji y confianza

---

## 🐛 Troubleshooting

### Error: "Module 'streamlit' not found"
```bash
pip install -r requirements_streamlit.txt
```

### Error: "Port 8501 already in use"
```bash
streamlit run app.py --server.port 8502
```

### Error: "CUDA not available"
```
No es un problema. El modelo funciona perfectamente en CPU.
```

**Ver `TROUBLESHOOTING.md` para más soluciones.**

---

## 📚 Documentación Completa

- **`README_DEPLOY.md`** - Documentación exhaustiva (500+ líneas)
- **`QUICK_START.md`** - Inicio rápido en 3 pasos
- **`TROUBLESHOOTING.md`** - 10 problemas comunes y soluciones
- **`TEST_APP.md`** - Checklist completo de testing
- **`ENTREGA_FINAL.md`** - Resumen ejecutivo del proyecto

---

## 🎓 Equipo

- **[Tu Nombre]**
- **[Compañero 1]**
- **[Compañero 2]**

**Curso:** Analítica Avanzada  
**Universidad:** [Tu Universidad]  
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

**¡Gracias por revisar nuestro proyecto! 🚀📈**
