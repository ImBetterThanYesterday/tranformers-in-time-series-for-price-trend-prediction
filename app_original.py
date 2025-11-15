"""
Aplicación de Streamlit para Predicción de Tendencias de Precios con TLOB
==========================================================================

Esta aplicación permite:
1. Cargar y visualizar datos de series temporales del Limit Order Book (LOB)
2. Seleccionar ejemplos precargados
3. Realizar predicciones con el modelo TLOB
4. Visualizar resultados de forma interactiva

Modelo: TLOB (Transformer with Dual Attention)
Dataset: Bitcoin LOB (2023-01-09 to 2023-01-20)
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from models.tlob import TLOB
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="TLOB - Predicción de Tendencias",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================
CHECKPOINT_PATH = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/val_loss=0.623_epoch=2.pt"
EXAMPLES_DIR = Path("data/BTC/individual_examples")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "hidden_dim": 40,
    "num_layers": 4,
    "seq_size": 128,
    "num_features": 40,
    "num_heads": 1,
    "is_sin_emb": True,
    "dataset_type": "BTC",
}

CLASS_LABELS = {0: "DOWN", 1: "STATIONARY", 2: "UP"}
CLASS_EMOJIS = {0: "📉", 1: "➡️", 2: "📈"}
CLASS_COLORS = {0: "#ef4444", 1: "#3b82f6", 2: "#10b981"}

# ============================================================================
# FUNCIONES DE CARGA DEL MODELO
# ============================================================================
@st.cache_resource(show_spinner="Cargando modelo TLOB...")
def load_model():
    """
    Carga el modelo TLOB desde el checkpoint entrenado.
    
    Proceso:
    1. Instancia la arquitectura TLOB con los hiperparámetros
    2. Carga el checkpoint (.pt) que contiene los pesos entrenados
    3. Remueve el prefijo "model." del state_dict (artefacto de PyTorch Lightning)
    4. Carga los pesos en el modelo
    5. Pone el modelo en modo evaluación
    
    Returns:
        model: Modelo TLOB cargado y listo para inferencia
    """
    # Crear configuración local para evitar problemas de hashing
    config = {
        "hidden_dim": 40,
        "num_layers": 4,
        "seq_size": 128,
        "num_features": 40,
        "num_heads": 1,
        "is_sin_emb": True,
        "dataset_type": "BTC",
    }
    
    # Crear instancia del modelo con la arquitectura definida
    model = TLOB(**config)
    model.to(DEVICE)
    model.eval()  # Modo evaluación (desactiva dropout, batch norm, etc.)
    
    # Cargar checkpoint con los pesos entrenados
    checkpoint_path = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/val_loss=0.623_epoch=2.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Limpiar el state_dict: remover prefijo "model." de las keys
    # Esto es necesario porque PyTorch Lightning guarda con ese prefijo
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[6:]  # Remover "model."
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Cargar los pesos en el modelo
    model.load_state_dict(new_state_dict)
    
    return model

# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================
def load_example_files():
    """
    Carga la lista de archivos de ejemplo disponibles.
    
    Returns:
        list: Lista de rutas a los archivos example_N.npy
    """
    if not EXAMPLES_DIR.exists():
        return []
    
    # Buscar solo archivos example_N.npy (no los _result.npy)
    all_files = sorted(EXAMPLES_DIR.glob("example_*.npy"))
    example_files = [f for f in all_files if not f.stem.endswith("_result")]
    
    return example_files

def load_lob_window(file_path):
    """
    Carga una ventana de LOB desde archivo .npy
    
    Preprocesamiento:
    - Los datos ya están normalizados con Z-score
    - Shape esperado: (128, 40)
    - 128 timesteps consecutivos
    - 40 features del LOB (10 niveles × 4 tipos)
    
    Args:
        file_path: Ruta al archivo .npy
        
    Returns:
        np.ndarray: Ventana LOB de shape (128, 40)
    """
    window = np.load(file_path)
    
    # Validar shape
    expected_shape = (128, 40)
    if window.shape != expected_shape:
        st.error(f"❌ Shape incorrecto. Esperado: {expected_shape}, Recibido: {window.shape}")
        return None
    
    return window

# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================
def predict(model, window):
    """
    Realiza predicción sobre una ventana de LOB.
    
    Proceso de inferencia:
    1. Preprocesar: Añadir dimensión de batch (1, 128, 40)
    2. Convertir a tensor de PyTorch
    3. Forward pass del modelo (sin calcular gradientes)
    4. Aplicar softmax a los logits para obtener probabilidades
    5. Extraer la clase con mayor probabilidad
    
    Args:
        model: Modelo TLOB cargado
        window: Ventana LOB de shape (128, 40)
        
    Returns:
        tuple: (logits, probabilidades, clase_predicha)
    """
    # Añadir dimensión de batch: (128, 40) → (1, 128, 40)
    X = np.expand_dims(window, axis=0)
    
    # Convertir a tensor de PyTorch
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    # Inferencia (sin calcular gradientes para ahorrar memoria)
    with torch.no_grad():
        # Forward pass: obtener logits (salida cruda del modelo)
        logits = model(X_tensor)
        
        # Aplicar softmax para convertir logits a probabilidades
        # Softmax: e^x_i / sum(e^x_j) → valores entre 0 y 1 que suman 1
        probs = F.softmax(logits, dim=1)
        
        # Obtener la clase predicha (índice con mayor probabilidad)
        pred = torch.argmax(probs, dim=1)
    
    # Convertir a NumPy y extraer valores escalares
    return (
        logits[0].cpu().numpy(),  # Logits: 3 valores
        probs[0].cpu().numpy(),   # Probabilidades: 3 valores que suman 1.0
        pred[0].item()            # Predicción: 0, 1, o 2
    )

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================
def plot_lob_heatmap(window):
    """
    Crea un heatmap interactivo de la ventana LOB.
    
    Visualización:
    - Eje X: Timesteps (0-127)
    - Eje Y: Features (0-39)
    - Color: Valor normalizado (Z-score)
    """
    fig = go.Figure(data=go.Heatmap(
        z=window.T,
        x=list(range(128)),
        y=list(range(40)),
        colorscale='RdYlBu_r',
        colorbar=dict(title="Valor Z-score")
    ))
    
    fig.update_layout(
        title="Heatmap de la Ventana LOB (128 timesteps × 40 features)",
        xaxis_title="Timestep",
        yaxis_title="Feature Index",
        height=500
    )
    
    return fig

def plot_temporal_evolution(window):
    """
    Grafica la evolución temporal de 4 features clave del LOB.
    """
    fig = go.Figure()
    
    features_to_plot = [
        (0, "ASK Price (Nivel 1)", "#ef4444"),
        (10, "ASK Volume (Nivel 1)", "#f97316"),
        (20, "BID Price (Nivel 1)", "#10b981"),
        (30, "BID Volume (Nivel 1)", "#3b82f6")
    ]
    
    for feat_idx, name, color in features_to_plot:
        fig.add_trace(go.Scatter(
            x=list(range(128)),
            y=window[:, feat_idx],
            mode='lines',
            name=name,
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title="Evolución Temporal de Features Clave",
        xaxis_title="Timestep",
        yaxis_title="Valor (Z-score normalizado)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_feature_distribution(window):
    """
    Muestra la distribución de valores de las primeras 10 features.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Distribución de Valores - Primeras 10 Features", fontsize=14, fontweight='bold')
    
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.hist(window[:, i], bins=20, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_title(f'Feature {i}', fontsize=10)
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_probabilities(probs):
    """
    Crea un gráfico de barras con las probabilidades de cada clase.
    """
    labels = ["DOWN 📉", "STATIONARY ➡️", "UP 📈"]
    colors = ["#ef4444", "#3b82f6", "#10b981"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probs * 100,
            marker_color=colors,
            text=[f"{p:.2%}" for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probabilidades de Predicción",
        yaxis_title="Probabilidad (%)",
        height=400,
        showlegend=False
    )
    
    return fig

# ============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ============================================================================

def main():
    # Header
    st.title("📈 TLOB: Predicción de Tendencias de Precios")
    st.markdown("""
    **Modelo:** Transformer con Dual Attention para datos de Limit Order Book (LOB)  
    **Paper:** "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction"  
    **Dataset:** Bitcoin LOB (Enero 2023)
    """)
    
    st.divider()
    
    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        st.subheader("📊 Información del Modelo")
        st.info(f"""
        **Arquitectura:** TLOB  
        **Parámetros:** 1,135,974  
        **Horizonte:** 10 timesteps  
        **Device:** {DEVICE}
        """)
        
        st.subheader("📂 Cargar Datos")
        
        # Cargar ejemplos disponibles
        example_files = load_example_files()
        
        if example_files:
            # Opción 1: Seleccionar ejemplo precargado
            st.write("**Ejemplos Precargados:**")
            selected_file = st.selectbox(
                "Selecciona un ejemplo:",
                example_files,
                format_func=lambda x: x.name
            )
            
            if st.button("🔄 Cargar Ejemplo", type="primary"):
                st.session_state['current_file'] = selected_file
                st.session_state['window'] = load_lob_window(selected_file)
                st.rerun()
        
        st.divider()
        
        # Opción 2: Subir archivo propio
        st.write("**O sube tu propio archivo:**")
        uploaded_file = st.file_uploader(
            "Subir archivo .npy",
            type=['npy'],
            help="El archivo debe tener shape (128, 40)"
        )
        
        if uploaded_file is not None:
            try:
                window = np.load(uploaded_file)
                if window.shape == (128, 40):
                    st.session_state['window'] = window
                    st.session_state['current_file'] = uploaded_file.name
                    st.success("✅ Archivo cargado correctamente")
                else:
                    st.error(f"❌ Shape incorrecto: {window.shape}. Esperado: (128, 40)")
            except Exception as e:
                st.error(f"❌ Error al cargar archivo: {e}")
    
    # Main content
    if 'window' not in st.session_state:
        # Pantalla de bienvenida
        st.info("👈 Selecciona un ejemplo o sube un archivo .npy desde el panel lateral para comenzar.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ejemplos Disponibles", len(example_files))
        with col2:
            st.metric("Shape Esperado", "(128, 40)")
        with col3:
            st.metric("Clases", "3 (DOWN, STAT, UP)")
        
        # Información adicional
        with st.expander("ℹ️ ¿Qué es el Limit Order Book (LOB)?"):
            st.markdown("""
            El **Limit Order Book** es una estructura que registra todas las órdenes de compra (BID) 
            y venta (ASK) pendientes en un mercado financiero.
            
            **Estructura de los datos (40 features):**
            - Features 0-9: Precios ASK (10 niveles de profundidad)
            - Features 10-19: Volúmenes ASK
            - Features 20-29: Precios BID
            - Features 30-39: Volúmenes BID
            
            **Ventana temporal:** 128 snapshots consecutivos del LOB
            """)
        
        with st.expander("🧠 ¿Cómo funciona el modelo TLOB?"):
            st.markdown("""
            **TLOB** es un modelo Transformer con **Dual Attention** que procesa series temporales del LOB.
            
            **Arquitectura:**
            1. **BiN Normalization:** Normalización a nivel de batch e instancia
            2. **Embedding:** Proyección lineal de 40 → 40 dimensiones
            3. **Positional Encoding:** Encoding sinusoidal para capturar orden temporal
            4. **Dual Attention (innovación clave):**
               - Branch 1 (Spatial): Captura relaciones entre features
               - Branch 2 (Temporal): Captura evolución temporal
            5. **MLP Final:** Clasificación en 3 clases
            
            **Salida:** Predicción de tendencia en próximos 10 timesteps
            """)
        
        return
    
    # Si hay datos cargados, mostrar análisis
    window = st.session_state['window']
    file_name = st.session_state.get('current_file', 'archivo cargado')
    
    st.success(f"✅ **Archivo cargado:** {file_name if isinstance(file_name, str) else file_name.name}")
    
    # Tabs para organizar la visualización
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🔍 Análisis", "🎯 Predicción", "📈 Resultados"])
    
    # ========================================================================
    # TAB 1: VISUALIZACIÓN DE DATOS
    # ========================================================================
    with tab1:
        st.header("📊 Visualización de Datos")
        
        # Estadísticas básicas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Shape", f"{window.shape[0]} × {window.shape[1]}")
        with col2:
            st.metric("Mean", f"{window.mean():.4f}")
        with col3:
            st.metric("Std", f"{window.std():.4f}")
        with col4:
            st.metric("Range", f"{window.min():.2f} ~ {window.max():.2f}")
        
        st.divider()
        
        # Heatmap
        st.subheader("🌡️ Heatmap de la Ventana Temporal")
        st.plotly_chart(plot_lob_heatmap(window), use_container_width=True)
        
        # Evolución temporal
        st.subheader("📈 Evolución Temporal de Features Clave")
        st.plotly_chart(plot_temporal_evolution(window), use_container_width=True)
        
        # Tabla de datos (primeros 10 timesteps)
        with st.expander("🔢 Ver Datos Numéricos (Primeros 10 timesteps)"):
            df = pd.DataFrame(
                window[:10, :10],
                columns=[f"F{i}" for i in range(10)],
                index=[f"T{i}" for i in range(10)]
            )
            st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
    
    # ========================================================================
    # TAB 2: ANÁLISIS ESTADÍSTICO
    # ========================================================================
    with tab2:
        st.header("🔍 Análisis Estadístico")
        
        # Distribución de features
        st.subheader("📊 Distribución de Valores (Primeras 10 Features)")
        fig_dist = plot_feature_distribution(window)
        st.pyplot(fig_dist)
        
        # Estadísticas por feature
        st.subheader("📈 Estadísticas por Feature")
        
        stats_data = []
        for i in range(min(20, window.shape[1])):
            feat = window[:, i]
            stats_data.append({
                'Feature': f'F{i}',
                'Mean': feat.mean(),
                'Std': feat.std(),
                'Min': feat.min(),
                'Max': feat.max(),
                'Q25': np.percentile(feat, 25),
                'Q50': np.percentile(feat, 50),
                'Q75': np.percentile(feat, 75)
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
    
    # ========================================================================
    # TAB 3: PREDICCIÓN
    # ========================================================================
    with tab3:
        st.header("🎯 Realizar Predicción")
        
        st.info("""
        **¿Qué predice el modelo?**  
        El modelo TLOB predice la **tendencia del precio** en los próximos **10 timesteps**:
        - 📉 **DOWN:** El precio bajará
        - ➡️ **STATIONARY:** El precio se mantendrá estable
        - 📈 **UP:** El precio subirá
        """)
        
        if st.button("🚀 Ejecutar Predicción", type="primary", use_container_width=True):
            with st.spinner("Cargando modelo..."):
                model = load_model()
            
            with st.spinner("Realizando inferencia..."):
                logits, probs, pred = predict(model, window)
            
            # Guardar en session state
            st.session_state['logits'] = logits
            st.session_state['probs'] = probs
            st.session_state['pred'] = pred
            st.session_state['prediction_done'] = True
            
            st.success("✅ Predicción completada!")
            st.rerun()
    
    # ========================================================================
    # TAB 4: RESULTADOS
    # ========================================================================
    with tab4:
        st.header("📈 Resultados de la Predicción")
        
        if not st.session_state.get('prediction_done', False):
            st.warning("⚠️ Primero ejecuta la predicción en la pestaña 'Predicción'")
            return
        
        logits = st.session_state['logits']
        probs = st.session_state['probs']
        pred = st.session_state['pred']
        
        pred_label = CLASS_LABELS[pred]
        pred_emoji = CLASS_EMOJIS[pred]
        confidence = probs[pred]
        
        # Resultado principal
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, {CLASS_COLORS[pred]}22, {CLASS_COLORS[pred]}44); border-radius: 15px; margin: 20px 0;">
            <h1 style="font-size: 60px; margin: 0;">{pred_emoji}</h1>
            <h2 style="margin: 10px 0;">{pred_label}</h2>
            <h3 style="color: {CLASS_COLORS[pred]}; margin: 0;">Confianza: {confidence:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "📉 DOWN",
                f"{probs[0]:.2%}",
                delta=f"Logit: {logits[0]:.2f}"
            )
        with col2:
            st.metric(
                "➡️ STATIONARY",
                f"{probs[1]:.2%}",
                delta=f"Logit: {logits[1]:.2f}"
            )
        with col3:
            st.metric(
                "📈 UP",
                f"{probs[2]:.2%}",
                delta=f"Logit: {logits[2]:.2f}"
            )
        
        st.divider()
        
        # Gráfico de probabilidades
        st.subheader("📊 Distribución de Probabilidades")
        st.plotly_chart(plot_probabilities(probs), use_container_width=True)
        
        # Interpretación
        st.subheader("💡 Interpretación")
        
        if confidence > 0.90:
            conf_text = "**MUY ALTA**"
            conf_color = "green"
        elif confidence > 0.75:
            conf_text = "**ALTA**"
            conf_color = "blue"
        elif confidence > 0.60:
            conf_text = "**MODERADA**"
            conf_color = "orange"
        else:
            conf_text = "**BAJA**"
            conf_color = "red"
        
        st.markdown(f"""
        El modelo predice con confianza {conf_text} (:{conf_color}[{confidence:.2%}]) que el precio 
        tendrá una tendencia **{pred_label}** en los próximos **10 timesteps**.
        
        **Detalles técnicos:**
        - **Logits:** Salidas crudas del modelo antes de softmax
        - **Probabilidades:** Valores normalizados que suman 1.0
        - **Predicción:** Clase con mayor probabilidad
        """)
        
        # Información adicional
        with st.expander("🔬 Detalles Técnicos de la Predicción"):
            st.code(f"""
Entrada:
  Shape: {window.shape}
  Mean: {window.mean():.4f}
  Std: {window.std():.4f}

Logits (salida cruda):
  DOWN:       {logits[0]:>8.4f}
  STATIONARY: {logits[1]:>8.4f}
  UP:         {logits[2]:>8.4f}

Probabilidades (post-softmax):
  DOWN:       {probs[0]:>7.2%}
  STATIONARY: {probs[1]:>7.2%}
  UP:         {probs[2]:>7.2%}

Predicción Final: {pred_label} (clase {pred})
Confianza: {confidence:.4f}
            """, language="text")

# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================
if __name__ == "__main__":
    main()


