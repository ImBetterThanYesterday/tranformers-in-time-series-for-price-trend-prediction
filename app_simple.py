"""
Aplicación de Streamlit para Predicción de Tendencias de Precios con TLOB
==========================================================================

Versión simplificada sin caching complejo para evitar errores de recursión.
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from models.tlob import TLOB

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
# CONSTANTES
# ============================================================================
EXAMPLES_DIR = Path("data/BTC/individual_examples")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS = {0: "DOWN", 1: "STATIONARY", 2: "UP"}
CLASS_EMOJIS = {0: "📉", 1: "➡️", 2: "📈"}
CLASS_COLORS = {0: "#ef4444", 1: "#3b82f6", 2: "#10b981"}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_model():
    """Carga el modelo TLOB desde el checkpoint"""
    if 'model' not in st.session_state:
        with st.spinner("🔄 Cargando modelo TLOB..."):
            config = {
                "hidden_dim": 40,
                "num_layers": 4,
                "seq_size": 128,
                "num_features": 40,
                "num_heads": 1,
                "is_sin_emb": True,
                "dataset_type": "BTC",
            }
            
            model = TLOB(**config)
            model.to(DEVICE)
            model.eval()
            
            checkpoint_path = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/val_loss=0.623_epoch=2.pt"
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict)
            st.session_state['model'] = model
            st.success("✅ Modelo cargado correctamente!")
    
    return st.session_state['model']

def load_example_files():
    """Carga la lista de archivos de ejemplo disponibles"""
    if not EXAMPLES_DIR.exists():
        return []
    
    all_files = sorted(EXAMPLES_DIR.glob("example_*.npy"))
    example_files = [f for f in all_files if not f.stem.endswith("_result")]
    
    return example_files

def load_lob_window(file_path):
    """Carga una ventana de LOB desde archivo .npy"""
    window = np.load(file_path)
    
    expected_shape = (128, 40)
    if window.shape != expected_shape:
        st.error(f"❌ Shape incorrecto. Esperado: {expected_shape}, Recibido: {window.shape}")
        return None
    
    return window

def predict(model, window):
    """Realiza predicción sobre una ventana de LOB"""
    X = np.expand_dims(window, axis=0)
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
    
    return (
        logits[0].cpu().numpy(),
        probs[0].cpu().numpy(),
        pred[0].item()
    )

def plot_lob_heatmap(window):
    """Crea un heatmap interactivo de la ventana LOB"""
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
    """Grafica la evolución temporal de 4 features clave del LOB"""
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
    """Muestra la distribución de valores de las primeras 10 features"""
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
    """Crea un gráfico de barras con las probabilidades de cada clase"""
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
# INTERFAZ PRINCIPAL
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
    
    # Sidebar
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
        
        example_files = load_example_files()
        
        if example_files:
            st.write("**Ejemplos Precargados:**")
            selected_file = st.selectbox(
                "Selecciona un ejemplo:",
                example_files,
                format_func=lambda x: x.name
            )
            
            if st.button("🔄 Cargar Ejemplo", type="primary"):
                window = load_lob_window(selected_file)
                if window is not None:
                    st.session_state['current_file'] = selected_file
                    st.session_state['window'] = window
                    st.session_state.pop('prediction_done', None)  # Reset predicción
                    st.success(f"✅ Cargado: {selected_file.name}")
                    st.rerun()
        
        st.divider()
        
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
                    st.session_state.pop('prediction_done', None)
                    st.success("✅ Archivo cargado correctamente")
                    st.rerun()
                else:
                    st.error(f"❌ Shape incorrecto: {window.shape}. Esperado: (128, 40)")
            except Exception as e:
                st.error(f"❌ Error al cargar archivo: {e}")
    
    # Main content
    if 'window' not in st.session_state:
        st.info("👈 Selecciona un ejemplo o sube un archivo .npy desde el panel lateral para comenzar.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ejemplos Disponibles", len(example_files))
        with col2:
            st.metric("Shape Esperado", "(128, 40)")
        with col3:
            st.metric("Clases", "3 (DOWN, STAT, UP)")
        
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
    
    # Si hay datos cargados
    window = st.session_state['window']
    file_name = st.session_state.get('current_file', 'archivo cargado')
    
    st.success(f"✅ **Archivo cargado:** {file_name if isinstance(file_name, str) else file_name.name}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🔍 Análisis", "🎯 Predicción", "📈 Resultados"])
    
    with tab1:
        st.header("📊 Visualización de Datos")
        
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
        
        st.subheader("🌡️ Heatmap de la Ventana Temporal")
        st.plotly_chart(plot_lob_heatmap(window), use_container_width=True)
        
        st.subheader("📈 Evolución Temporal de Features Clave")
        st.plotly_chart(plot_temporal_evolution(window), use_container_width=True)
        
        with st.expander("🔢 Ver Datos Numéricos (Primeros 10 timesteps)"):
            df = pd.DataFrame(
                window[:10, :10],
                columns=[f"F{i}" for i in range(10)],
                index=[f"T{i}" for i in range(10)]
            )
            st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
    
    with tab2:
        st.header("🔍 Análisis Estadístico")
        
        st.subheader("📊 Distribución de Valores (Primeras 10 Features)")
        fig_dist = plot_feature_distribution(window)
        st.pyplot(fig_dist)
        
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
            model = load_model()
            
            with st.spinner("🔮 Realizando inferencia..."):
                logits, probs, pred = predict(model, window)
            
            st.session_state['logits'] = logits
            st.session_state['probs'] = probs
            st.session_state['pred'] = pred
            st.session_state['prediction_done'] = True
            
            st.success("✅ Predicción completada!")
            st.balloons()
            st.rerun()
    
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
        
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, {CLASS_COLORS[pred]}22, {CLASS_COLORS[pred]}44); border-radius: 15px; margin: 20px 0;">
            <h1 style="font-size: 60px; margin: 0;">{pred_emoji}</h1>
            <h2 style="margin: 10px 0;">{pred_label}</h2>
            <h3 style="color: {CLASS_COLORS[pred]}; margin: 0;">Confianza: {confidence:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📉 DOWN", f"{probs[0]:.2%}", delta=f"Logit: {logits[0]:.2f}")
        with col2:
            st.metric("➡️ STATIONARY", f"{probs[1]:.2%}", delta=f"Logit: {logits[1]:.2f}")
        with col3:
            st.metric("📈 UP", f"{probs[2]:.2%}", delta=f"Logit: {logits[2]:.2f}")
        
        st.divider()
        
        st.subheader("📊 Distribución de Probabilidades")
        st.plotly_chart(plot_probabilities(probs), use_container_width=True)
        
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
        """)
        
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

if __name__ == "__main__":
    main()

