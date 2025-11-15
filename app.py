"""
Aplicación de Streamlit para Predicción de Tendencias con TLOB
================================================================
Versión simplificada y robusta - Python 3.12
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sys

# Importar modelo
sys.path.append('.')
from models.tlob import TLOB

# Configuración
st.set_page_config(
    page_title="TLOB - Predicción de Tendencias",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXAMPLES_DIR = Path("data/BTC/individual_examples")
CHECKPOINT_PATH = "data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/val_loss=0.623_epoch=2.pt"

# Mapeo de clases
CLASSES = {0: "DOWN 📉", 1: "STATIONARY ➡️", 2: "UP 📈"}
COLORS = {0: "#ef4444", 1: "#3b82f6", 2: "#10b981"}

# ============================================================================
# FUNCIONES
# ============================================================================

def get_model():
    """Obtiene el modelo desde session_state o lo carga"""
    if 'tlob_model' not in st.session_state:
        with st.spinner("🔄 Cargando modelo TLOB..."):
            try:
                # Configuración del modelo
                model = TLOB(
                    hidden_dim=40,
                    num_layers=4,
                    seq_size=128,
                    num_features=40,
                    num_heads=1,
                    is_sin_emb=True,
                    dataset_type="BTC"
                )
                model.to(DEVICE)
                model.eval()
                
                # Cargar pesos
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
                state_dict = checkpoint["state_dict"]
                
                # Limpiar keys
                clean_dict = {}
                for k, v in state_dict.items():
                    clean_key = k.replace("model.", "") if k.startswith("model.") else k
                    clean_dict[clean_key] = v
                
                model.load_state_dict(clean_dict)
                st.session_state['tlob_model'] = model
                st.success("✅ Modelo cargado")
            except Exception as e:
                st.error(f"❌ Error cargando modelo: {e}")
                return None
    
    return st.session_state.get('tlob_model')

def get_examples():
    """Lista archivos de ejemplo"""
    if not EXAMPLES_DIR.exists():
        return []
    files = list(EXAMPLES_DIR.glob("example_*.npy"))
    return [f for f in files if not f.stem.endswith("_result")]

def load_data(filepath):
    """Carga archivo .npy"""
    try:
        data = np.load(filepath)
        if data.shape != (128, 40):
            st.error(f"❌ Shape incorrecto: {data.shape}. Esperado: (128, 40)")
            return None
        return data
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def run_prediction(model, data):
    """Ejecuta predicción"""
    try:
        x = torch.from_numpy(data[None, :, :]).float().to(DEVICE)
        with torch.no_grad():
            logits = model(x)[0].cpu().numpy()
            probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
            pred = int(np.argmax(probs))
        return logits, probs, pred
    except Exception as e:
        st.error(f"❌ Error en predicción: {e}")
        return None, None, None

def plot_heatmap(data):
    """Heatmap de la ventana LOB"""
    fig = go.Figure(data=go.Heatmap(
        z=data.T,
        x=list(range(128)),
        y=list(range(40)),
        colorscale='RdYlBu_r',
        colorbar=dict(title="Z-score")
    ))
    fig.update_layout(
        title="Heatmap LOB (128 × 40)",
        xaxis_title="Timestep",
        yaxis_title="Feature",
        height=500
    )
    return fig

def plot_timeseries(data):
    """Series temporales de features clave"""
    fig = go.Figure()
    features = [(0, "ASK Price", "#ef4444"), (10, "ASK Vol", "#f97316"),
                (20, "BID Price", "#10b981"), (30, "BID Vol", "#3b82f6")]
    
    for idx, name, color in features:
        fig.add_trace(go.Scatter(
            x=list(range(128)),
            y=data[:, idx],
            mode='lines',
            name=name,
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title="Evolución Temporal",
        xaxis_title="Timestep",
        yaxis_title="Valor",
        height=400,
        hovermode='x unified'
    )
    return fig

def plot_distributions(data):
    """Distribuciones de features"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Distribución de Features (0-9)", fontweight='bold')
    
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.hist(data[:, i], bins=20, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_title(f'F{i}', fontsize=10)
        ax.set_xlabel('Valor')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_probs_bar(probs):
    """Gráfico de barras de probabilidades"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(CLASSES.values()),
            y=probs * 100,
            marker_color=list(COLORS.values()),
            text=[f"{p:.1%}" for p in probs],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Probabilidades",
        yaxis_title="Probabilidad (%)",
        height=400,
        showlegend=False
    )
    return fig

# ============================================================================
# APP PRINCIPAL
# ============================================================================

def main():
    # Header
    st.title("📈 TLOB: Predicción de Tendencias de Precios")
    st.markdown("""
    **Modelo:** Transformer con Dual Attention  
    **Dataset:** Bitcoin LOB (Enero 2023)  
    **Horizonte:** 10 timesteps
    """)
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        st.info(f"""
        **Arquitectura:** TLOB  
        **Parámetros:** 1.1M  
        **Device:** {DEVICE}
        """)
        
        st.subheader("📂 Cargar Datos")
        
        # Ejemplos precargados
        examples = get_examples()
        if examples:
            st.markdown("**Ejemplos:**")
            # Usar nombres en lugar de objetos Path para evitar recursión
            example_names = [f.name for f in examples]
            selected_name = st.selectbox(
                "Selecciona:",
                example_names,
                key="example_selector"
            )
            
            if st.button("🔄 Cargar", type="primary", key="load_btn"):
                # Encontrar el archivo correspondiente
                selected_file = None
                for f in examples:
                    if f.name == selected_name:
                        selected_file = f
                        break
                
                if selected_file:
                    data = load_data(selected_file)
                    if data is not None:
                        st.session_state['data'] = data
                        st.session_state['filename'] = selected_name
                        if 'pred_result' in st.session_state:
                            del st.session_state['pred_result']
                        st.success(f"✅ {selected_name}")
                        st.rerun()
        
        st.divider()
        
        # Upload personalizado
        st.markdown("**O sube archivo:**")
        uploaded = st.file_uploader("Archivo .npy", type=['npy'])
        
        if uploaded is not None:
            # Cargar sin rerun inmediato
            data = load_data(uploaded)
            if data is not None and 'data' not in st.session_state:
                st.session_state['data'] = data
                st.session_state['filename'] = uploaded.name
                st.success("✅ Cargado")
    
    # Main content
    if 'data' not in st.session_state:
        st.info("👈 Selecciona un ejemplo o sube un archivo .npy")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Ejemplos", len(examples))
        col2.metric("Shape", "(128, 40)")
        col3.metric("Clases", "3")
        
        with st.expander("ℹ️ ¿Qué es el LOB?"):
            st.markdown("""
            El **Limit Order Book** registra órdenes de compra/venta pendientes.
            
            **40 features:**
            - 0-9: ASK Prices (10 niveles)
            - 10-19: ASK Volumes
            - 20-29: BID Prices
            - 30-39: BID Volumes
            
            **128 timesteps consecutivos**
            """)
        
        with st.expander("🧠 ¿Cómo funciona TLOB?"):
            st.markdown("""
            **Dual Attention:**
            - **Spatial:** Relaciones entre features
            - **Temporal:** Evolución temporal
            
            **Output:** DOWN / STATIONARY / UP
            """)
        return
    
    # Datos cargados
    data = st.session_state['data']
    filename = st.session_state.get('filename', 'archivo')
    
    st.success(f"✅ **Archivo:** {filename}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🔍 Análisis", "🎯 Predicción", "📈 Resultados"])
    
    # TAB 1: Datos
    with tab1:
        st.header("📊 Visualización")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shape", f"{data.shape[0]} × {data.shape[1]}")
        c2.metric("Mean", f"{data.mean():.3f}")
        c3.metric("Std", f"{data.std():.3f}")
        c4.metric("Range", f"{data.min():.1f} ~ {data.max():.1f}")
        
        st.divider()
        
        st.subheader("🌡️ Heatmap")
        st.plotly_chart(plot_heatmap(data), use_container_width=True)
        
        st.subheader("📈 Series Temporales")
        st.plotly_chart(plot_timeseries(data), use_container_width=True)
        
        with st.expander("🔢 Datos Numéricos (10×10)"):
            df = pd.DataFrame(
                data[:10, :10],
                columns=[f"F{i}" for i in range(10)],
                index=[f"T{i}" for i in range(10)]
            )
            st.dataframe(df.style.format("{:.3f}"))
    
    # TAB 2: Análisis
    with tab2:
        st.header("🔍 Análisis Estadístico")
        
        st.subheader("📊 Distribuciones")
        st.pyplot(plot_distributions(data))
        
        st.subheader("📈 Estadísticas")
        stats = []
        for i in range(min(20, data.shape[1])):
            feat = data[:, i]
            stats.append({
                'Feature': f'F{i}',
                'Mean': feat.mean(),
                'Std': feat.std(),
                'Min': feat.min(),
                'Max': feat.max()
            })
        
        # Formatear solo columnas numéricas
        stats_df = pd.DataFrame(stats)
        st.dataframe(stats_df.style.format({
            'Mean': '{:.3f}',
            'Std': '{:.3f}',
            'Min': '{:.3f}',
            'Max': '{:.3f}'
        }))
    
    # TAB 3: Predicción
    with tab3:
        st.header("🎯 Realizar Predicción")
        
        st.info("""
        El modelo predice la **tendencia** en los próximos **10 timesteps**:
        - 📉 **DOWN:** Precio bajará
        - ➡️ **STATIONARY:** Precio estable
        - 📈 **UP:** Precio subirá
        """)
        
        if st.button("🚀 Ejecutar Predicción", type="primary", use_container_width=True):
            model = get_model()
            if model is not None:
                with st.spinner("🔮 Prediciendo..."):
                    logits, probs, pred = run_prediction(model, data)
                
                if pred is not None:
                    st.session_state['pred_result'] = {
                        'logits': logits,
                        'probs': probs,
                        'pred': pred
                    }
                    st.success("✅ Predicción completada!")
                    st.balloons()
                    st.rerun()
    
    # TAB 4: Resultados
    with tab4:
        st.header("📈 Resultados")
        
        if 'pred_result' not in st.session_state:
            st.warning("⚠️ Ejecuta la predicción primero")
            return
        
        result = st.session_state['pred_result']
        logits = result['logits']
        probs = result['probs']
        pred = result['pred']
        
        label = CLASSES[pred]
        confidence = probs[pred]
        color = COLORS[pred]
        
        # Resultado principal
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, {color}22, {color}44); border-radius: 15px;">
            <h1 style="font-size: 70px; margin: 0;">{label.split()[1]}</h1>
            <h2 style="margin: 10px 0;">{label.split()[0]}</h2>
            <h3 style="color: {color}; margin: 0;">Confianza: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("📉 DOWN", f"{probs[0]:.1%}", f"Logit: {logits[0]:.2f}")
        c2.metric("➡️ STATIONARY", f"{probs[1]:.1%}", f"Logit: {logits[1]:.2f}")
        c3.metric("📈 UP", f"{probs[2]:.1%}", f"Logit: {logits[2]:.2f}")
        
        st.divider()
        
        # Gráfico
        st.subheader("📊 Distribución de Probabilidades")
        st.plotly_chart(plot_probs_bar(probs), use_container_width=True)
        
        # Interpretación
        st.subheader("💡 Interpretación")
        
        if confidence > 0.90:
            nivel = "**MUY ALTA** :green[⭐⭐⭐]"
        elif confidence > 0.75:
            nivel = "**ALTA** :blue[⭐⭐]"
        elif confidence > 0.60:
            nivel = "**MODERADA** :orange[⭐]"
        else:
            nivel = "**BAJA**"
        
        st.markdown(f"""
        Confianza {nivel} ({confidence:.1%})
        
        → El precio tendrá tendencia **{label.split()[0]}** en los próximos **10 timesteps**.
        """)
        
        with st.expander("🔬 Detalles Técnicos"):
            st.code(f"""
Shape entrada: {data.shape}
Mean: {data.mean():.4f}
Std: {data.std():.4f}

Logits:
  DOWN:       {logits[0]:>8.4f}
  STATIONARY: {logits[1]:>8.4f}
  UP:         {logits[2]:>8.4f}

Probabilidades (post-softmax):
  DOWN:       {probs[0]:>7.1%}
  STATIONARY: {probs[1]:>7.1%}
  UP:         {probs[2]:>7.1%}

Predicción: {label.split()[0]} (clase {pred})
            """)

if __name__ == "__main__":
    main()
