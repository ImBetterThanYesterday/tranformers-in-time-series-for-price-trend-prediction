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
from src.models.tlob import TLOB

# Configuración
st.set_page_config(
    page_title="TLOB - Predicción de Tendencias",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXAMPLES_DIR = Path("src/data/BTC/individual_examples")
CHECKPOINT_PATH = "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_42/pt/val_loss=0.624_epoch=2.pt"

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
                # IMPORTANTE: Crear alias para módulos antiguos en el checkpoint
                # El checkpoint fue entrenado con imports antiguos (config, models, etc.)
                # Necesitamos crear aliases para que PyTorch pueda deserializar
                import src.config
                import src.config.config
                import src.models
                import src.models.tlob
                import src.models.engine
                import src.utils
                import src.preprocessing
                import src.constants
                
                # Registrar aliases en sys.modules
                sys.modules['config'] = src.config
                sys.modules['config.config'] = src.config.config
                sys.modules['models'] = src.models
                sys.modules['models.tlob'] = src.models.tlob
                sys.modules['models.engine'] = src.models.engine
                sys.modules['utils'] = src.utils
                sys.modules['preprocessing'] = src.preprocessing
                sys.modules['constants'] = src.constants
                
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

def normalize_raw_data(data):
    """
    Aplica Z-score normalization a datos crudos
    
    Args:
        data: numpy array (128, 40) con valores sin normalizar
    
    Returns:
        numpy array normalizado
    """
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    # Columnas pares = precios, impares = volúmenes
    mean_prices = df.iloc[:, 0::2].stack().mean()
    std_prices = df.iloc[:, 0::2].stack().std()
    mean_volumes = df.iloc[:, 1::2].stack().mean()
    std_volumes = df.iloc[:, 1::2].stack().std()
    
    # Normalizar
    for col in df.columns[0::2]:  # Precios
        df[col] = (df[col] - mean_prices) / std_prices
    
    for col in df.columns[1::2]:  # Volúmenes
        df[col] = (df[col] - mean_volumes) / std_volumes
    
    return df.values

def is_data_normalized(data):
    """
    Detecta si los datos ya están normalizados
    
    Heurística: Si mean ≈ 0 y std ≈ 1, probablemente ya está normalizado
    Si mean >> 1000, probablemente son datos crudos (precios BTC)
    """
    mean = np.abs(data.mean())
    std = data.std()
    
    # Si el mean es muy grande, son datos crudos
    if mean > 100:
        return False, "raw"
    # Si mean ≈ 0 y std ≈ 1, ya está normalizado
    elif mean < 1 and 0.5 < std < 2:
        return True, "normalized"
    # No estamos seguros
    else:
        return None, "unknown"

def load_data(filepath):
    """
    Carga archivo .npy y normaliza automáticamente si es necesario
    
    Returns:
        tuple: (data_normalized, data_raw) o (data_normalized, None) si ya está normalizado
    
    Soporta:
    - Archivos .npy ya normalizados
    - Archivos .npy crudos (se normalizan automáticamente)
    - Archivos .csv crudos (se normalizan automáticamente)
    """
    try:
        # Determinar tipo de archivo y extensión
        if hasattr(filepath, 'name'):  # UploadedFile de Streamlit
            file_extension = Path(filepath.name).suffix
            is_uploaded_file = True
        elif isinstance(filepath, str):
            filepath = Path(filepath)
            file_extension = filepath.suffix
            is_uploaded_file = False
        else:  # Path object
            file_extension = filepath.suffix
            is_uploaded_file = False
        
        # Cargar datos según formato
        if file_extension == '.csv':
            import pandas as pd
            df = pd.read_csv(filepath)
            # Si tiene timestamp, eliminarlo
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])
            data = df.values
        elif file_extension == '.npy':
            data = np.load(filepath)
        else:
            st.error(f"❌ Formato no soportado: {file_extension}")
            return None, None
        
        # Verificar shape
        if data.shape != (128, 40):
            st.error(f"❌ Shape incorrecto: {data.shape}. Esperado: (128, 40)")
            return None, None
        
        # Detectar si necesita normalización
        is_normalized, data_type = is_data_normalized(data)
        
        if is_normalized == False:  # Datos crudos
            st.info("🔄 Detectados datos crudos. Aplicando normalización Z-score...")
            data_raw = data.copy()  # Guardar copia de datos crudos
            data_normalized = normalize_raw_data(data)
            st.success(f"✅ Normalización completada (mean={data_normalized.mean():.4f}, std={data_normalized.std():.4f})")
            return data_normalized, data_raw  # Retornar AMBOS
        elif is_normalized == True:  # Ya normalizado
            st.success(f"✅ Datos ya normalizados (mean={data.mean():.4f}, std={data.std():.4f})")
            return data, None  # Solo datos normalizados, sin crudos
        else:  # No estamos seguros
            st.warning(f"⚠️ Tipo de datos ambiguo. Usando tal cual (mean={data.mean():.4f}, std={data.std():.4f})")
            return data, None
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None, None

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
    """Distribuciones de las 40 features"""
    # 8 filas x 5 columnas = 40 features
    fig, axes = plt.subplots(8, 5, figsize=(20, 24))
    fig.suptitle("Distribución de las 40 Features del LOB", fontsize=16, fontweight='bold')
    
    for i in range(40):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        ax.hist(data[:, i], bins=15, alpha=0.7, edgecolor='black', color='steelblue')
        
        # Nombres más descriptivos
        if i < 10:
            label = f'F{i}: ASK Price L{i+1}'
        elif i < 20:
            label = f'F{i}: ASK Vol L{i-9}'
        elif i < 30:
            label = f'F{i}: BID Price L{i-19}'
        else:
            label = f'F{i}: BID Vol L{i-29}'
        
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Valor', fontsize=8)
        ax.set_ylabel('Frecuencia', fontsize=8)
        ax.tick_params(labelsize=7)
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
        
        # Selector de fuente
        example_source = st.radio(
            "Fuente:",
            ["📦 Preprocesados", "📄 Crudos (CSV/NPY)"],
            help="Preprocesados: Ya normalizados. Crudos: Se normalizan automáticamente"
        )
        
        # Cargar según fuente
        if example_source == "📦 Preprocesados":
            examples_dir = Path("src/data/BTC/individual_examples")
            examples = sorted(examples_dir.glob("example_*.npy"))
            source_key = "prep"
        else:
            examples_dir = Path("src/data/BTC/raw_examples")
            # Buscar archivos CSV crudos, NPY crudos y NPY normalizados
            csv_examples = sorted(examples_dir.glob("raw_example_*.csv"))
            npy_raw_examples = sorted(examples_dir.glob("raw_example_*.npy"))
            npy_norm_examples = sorted(examples_dir.glob("normalized_example_*.npy"))
            examples = csv_examples + npy_raw_examples + npy_norm_examples
            source_key = "raw"
        
        if examples:
            st.markdown(f"**{len(examples)} ejemplos:**")
            example_names = [f.name for f in examples]
            selected_name = st.selectbox(
                "Selecciona:",
                example_names,
                key=f"example_selector_{source_key}"
            )
            
            if st.button("🔄 Cargar", type="primary", key=f"load_btn_{source_key}"):
                selected_file = None
                for f in examples:
                    if f.name == selected_name:
                        selected_file = f
                        break
                
                if selected_file:
                    data_normalized, data_raw = load_data(selected_file)
                    if data_normalized is not None:
                        st.session_state['data'] = data_normalized
                        st.session_state['data_raw'] = data_raw  # Guardar datos crudos también
                        st.session_state['filename'] = selected_name
                        st.session_state['source'] = example_source
                        if 'pred_result' in st.session_state:
                            del st.session_state['pred_result']
                        st.success(f"✅ {selected_name}")
                        st.rerun()
        else:
            st.warning(f"⚠️ No hay ejemplos en {examples_dir}")
            if source_key == "raw":
                st.info("💡 Ejecuta:\n`python3 create_raw_examples.py`")
        
        st.divider()
        
        # Upload personalizado
        st.markdown("**O sube archivo:**")
        uploaded = st.file_uploader("Archivo .npy o .csv", type=['npy', 'csv'], key='file_uploader')
        
        if uploaded is not None:
            # Verificar si es un archivo nuevo
            current_filename = uploaded.name
            previous_filename = st.session_state.get('filename', None)
            
            # Si es un archivo diferente, limpiar estado y cargar nuevo
            if current_filename != previous_filename:
                # Limpiar resultados anteriores
                for key in ['prediction', 'probabilities', 'logits']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Cargar nuevo archivo
                data_normalized, data_raw = load_data(uploaded)
                if data_normalized is not None:
                    st.session_state['data'] = data_normalized
                    st.session_state['data_raw'] = data_raw
                    st.session_state['filename'] = current_filename
                    st.session_state['source'] = "📁 Subido"
                    st.success(f"✅ Archivo cargado: {current_filename}")
                    st.rerun()  # Forzar recarga de la interfaz
    
    # Main content
    if 'data' not in st.session_state:
        st.info("👈 Selecciona un ejemplo o sube un archivo .npy")
        
        # Contar ejemplos de ambas fuentes
        prep_examples = len(list(Path("src/data/BTC/individual_examples").glob("example_*.npy")))
        raw_csv_examples = len(list(Path("src/data/BTC/raw_examples").glob("raw_example_*.csv")))
        raw_npy_examples = len(list(Path("src/data/BTC/raw_examples").glob("raw_example_*.npy")))
        norm_npy_examples = len(list(Path("src/data/BTC/raw_examples").glob("normalized_example_*.npy")))
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Preprocesados", prep_examples)
        col2.metric("📄 CSV/NPY Crudos", raw_csv_examples + raw_npy_examples)
        col3.metric("✅ Normalizados", norm_npy_examples)
        col4.metric("Clases", "3")
        
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
    source = st.session_state.get('source', 'Desconocido')
    
    # Header con botón para limpiar
    col_info, col_clear = st.columns([4, 1])
    with col_info:
        st.success(f"✅ **Archivo:** {filename}  |  **Fuente:** {source}")
    with col_clear:
        if st.button("🔄 Nuevo Ejemplo", use_container_width=True):
            # Limpiar todo el estado
            for key in list(st.session_state.keys()):
                if key != 'tlob_model':  # Mantener el modelo cargado
                    del st.session_state[key]
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🔍 Análisis", "🎯 Predicción", "📈 Resultados"])
    
    # TAB 1: Datos
    with tab1:
        st.header("📊 Visualización")
        
        # Mostrar comparación si hay datos crudos disponibles
        data_raw = st.session_state.get('data_raw', None)
        if data_raw is not None:
            st.info("🔄 **Preprocesamiento Aplicado**: Este archivo fue cargado con datos crudos y normalizado automáticamente")
            
            # Comparación lado a lado
            col_raw, col_norm = st.columns(2)
            
            with col_raw:
                st.markdown("### 📥 Datos Originales (Crudos)")
                st.caption("Valores reales del mercado BTC")
                st.metric("Mean", f"{data_raw.mean():.2f}", help="Promedio de precios y volúmenes sin normalizar")
                st.metric("Std", f"{data_raw.std():.2f}", help="Desviación estándar sin normalizar")
                st.metric("Range", f"{data_raw.min():.1f} ~ {data_raw.max():.1f}", help="Rango de valores")
                
                # Mostrar primeras filas de datos crudos
                with st.expander("🔢 Ver primeras 10 filas"):
                    df_raw = pd.DataFrame(
                        data_raw[:10, :10],  # Primeras 10 filas, primeras 10 features
                        columns=[f"F{i}" for i in range(10)],
                        index=[f"T{i}" for i in range(10)]
                    )
                    st.dataframe(df_raw.style.format("{:.2f}"), height=400)
                    st.caption("Precios en USDT, volúmenes en BTC")
            
            with col_norm:
                st.markdown("### ✅ Datos Normalizados")
                st.caption("Z-score: mean≈0, std≈1")
                st.metric("Mean", f"{data.mean():.6f}", help="Promedio después de normalización")
                st.metric("Std", f"{data.std():.6f}", help="Desviación estándar después de normalización")
                st.metric("Range", f"{data.min():.2f} ~ {data.max():.2f}", help="Rango de z-scores")
                
                # Mostrar primeras filas de datos normalizados
                with st.expander("🔢 Ver primeras 10 filas"):
                    df_norm = pd.DataFrame(
                        data[:10, :10],  # Primeras 10 filas, primeras 10 features
                        columns=[f"F{i}" for i in range(10)],
                        index=[f"T{i}" for i in range(10)]
                    )
                    st.dataframe(df_norm.style.format("{:.6f}"), height=400)
                    st.caption("Z-scores normalizados")
            
            st.divider()
        
        # Métricas generales
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
        
        with st.expander("🔢 Datos Numéricos Completos (128×40)"):
            # Mostrar TODOS los 128 timesteps y las 40 features
            df = pd.DataFrame(
                data[:, :40],  # Todos los timesteps, todas las features
                columns=[f"F{i}" for i in range(40)],
                index=[f"T{i}" for i in range(128)]
            )
            st.caption("📌 Matriz completa: 128 timesteps × 40 features")
            st.dataframe(df.style.format("{:.3f}"), height=600)
    
    # TAB 2: Análisis
    with tab2:
        st.header("🔍 Análisis Estadístico")
        
        st.subheader("📊 Distribuciones")
        st.pyplot(plot_distributions(data))
        
        st.subheader("📈 Estadísticas de las 40 Features")
        stats = []
        # Ahora mostramos todas las 40 features
        for i in range(40):
            feat = data[:, i]
            
            # Nombres descriptivos para cada feature
            if i < 10:
                label = f'F{i}: ASK Price L{i+1}'
            elif i < 20:
                label = f'F{i}: ASK Vol L{i-9}'
            elif i < 30:
                label = f'F{i}: BID Price L{i-19}'
            else:
                label = f'F{i}: BID Vol L{i-29}'
            
            stats.append({
                'Feature': label,
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
        }), height=600)
    
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
