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

# Checkpoints disponibles para diferentes horizontes
CHECKPOINTS = {
    10: "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_42/pt/val_loss=0.624_epoch=2.pt",
    20: "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_20_seed_42/pt/val_loss=0.822_epoch=1.pt",
    50: "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_50_seed_42/pt/val_loss=0.962_epoch=0.pt",
    100: "src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_100_seed_42/pt/val_loss=1.013_epoch=0.pt"
}

# Mapeo de clases (según utils_data.py línea 158)
# percentage_change < -alpha → label=2 (DOWN)
# percentage_change > alpha  → label=0 (UP)
# else                      → label=1 (STATIONARY)
CLASSES = {0: "UP 📈", 1: "STATIONARY ➡️", 2: "DOWN 📉"}
COLORS = {0: "#10b981", 1: "#3b82f6", 2: "#ef4444"}

# ============================================================================
# FUNCIONES
# ============================================================================

def get_model(horizon=10):
    """
    CARGA DE PESOS DEL MODELO
    =========================
    
    Esta función carga el modelo TLOB pre-entrenado desde un checkpoint
    específico según el horizonte de predicción seleccionado.
    
    Args:
        horizon (int): Horizonte de predicción en timesteps (10, 20, 50, 100)
    
    Returns:
        TLOB: Modelo cargado con pesos pre-entrenados, listo para inferencia en modo eval()
    
    Proceso de Carga:
    -----------------
    1. Verificar si el modelo ya está en session_state (caché)
    2. Crear aliases de módulos antiguos (compatibilidad con checkpoints entrenados)
    3. Instanciar arquitectura TLOB con hiperparámetros correctos para BTC
    4. Cargar checkpoint .pt correspondiente al horizonte desde disco
    5. Limpiar keys del state_dict (remover prefijo 'model.' si existe)
    6. Cargar pesos en el modelo usando load_state_dict()
    7. Configurar modelo en modo evaluación (.eval())
    8. Guardar en session_state para reutilización sin recarga
    
    Nota Importante - Aliases de Módulos:
    -------------------------------------
    Los checkpoints fueron entrenados con la estructura antigua del repositorio
    (imports como 'config', 'models', etc. sin prefijo 'src.'). PyTorch serializa
    los imports usados durante el entrenamiento en el checkpoint.
    
    Para deserializar correctamente, necesitamos crear aliases en sys.modules:
        'config' → 'src.config'
        'models' → 'src.models'
        'utils' → 'src.utils'
        etc.
    
    Esto permite a torch.load() encontrar los módulos correctos sin modificar
    los checkpoints entrenados originalmente.
    
    Checkpoints Disponibles:
    ------------------------
    - Horizonte 10: src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_42/pt/val_loss=0.624_epoch=2.pt
    - Horizonte 20: src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_20_seed_42/pt/val_loss=0.822_epoch=1.pt
    - Horizonte 50: src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_50_seed_42/pt/val_loss=0.962_epoch=0.pt
    - Horizonte 100: src/data/checkpoints/TLOB/BTC_seq_size_128_horizon_100_seed_42/pt/val_loss=1.013_epoch=0.pt
    
    Hiperparámetros del Modelo TLOB para BTC:
    ------------------------------------------
    - hidden_dim: 40 (dimensión de embeddings)
    - num_layers: 4 (número de pares de TransformerLayers)
    - seq_size: 128 (longitud de secuencia temporal)
    - num_features: 40 (número de features del LOB)
    - num_heads: 1 (cabezas de atención por layer)
    - is_sin_emb: True (usar positional encoding sinusoidal)
    - dataset_type: "BTC"
    
    Ejemplo de Uso:
    ---------------
    ```python
    # Cargar modelo para horizonte de 10 timesteps
    model = get_model(horizon=10)
    
    # Primera llamada: carga desde disco (~2-3 segundos)
    # Llamadas subsecuentes: recupera desde session_state (instantáneo)
    ```
    """
    # Crear una clave única para cada horizonte
    model_key = f'tlob_model_h{horizon}'
    
    if model_key not in st.session_state or st.session_state.get('current_horizon') != horizon:
        with st.spinner(f"🔄 Cargando modelo TLOB (horizonte {horizon})..."):
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
                
                # Cargar pesos del checkpoint correspondiente al horizonte
                checkpoint_path = CHECKPOINTS[horizon]
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                state_dict = checkpoint["state_dict"]
                
                # Limpiar keys
                clean_dict = {}
                for k, v in state_dict.items():
                    clean_key = k.replace("model.", "") if k.startswith("model.") else k
                    clean_dict[clean_key] = v
                
                model.load_state_dict(clean_dict)
                st.session_state[model_key] = model
                st.session_state['current_horizon'] = horizon
                st.success(f"✅ Modelo cargado (horizonte {horizon} timesteps)")
            except Exception as e:
                st.error(f"❌ Error cargando modelo: {e}")
                return None
    
    return st.session_state.get(model_key)

def get_examples():
    """Lista archivos de ejemplo"""
    if not EXAMPLES_DIR.exists():
        return []
    files = list(EXAMPLES_DIR.glob("example_*.npy"))
    return [f for f in files if not f.stem.endswith("_result")]

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
        # Simular el cálculo de labels para obtener alpha
        if horizon >= len(mid_prices):
            len_smooth = min(horizon, len_smooth)
        
        # Calcular cambio porcentual entre ventanas
        if len(mid_prices) > horizon + len_smooth:
            previous_prices = mid_prices[:-horizon]
            future_prices = mid_prices[horizon:]
            percentage_change = (future_prices - previous_prices) / previous_prices
            alpha = np.abs(percentage_change).mean() / 2
        else:
            # Si no hay suficientes datos, usar un alpha por defecto
            alpha = 0.002  # 0.2%
    
    return alpha

def normalize_raw_data(data):
    """
    PREPROCESAMIENTO: Z-SCORE NORMALIZATION
    ========================================
    
    Normaliza datos crudos del LOB usando Z-score para cada tipo de feature.
    Esta normalización es CRÍTICA porque el modelo TLOB fue entrenado con datos
    normalizados (mean≈0, std≈1).
    
    Args:
        data (np.array): Datos crudos shape (128, 40)
                        - Columnas pares (0, 2, 4, ..., 38): Precios (en USDT)
                        - Columnas impares (1, 3, 5, ..., 39): Volúmenes (en BTC)
    
    Returns:
        np.array: Datos normalizados (mean≈0, std≈1) shape (128, 40)
    
    Proceso de Normalización:
    -------------------------
    1. Convertir a DataFrame de pandas para manipulación flexible
    2. Separar columnas por tipo:
       - Precios: columnas pares (0::2)
       - Volúmenes: columnas impares (1::2)
    3. Calcular estadísticas globales por tipo:
       - mean_prices, std_prices: de TODAS las columnas pares
       - mean_volumes, std_volumes: de TODAS las columnas impares
    4. Aplicar Z-score a cada columna según su tipo:
       - Precios: (x - mean_prices) / std_prices
       - Volúmenes: (x - mean_volumes) / std_volumes
    5. Retornar como numpy array
    
    Razón del Preprocesamiento:
    ---------------------------
    El modelo TLOB fue entrenado con datos normalizados. La normalización:
    - **Estabiliza el entrenamiento**: Evita gradientes explosivos
    - **Generalización**: Permite que el modelo funcione con diferentes rangos de precios
    - **Convergencia**: Facilita la optimización (gradientes más estables)
    - **Comparabilidad**: Precios y volúmenes en escalas similares
    
    Ejemplo con Datos Reales de BTC:
    ---------------------------------
    Entrada (datos crudos):
    ```
    ASK_P1 = 42150.5 USDT, ASK_V1 = 0.524 BTC
    BID_P1 = 42148.2 USDT, BID_V1 = 0.631 BTC
    ...
    ```
    
    Después de normalización:
    ```
    ASK_P1 = 0.765, ASK_V1 = 0.909
    BID_P1 = -1.490, BID_V1 = -1.091
    ...
    ```
    
    Estadísticas resultantes:
    ```
    mean_normalized ≈ 0.0001 (casi 0)
    std_normalized ≈ 0.998 (casi 1)
    ```
    
    Nota Importante:
    ----------------
    Esta función normaliza GLOBALMENTE (usando estadísticas de toda la ventana).
    Es diferente de la normalización por feature individual. El approach global
    es el usado durante el entrenamiento del modelo TLOB.
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
    """
    GENERACIÓN DE INFERENCIA
    =========================
    
    Ejecuta la predicción del modelo TLOB sobre una ventana LOB.
    Esta función maneja el forward pass y corrige el orden del softmax.
    
    Args:
        model (TLOB): Modelo cargado y en modo eval()
        data (np.array): Datos normalizados shape (128, 40)
    
    Returns:
        tuple: (logits, probs, pred)
            - logits (np.array): Salidas raw del modelo antes de softmax, shape (3,)
                                Orden: [UP, STATIONARY, DOWN]
            - probs (np.array): Probabilidades después de softmax, shape (3,)
                               Orden: [UP, STATIONARY, DOWN]
                               Suman a 1.0
            - pred (int): Clase predicha (0=UP, 1=STATIONARY, 2=DOWN)
                         Resultado de argmax(probs)
    
    Proceso de Inferencia:
    ----------------------
    1. **Preparar input**:
       - Convertir numpy array a torch tensor
       - Agregar dimensión de batch: (128,40) → (1,128,40)
       - Mover a device (CPU o GPU)
    
    2. **Forward pass sin gradientes**:
       - Usar torch.no_grad() para ahorrar memoria
       - Ejecutar model(x) para obtener logits raw
       - Shape de salida: (1, 3) → extraer [0] para obtener (3,)
    
    3. **Aplicar softmax**:
       - Convertir logits a probabilidades
       - softmax(x_i) = exp(x_i) / sum(exp(x_j))
       - Resultado: 3 valores que suman 1.0
    
    4. **⚠️ INVERSIÓN CRÍTICA DEL ORDEN**:
       - El modelo da orden inverso a las etiquetas
       - Reordenar para coincidir con CLASSES mapping
    
    5. **Obtener predicción final**:
       - argmax(probs) → clase con mayor probabilidad
    
    ⚠️ IMPORTANTE: INVERSIÓN DEL ORDEN DEL SOFTMAX
    -----------------------------------------------
    
    ### El Problema:
    
    Durante el entrenamiento, las etiquetas se asignaron así (utils_data.py línea 158):
    ```python
    labels = np.where(
        percentage_change < -alpha, 2,  # DOWN
        np.where(percentage_change > alpha, 0, 1)  # UP, STATIONARY
    )
    ```
    
    Por lo tanto:
    - **Etiqueta 0** = UP 📈 (cambio > +alpha)
    - **Etiqueta 1** = STATIONARY ➡️ (cambio dentro de ±alpha)
    - **Etiqueta 2** = DOWN 📉 (cambio < -alpha)
    
    ### El Modelo (PyTorch):
    
    Sin embargo, el modelo de PyTorch aprende a dar salidas en orden NUMÉRICO
    de las etiquetas durante el entrenamiento, resultando en:
    
    ```
    softmax_raw[0] = probabilidad de etiqueta 2 (DOWN)
    softmax_raw[1] = probabilidad de etiqueta 1 (STATIONARY)
    softmax_raw[2] = probabilidad de etiqueta 0 (UP)
    ```
    
    Esto es [DOWN, STATIONARY, UP] en lugar de [UP, STATIONARY, DOWN]
    
    ### La Solución:
    
    Invertimos el orden para que coincida con el mapeo de etiquetas:
    ```python
    logits = [logits_raw[2], logits_raw[1], logits_raw[0]]
    probs = [probs_raw[2], probs_raw[1], probs_raw[0]]
    ```
    
    Ahora:
    - **probs[0]** = probabilidad de UP (etiqueta 0) ✓
    - **probs[1]** = probabilidad de STATIONARY (etiqueta 1) ✓
    - **probs[2]** = probabilidad de DOWN (etiqueta 2) ✓
    
    Esto asegura que `CLASSES[pred]` retorne la etiqueta correcta.
    
    Ejemplo de Inferencia:
    ----------------------
    ```python
    # Input: ventana LOB normalizada (128, 40)
    logits, probs, pred = run_prediction(model, data)
    
    # Output ejemplo:
    # logits = [2.341, 0.156, -1.892]  # [UP, STATIONARY, DOWN]
    # probs = [0.852, 0.123, 0.025]     # [85.2%, 12.3%, 2.5%]
    # pred = 0                           # Clase UP
    # CLASSES[pred] = "UP 📈"
    ```
    
    Verificación:
    -------------
    Para verificar que el orden es correcto, se puede comparar con predicciones
    de ejemplos conocidos del conjunto de validación.
    """
    try:
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
    """
    APLICACIÓN STREAMLIT - INTERFAZ PRINCIPAL
    ==========================================
    
    Interfaz web interactiva para inferencia con TLOB.
    Permite cargar datos LOB, visualizarlos, configurar parámetros y ejecutar predicciones.
    
    Estructura de la Aplicación:
    ----------------------------
    
    ### 1. CONFIGURACIÓN INICIAL (líneas 567-587)
    ```python
    st.set_page_config(
        page_title="TLOB - Predicción de Tendencias",
        page_icon="📈",
        layout="wide",  # Usa todo el ancho de la pantalla
        initial_sidebar_state="expanded"
    )
    ```
    
    Configura:
    - Título de la pestaña del navegador
    - Icono (emoji)
    - Layout ancho para mejor visualización
    - Sidebar expandido por defecto
    
    ### 2. SIDEBAR - CARGA DE DATOS (líneas 588-662)
    
    **Selector de Fuente:**
    - 📦 Preprocesados: Archivos .npy ya normalizados (mean≈0, std≈1)
    - 📄 Crudos: Archivos .csv o .npy sin normalizar
    
    **Flujo de Carga:**
    1. Usuario selecciona fuente (radio buttons)
    2. Se buscan archivos en el directorio correspondiente
    3. Usuario selecciona un archivo (selectbox)
    4. Click en "Cargar" ejecuta load_data()
    5. Datos se guardan en st.session_state['data']
    6. Datos raw (si existen) se guardan en st.session_state['data_raw']
    
    **File Uploader:**
    - Permite subir archivos personalizados
    - Soporta .npy y .csv
    - Detección automática de normalización
    
    ### 3. EXPANDER CON INFORMACIÓN (líneas 588-600)
    
    Explica el orden del etiquetado y softmax:
    - Cómo se asignaron las etiquetas durante entrenamiento
    - Por qué el modelo da orden inverso
    - Cómo la app corrige automáticamente
    
    ### 4. TABS DE LA INTERFAZ (líneas 740-1089)
    
    #### **TAB 1 - 📊 Datos** (líneas 745-814)
    
    **Visualizaciones:**
    - Métricas: Shape, Mean, Std, Range
    - Heatmap interactivo (plotly): 128×40 matriz de valores
    - Series temporales: Evolución de ASK/BID prices y volumes
    - Tabla completa de datos (expandible)
    
    **Comparación Raw vs Normalized:**
    Si se cargaron datos crudos:
    - Lado izquierdo: Datos originales (precios en USDT, volúmenes en BTC)
    - Lado derecho: Datos normalizados (z-scores)
    - Muestra primeras 10 filas de cada tipo
    
    **Código Clave:**
    ```python
    data = st.session_state['data']  # Datos normalizados
    data_raw = st.session_state.get('data_raw', None)  # Datos crudos (opcional)
    
    if data_raw is not None:
        # Mostrar comparación lado a lado
        col_raw, col_norm = st.columns(2)
        with col_raw:
            st.metric("Mean", f"{data_raw.mean():.2f}")
        with col_norm:
            st.metric("Mean", f"{data.mean():.6f}")
    ```
    
    #### **TAB 2 - 🔍 Análisis** (líneas 817-855)
    
    **Distribuciones:**
    - 40 histogramas (8 filas × 5 columnas)
    - Cada feature tiene su propio histograma
    - Nombres descriptivos: "F0: ASK Price L1", "F10: ASK Vol L1", etc.
    
    **Estadísticas Descriptivas:**
    - Tabla con Mean, Std, Min, Max para cada feature
    - Formato numérico consistente (3 decimales)
    - 600px de altura para scroll cómodo
    
    **Código Clave:**
    ```python
    stats = []
    for i in range(40):
        feat = data[:, i]
        if i < 10:
            label = f'F{i}: ASK Price L{i+1}'
        elif i < 20:
            label = f'F{i}: ASK Vol L{i-9}'
        # ... etc
        stats.append({'Feature': label, 'Mean': feat.mean(), ...})
    ```
    
    #### **TAB 3 - 🎯 Predicción** (líneas 858-977)
    
    **Selectores de Configuración:**
    
    1. **Horizonte de Predicción** (selectbox):
       - Opciones: 10, 20, 50, 100 timesteps
       - Cada horizonte usa un modelo diferente
       - Horizonte 10 ≈ 0.5 segundos
       - Horizonte 100 ≈ 5 segundos
    
    2. **Tipo de Umbral (Alpha)** (radio buttons):
       - 📊 Normal: alpha = mean(|% change|) / 2
         * Basado en volatilidad natural
         * Usado durante entrenamiento
       - 💹 Spread: alpha = mean(ask - bid) / mid_price
         * Basado en costos de transacción
         * Más restrictivo (solo cambios > spread son rentables)
    
    **Info Box Explicativo:**
    - Explica cómo se etiquetan las tendencias
    - Muestra configuración actual (horizonte, umbral)
    - Nota sobre inversión automática del softmax
    
    **Botón "Ejecutar Predicción":**
    1. Verifica que hay datos cargados
    2. Carga el modelo para el horizonte seleccionado
    3. Calcula alpha (dinámico si hay datos raw, teórico si no)
    4. Ejecuta run_prediction()
    5. Guarda resultados en session_state['pred_result']
    6. Muestra balloons 🎈 y recarga la app
    
    **Código Clave:**
    ```python
    if st.button("🚀 Ejecutar Predicción", type="primary"):
        model = get_model(horizon=horizon)
        
        # Calcular alpha
        data_for_alpha = st.session_state.get('data_raw', None)
        if data_for_alpha is not None:
            alpha = calculate_alpha(data_for_alpha, horizon, use_spread)
        else:
            alpha = 0.005 if use_spread else 0.002  # Teórico
        
        # Predicción
        logits, probs, pred = run_prediction(model, data)
        
        # Guardar resultados
        st.session_state['pred_result'] = {
            'logits': logits,
            'probs': probs,
            'pred': pred
        }
        st.rerun()
    ```
    
    #### **TAB 4 - 📈 Resultados** (líneas 980-1089)
    
    **Resultado Principal:**
    - Visualización grande centrada con emoji
    - Color de fondo según predicción:
      * Verde: UP 📈
      * Azul: STATIONARY ➡️
      * Rojo: DOWN 📉
    - Confianza en porcentaje
    
    **Info Box de Configuración:**
    - Horizonte usado
    - Tipo de umbral
    - Alpha calculado o teórico
    - Interpretación del alpha
    
    **Métricas de Probabilidades:**
    - 3 columnas con st.metric()
    - UP, STATIONARY, DOWN
    - Muestra probabilidad y logit
    
    **Gráfico de Barras:**
    - Plotly bar chart interactivo
    - Colores según clase (verde/azul/rojo)
    - Valores en porcentaje
    
    **Interpretación Automática:**
    - Nivel de confianza:
      * >90%: MUY ALTA ⭐⭐⭐
      * >75%: ALTA ⭐⭐
      * >60%: MODERADA ⭐
      * <60%: BAJA
    - Explicación textual de la predicción
    
    **Expander "Detalles Técnicos":**
    - Shape de entrada
    - Mean y Std
    - Logits raw
    - Probabilidades post-softmax
    - Predicción final
    
    ### 5. GESTIÓN DE ESTADO CON SESSION STATE
    
    **Variables Clave:**
    ```python
    st.session_state = {
        'data': np.array,              # Datos normalizados (128, 40)
        'data_raw': np.array,          # Datos crudos (128, 40) [opcional]
        'filename': str,               # Nombre del archivo cargado
        'source': str,                 # Fuente: "Preprocesados" o "Crudos"
        
        'tlob_model_h10': TLOB,        # Modelo para horizonte 10
        'tlob_model_h20': TLOB,        # Modelo para horizonte 20
        'tlob_model_h50': TLOB,        # Modelo para horizonte 50
        'tlob_model_h100': TLOB,       # Modelo para horizonte 100
        'current_horizon': int,        # Horizonte actual
        
        'pred_result': dict,           # Resultados de predicción
            # {'logits': array, 'probs': array, 'pred': int}
        'horizon': int,                # Horizonte usado en predicción
        'use_spread': bool,            # Tipo de umbral usado
        'alpha': float,                # Alpha calculado
        'alpha_type': str,             # "Normal" o "Spread"
        'alpha_calculated': bool,      # True si dinámico, False si teórico
    }
    ```
    
    **Ventajas de Session State:**
    - No recargar modelo en cada interacción
    - Mantener datos cargados entre tabs
    - Preservar resultados de predicciones
    - Experiencia de usuario fluida sin pérdida de estado
    
    ### 6. FLUJO DE USUARIO TÍPICO
    
    ```
    1. Usuario abre app en navegador (http://localhost:8501)
    2. Sidebar: Selecciona fuente de datos (Preprocesados o Crudos)
    3. Sidebar: Selecciona archivo y click "Cargar"
    4. load_data() detecta si necesita normalización
    5. Datos se guardan en session_state
    6. TAB 1: Usuario visualiza datos (heatmap, series temporales)
    7. TAB 2: Usuario explora distribuciones y estadísticas
    8. TAB 3: Usuario configura horizonte y umbral
    9. TAB 3: Click en "Ejecutar Predicción"
       a. get_model(horizon) carga modelo apropiado (o usa caché)
       b. calculate_alpha() determina umbral
       c. run_prediction() genera inferencia
       d. Resultados se guardan en session_state
   10. TAB 4: Usuario ve predicción final con visualizaciones
   11. Usuario puede cargar otro ejemplo o probar diferentes configuraciones
    ```
    
    ### 7. BOTÓN "NUEVO EJEMPLO"
    
    Ubicado en la parte superior (línea 736):
    - Limpia todo el estado excepto modelos cargados
    - Permite empezar de cero sin recargar la app
    - Mantiene modelos en memoria para rapidez
    
    **Código:**
    ```python
    if st.button("🔄 Nuevo Ejemplo"):
        for key in list(st.session_state.keys()):
            if not key.startswith('tlob_model'):  # Mantener modelos
                del st.session_state[key]
        st.rerun()
    ```
    
    ### 8. MANEJO DE ERRORES Y VALIDACIONES
    
    **Verificaciones de Estado:**
    - Si no hay datos: Mostrar info y métricas de archivos disponibles
    - Si no hay predicción: Mostrar warning en TAB 4
    - Si falla carga de modelo: Mostrar error con traceback
    
    **Validaciones de Datos:**
    - Shape correcto (128, 40)
    - Formato soportado (.npy, .csv)
    - Detección de normalización (evitar doble normalización)
    
    ### 9. OPTIMIZACIONES DE PERFORMANCE
    
    **Caché de Modelos:**
    - Modelos se cargan una sola vez por horizonte
    - Se guardan en session_state con clave única
    - No se recargan en cada predicción
    
    **Lazy Loading:**
    - Modelo se carga solo cuando se ejecuta predicción
    - No se cargan todos los modelos al inicio
    
    **Session State Persistence:**
    - Datos persisten entre tabs
    - No hay recarga de archivos al cambiar de tab
    
    ### 10. INTEGRACIÓN VISUAL
    
    **Plotly (Gráficos Interactivos):**
    - Heatmap: Hover muestra valores exactos
    - Series temporales: Zoom, pan, hover
    - Barras de probabilidades: Interactivo
    
    **Matplotlib (Distribuciones):**
    - 40 histogramas en grid 8×5
    - Exportable como imagen
    
    **Streamlit Components:**
    - Metrics: Números grandes con deltas
    - Expanders: Info adicional colapsable
    - Columns: Layout responsive
    - Spinner: Feedback visual durante carga
    """
    # Header
    st.title("📈 TLOB: Predicción de Tendencias de Precios")
    st.markdown("""
    **Modelo:** Transformer con Dual Attention  
    **Dataset:** Bitcoin LOB (Enero 2023)
    
    Predice tendencias de precios (UP/DOWN/STATIONARY) usando datos de Limit Order Book.
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
        
        with st.expander("ℹ️ Sobre el etiquetado y salida del modelo"):
            st.markdown("""
            **Etiquetado durante el entrenamiento:**
            
            En `utils_data.py` línea 158:
            ```python
            labels = np.where(
                percentage_change < -alpha, 2,  # DOWN
                np.where(percentage_change > alpha, 0, 1)  # UP, STATIONARY
            )
            ```
            
            **Etiquetas (ground truth):**
            - **Clase 0**: UP 📈 (cambio > +alpha)
            - **Clase 1**: STATIONARY ➡️ (cambio dentro de ±alpha)
            - **Clase 2**: DOWN 📉 (cambio < -alpha)
            
            **⚠️ IMPORTANTE: Orden del softmax**
            
            El modelo de PyTorch da salidas en **ORDEN INVERSO**:
            ```
            softmax[0] = probabilidad de DOWN (etiqueta 2)
            softmax[1] = probabilidad de STATIONARY (etiqueta 1)
            softmax[2] = probabilidad de UP (etiqueta 0)
            ```
            
            La app **invierte automáticamente** las probabilidades para mostrarlas correctamente.
            """)
        
        st.divider()
        
        # ============ CARGAR DATOS ============
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
        
        # ============ CONFIGURACIÓN DE PREDICCIÓN ============
        st.subheader("⚙️ Parámetros de Predicción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Selector de horizonte
            horizon = st.selectbox(
                "**Horizonte de Predicción:**",
                options=[10, 20, 50, 100],
                index=0,
                help="""
                Número de timesteps hacia el futuro para predecir:
                - 10: ~0.5 segundos
                - 20: ~1 segundo  
                - 50: ~2.5 segundos
                - 100: ~5 segundos
                """
            )
        
        with col2:
            # Selector de tipo de umbral
            threshold_type = st.radio(
                "**Tipo de Umbral (Alpha):**",
                options=["📊 Normal", "💹 Spread"],
                index=0,
                help="""
                **Normal:** alpha = mean(|% change|) / 2
                - Basado en la volatilidad natural del activo
                - Usado durante el entrenamiento del modelo
                
                **Spread:** alpha = mean(ask - bid) / mid_price  
                - Basado en costos de transacción reales
                - Más restrictivo: solo cambios > spread son rentables
                - Útil para evaluar estrategias de trading reales
                
                ⚠️ NOTA: El modelo fue entrenado con umbral Normal.
                Cambiar a Spread es solo para análisis de rentabilidad.
                """
            )
        
        use_spread = (threshold_type == "💹 Spread")
        threshold_name = "Spread" if use_spread else "Normal"
        
        st.divider()
        
        st.info(f"""
        El modelo predice la **tendencia** en los próximos **{horizon} timesteps**:
        
        **Etiquetado durante entrenamiento** (utils_data.py):
        - Si `cambio_porcentual > +alpha` → **UP 📈** (clase 0)
        - Si `cambio_porcentual < -alpha` → **DOWN 📉** (clase 2)
        - Si está dentro de ±alpha → **STATIONARY ➡️** (clase 1)
        
        **Nota:** El modelo da softmax en orden inverso [DOWN, STABLE, UP], 
        pero la app lo invierte automáticamente para mostrar correctamente.
        
        **Configuración actual:**
        - Horizonte: {horizon} timesteps
        - Umbral: {threshold_name}
        """)
        
        if st.button("🚀 Ejecutar Predicción", type="primary", use_container_width=True):
            # Verificar que hay datos cargados
            if 'data' not in st.session_state:
                st.error("⚠️ Primero debes cargar datos en la pestaña 'Datos'")
                st.stop()
            
            data = st.session_state['data']
            
            model = get_model(horizon=horizon)
            if model is not None:
                with st.spinner("🔮 Prediciendo..."):
                    # Calcular alpha según configuración
                    # Usar datos raw si existen
                    data_for_alpha = st.session_state.get('data_raw', None)
                    
                    if data_for_alpha is not None:
                        # Tenemos datos raw, calcular alpha dinámicamente
                        alpha = calculate_alpha(data_for_alpha, horizon=horizon, use_spread=use_spread)
                        alpha_calculated = True
                    else:
                        # Datos preprocesados, usar alpha teórico por defecto
                        if use_spread:
                            alpha = 0.005  # 0.5% (spread típico de Bitcoin)
                        else:
                            alpha = 0.002  # 0.2% (volatilidad típica)
                        alpha_calculated = False
                        st.info(f"""
                        ℹ️ Usando datos preprocesados. Alpha no calculado dinámicamente.
                        
                        Usando alpha teórico por defecto:
                        - Normal: 0.2% (volatilidad típica)
                        - Spread: 0.5% (spread típico)
                        
                        Para cálculo dinámico de alpha, usa datos crudos (CSV/NPY sin procesar).
                        """)
                    
                    # Guardar configuración en session state
                    st.session_state['horizon'] = horizon
                    st.session_state['use_spread'] = use_spread
                    st.session_state['alpha'] = alpha
                    st.session_state['alpha_type'] = threshold_name
                    st.session_state['alpha_calculated'] = alpha_calculated
                    
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
        
        # Configuración de la predicción
        if 'alpha' in st.session_state:
            alpha = st.session_state['alpha']
            alpha_type = st.session_state.get('alpha_type', 'Normal')
            horizon = st.session_state.get('horizon', 10)
            alpha_calculated = st.session_state.get('alpha_calculated', True)
            
            if alpha_calculated:
                alpha_label = f"**Alpha calculado:** {alpha:.4f} ({alpha*100:.2f}%)"
                alpha_note = "Calculado dinámicamente desde datos crudos"
            else:
                alpha_label = f"**Alpha teórico:** {alpha:.4f} ({alpha*100:.2f}%)"
                alpha_note = "Valor por defecto (datos preprocesados)"
            
            st.info(f"""
            **Configuración de la predicción:**
            - **Horizonte:** {horizon} timesteps
            - **Tipo de umbral:** {alpha_type}
            - {alpha_label}
            - *{alpha_note}*
            
            Los cambios de precio menores a ±{alpha*100:.2f}% se consideran **STATIONARY**.
            """)
        
        st.divider()
        
        # Métricas (Orden correcto: 0=UP, 1=STATIONARY, 2=DOWN)
        c1, c2, c3 = st.columns(3)
        c1.metric("📈 UP", f"{probs[0]:.1%}", f"Logit: {logits[0]:.2f}")
        c2.metric("➡️ STATIONARY", f"{probs[1]:.1%}", f"Logit: {logits[1]:.2f}")
        c3.metric("📉 DOWN", f"{probs[2]:.1%}", f"Logit: {logits[2]:.2f}")
        
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
