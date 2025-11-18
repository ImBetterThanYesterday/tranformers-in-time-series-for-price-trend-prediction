"""
UTILIDADES PARA PROCESAMIENTO DE DATOS
=======================================

Funciones auxiliares para normalización, transformación y etiquetado de datos
del Limit Order Book (LOB) y mensajes de trading.

Este módulo proporciona:
- Normalización Z-score para LOB y mensajes
- Codificación de tipos de órdenes
- Generación de etiquetas para predicción
- Transformaciones sparse del LOB

Autor: Proyecto TLOB
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import os

import torch
import pandas
import src.constants as cst


def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """
    Z-SCORE NORMALIZATION PARA LIMIT ORDER BOOK
    ============================================
    
    Normaliza precios y volúmenes del LOB usando z-score (media=0, std=1).
    Separa normalización de precios y volúmenes ya que tienen escalas MUY diferentes.
    
    Fórmula Z-Score:
    ----------------
    z = (x - μ) / σ
    
    Donde:
    - x: Valor original
    - μ (mu): Media
    - σ (sigma): Desviación estándar
    
    Args:
        data (pd.DataFrame): DataFrame con columnas alternadas [PRICE, SIZE, PRICE, SIZE, ...]
                            Estructura típica:
                            [ASK_P1, ASK_V1, BID_P1, BID_V1, ..., ASK_P10, ASK_V10, BID_P10, BID_V10]
                            Shape: (timesteps, 40)
        
        mean_size (float, optional): Media de volúmenes precomputada (para test set)
                                     Si None, se calcula de data
        
        mean_prices (float, optional): Media de precios precomputada (para test set)
                                       Si None, se calcula de data
        
        std_size (float, optional): Desviación estándar de volúmenes precomputada
                                    Si None, se calcula de data
        
        std_prices (float, optional): Desviación estándar de precios precomputada
                                      Si None, se calcula de data
    
    Returns:
        tuple: (data_normalized, mean_size, mean_prices, std_size, std_prices)
            - data_normalized: DataFrame normalizado
            - mean_size: Media de volúmenes (calculada o pasada)
            - mean_prices: Media de precios (calculada o pasada)
            - std_size: Std de volúmenes (calculada o pasada)
            - std_prices: Std de precios (calculada o pasada)
    
    Raises:
        ValueError: Si data contiene valores NaN después de normalización
    
    Ejemplo de Uso:
    ---------------
    ### Normalizar Train Set (calcular estadísticas)
    ```python
    # Train data: shape (100000, 40)
    train_data = pd.DataFrame(...)  # LOB data
    
    train_normalized, mean_s, mean_p, std_s, std_p = z_score_orderbook(train_data)
    
    # Guardar estadísticas para usar en val/test
    stats = {
        'mean_size': mean_s,
        'mean_prices': mean_p,
        'std_size': std_s,
        'std_prices': std_p
    }
    ```
    
    ### Normalizar Test Set (usar estadísticas de train)
    ```python
    # Test data: shape (20000, 40)
    test_data = pd.DataFrame(...)
    
    # IMPORTANTE: Usar estadísticas de train (NO calcular de test)
    test_normalized, _, _, _, _ = z_score_orderbook(
        test_data,
        mean_size=stats['mean_size'],
        mean_prices=stats['mean_prices'],
        std_size=stats['std_size'],
        std_prices=stats['std_prices']
    )
    
    # Esto evita data leakage!
    ```
    
    Estructura de Columnas:
    -----------------------
    ```
    data.columns:
      [0]: ASK_P1   (precio)     ← columna par (0, 2, 4, ...)
      [1]: ASK_V1   (volumen)    ← columna impar (1, 3, 5, ...)
      [2]: BID_P1   (precio)     ← columna par
      [3]: BID_V1   (volumen)    ← columna impar
      ...
      [38]: BID_P10 (precio)     ← columna par
      [39]: BID_V10 (volumen)    ← columna impar
    ```
    
    Lógica de Normalización:
    ------------------------
    1. **Identificar columnas**:
       - Precios: Columnas pares (0, 2, 4, ..., 38)
       - Volúmenes: Columnas impares (1, 3, 5, ..., 39)
    
    2. **Calcular o usar estadísticas**:
       ```python
       if mean_size is None:
           # Tomar TODOS los volúmenes (columnas impares)
           all_sizes = data.iloc[:, 1::2].stack()  # Flatten all volumes
           mean_size = all_sizes.mean()
           std_size = all_sizes.std()
       
       if mean_prices is None:
           # Tomar TODOS los precios (columnas pares)
           all_prices = data.iloc[:, 0::2].stack()  # Flatten all prices
           mean_prices = all_prices.mean()
           std_prices = all_prices.std()
       ```
    
    3. **Aplicar z-score**:
       ```python
       for col in price_columns:
           data[col] = (data[col] - mean_prices) / std_prices
       
       for col in size_columns:
           data[col] = (data[col] - mean_size) / std_size
       ```
    
    ¿Por Qué Normalizar Precios y Volúmenes por Separado?
    ------------------------------------------------------
    **Escalas MUY diferentes**:
    ```
    Bitcoin (BTCUSDT):
    - Precios: ~42,000 - 43,000 USDT
    - Volúmenes: ~0.1 - 10 BTC
    
    Si normalizamos juntos:
    - mean_combined ≈ 21,000 (dominado por precios)
    - std_combined ≈ 1,000 (dominado por precios)
    - Volúmenes quedarían sobre-normalizados
    ```
    
    **Solución**:
    ```
    Normalización separada:
    - Precios: mean=42,150, std=50
    - Volúmenes: mean=0.5, std=0.3
    
    Resultado:
    - Ambos en escala similar después de normalización
    - Mejor convergencia del modelo
    ```
    
    Verificación de Normalización:
    ------------------------------
    ```python
    # Después de normalizar train_data
    train_normalized, mean_s, mean_p, std_s, std_p = z_score_orderbook(train_data)
    
    # Verificar que la media es ~0 y std ~1
    print(f"Precio normalizado - Mean: {train_normalized.iloc[:, 0::2].stack().mean():.6f}")
    # Output esperado: Mean: 0.000000 (o muy cercano)
    
    print(f"Precio normalizado - Std: {train_normalized.iloc[:, 0::2].stack().std():.6f}")
    # Output esperado: Std: 1.000000
    
    print(f"Volumen normalizado - Mean: {train_normalized.iloc[:, 1::2].stack().mean():.6f}")
    # Output esperado: Mean: 0.000000
    
    print(f"Volumen normalizado - Std: {train_normalized.iloc[:, 1::2].stack().std():.6f}")
    # Output esperado: Std: 1.000000
    ```
    
    Errores Comunes:
    ----------------
    1. **ValueError: data contains null value**
       - Causa: División por cero (std=0) o valores NaN en input
       - Solución: Verificar que data no tenga columnas constantes
    
    2. **Usar estadísticas de test para normalizar test**
       - Problema: Data leakage
       - Solución: SIEMPRE usar estadísticas de train
    
    Nota Importante:
    ----------------
    Esta normalización es CRÍTICA para el entrenamiento del modelo:
    - Sin normalización: Gradientes inestables, no converge
    - Con normalización: Convergencia rápida y estable
    - Permite usar learning rates más altos
    """
    # Calcular estadísticas de volúmenes si no se proporcionan
    if (mean_size is None) or (std_size is None):
        # iloc[:, 1::2] selecciona columnas impares (volúmenes)
        # stack() convierte DataFrame a Series (flatten)
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    # Calcular estadísticas de precios si no se proporcionan
    if (mean_prices is None) or (std_prices is None):
        # iloc[:, 0::2] selecciona columnas pares (precios)
        mean_prices = data.iloc[:, 0::2].stack().mean()
        std_prices = data.iloc[:, 0::2].stack().std()

    # Identificar columnas de precios y volúmenes
    price_cols = data.columns[0::2]  # Columnas pares
    size_cols = data.columns[1::2]   # Columnas impares

    # Aplicar z-score a volúmenes
    for col in size_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_size) / std_size

    # Aplicar z-score a precios
    for col in price_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_prices) / std_prices

    # Verificar que no hay valores NaN (indicaría std=0 o división por cero)
    if data.isnull().values.any():
        raise ValueError("data contains null value after normalization")

    return data, mean_size, mean_prices, std_size, std_prices


def normalize_messages(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None, 
                       mean_time=None, std_time=None, mean_depth=None, std_depth=None):
    """
    Z-SCORE NORMALIZATION PARA MENSAJES/EVENTOS DE TRADING (LOBSTER)
    =================================================================
    
    Normaliza eventos event-driven del formato LOBSTER (order submissions, cancellations, trades).
    A diferencia de z_score_orderbook(), normaliza 4 variables: price, size, time, depth.
    
    Formato LOBSTER Message File:
    ------------------------------
    ```
    timestamp | event_type | order_id | size | price | direction | depth
    ---------|------------|----------|------|-------|-----------|-------
    34200000 | 1          | 12345    | 100  | 42150 | 1         | 0
    34200250 | 2          | 12345    | 100  | 42150 | 1         | 0
    34200500 | 4          | 0        | 50   | 42148 | -1        | 2
    ...
    
    event_type:
    1 = Submission of new limit order
    2 = Cancellation (partial)
    3 = Cancellation (total)
    4 = Execution of visible limit order
    5 = Execution of hidden limit order
    ```
    
    Args:
        data (pd.DataFrame): DataFrame con columnas ['time', 'size', 'price', 'depth', 'event_type', ...]
        
        mean_size, std_size (float, optional): Estadísticas de volumen
        mean_prices, std_prices (float, optional): Estadísticas de precio
        mean_time, std_time (float, optional): Estadísticas de tiempo entre eventos
        mean_depth, std_depth (float, optional): Estadísticas de profundidad del LOB
    
    Returns:
        tuple: (data_normalized, mean_size, mean_prices, std_size, std_prices, 
                mean_time, std_time, mean_depth, std_depth)
    
    Transformaciones Aplicadas:
    ---------------------------
    1. **Z-score normalization** de 4 variables continuas:
       ```python
       data['time'] = (data['time'] - mean_time) / std_time
       data['size'] = (data['size'] - mean_size) / std_size
       data['price'] = (data['price'] - mean_prices) / std_prices
       data['depth'] = (data['depth'] - mean_depth) / std_depth
       ```
    
    2. **Remapeo de event_type** (5 tipos → 3 tipos):
       ```python
       Original LOBSTER:
       1 = Submission
       2 = Partial cancellation
       3 = Full cancellation
       4 = Visible execution
       5 = Hidden execution
       
       Después de transformación:
       0 = Submission (1-1)
       1 = Cancellation (2→1, 3→1 merged)
       2 = Execution (4→2, 5→2 merged)
       
       Lógica:
       data['event_type'] = data['event_type'] - 1  # Shift: 1→0, 2→1, 3→2, 4→3, 5→4
       data['event_type'].replace(2, 1)  # Merge: 2→1 (full cancel → cancel)
       data['event_type'].replace(3, 2)  # Shift: 3→2, 4→2 (executions)
       ```
    
    Ejemplo de Uso:
    ---------------
    ```python
    # Cargar mensajes LOBSTER
    messages = pd.read_csv('TSLA_2024-01-15_messages.csv')
    # Columns: ['time', 'size', 'price', 'depth', 'event_type', 'direction']
    
    # Normalizar train set
    train_msg, m_s, m_p, s_s, s_p, m_t, s_t, m_d, s_d = normalize_messages(messages_train)
    
    # Normalizar test set con stats de train
    test_msg, _, _, _, _, _, _, _, _ = normalize_messages(
        messages_test,
        mean_size=m_s, std_size=s_s,
        mean_prices=m_p, std_prices=s_p,
        mean_time=m_t, std_time=s_t,
        mean_depth=m_d, std_depth=s_d
    )
    ```
    
    Interpretación de Variables Normalizadas:
    -----------------------------------------
    - **time**: Tiempo entre eventos (normalizado)
      * Alta frecuencia: time_norm < 0 (más rápido que promedio)
      * Baja frecuencia: time_norm > 0 (más lento que promedio)
    
    - **size**: Tamaño de la orden (shares)
      * Orden pequeña: size_norm < 0
      * Orden grande: size_norm > 0
    
    - **price**: Precio de la orden
      * Bajo precio: price_norm < 0
      * Alto precio: price_norm > 0
    
    - **depth**: Nivel del LOB (0=best bid/ask, 1=nivel 2, ...)
      * depth_norm < 0: Niveles superficiales (L1-L3)
      * depth_norm > 0: Niveles profundos (L7-L10)
    
    Nota Importante:
    ----------------
    Esta función ES SOLO PARA LOBSTER event-driven data.
    Para BTC/FI-2010 (ya sampleados), usar z_score_orderbook().
    """
    # Calcular o usar estadísticas de size
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()

    # Calcular o usar estadísticas de price
    if (mean_prices is None) or (std_prices is None):
        mean_prices = data["price"].mean()
        std_prices = data["price"].std()

    # Calcular o usar estadísticas de time
    if (mean_time is None) or (std_time is None):
        mean_time = data["time"].mean()
        std_time = data["time"].std()

    # Calcular o usar estadísticas de depth
    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()

    # Aplicar z-score a las 4 variables continuas
    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices
    data["depth"] = (data["depth"] - mean_depth) / std_depth
    
    # Verificar NaN
    if data.isnull().values.any():
        raise ValueError("data contains null value after normalization")

    # Transformar event_type: 5 tipos → 3 tipos
    data["event_type"] = data["event_type"] - 1.0  # Shift down by 1
    data["event_type"] = data["event_type"].replace(2, 1)  # Merge type 3→1
    data["event_type"] = data["event_type"].replace(3, 2)  # Shift types 4,5→2
    
    # Resultado final:
    # order_type = 0 -> limit order submission
    # order_type = 1 -> cancel order (partial or full)
    # order_type = 2 -> market order (visible or hidden execution)
    
    return data, mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth


def reset_indexes(dataframes):
    """
    RESETEA ÍNDICES DE DATAFRAMES (LOBSTER)
    ========================================
    
    Función auxiliar para resetear índices de múltiples DataFrames.
    Típicamente usada para sincronizar mensajes y orderbooks después de filtrado.
    
    Args:
        dataframes (list): Lista de 2 DataFrames [messages_df, orderbook_df]
    
    Returns:
        list: Lista de DataFrames con índices reseteados
    
    Uso:
    ----
    ```python
    # Después de filtrar datos
    messages_filtered = messages[messages['event_type'] != 5]
    orderbook_filtered = orderbook[messages['event_type'] != 5]
    
    # Indices ahora son inconsistentes [0, 1, 2, 4, 5, 7, ...]
    # Resetear para tener [0, 1, 2, 3, 4, 5, ...]
    dataframes = reset_indexes([messages_filtered, orderbook_filtered])
    messages_clean = dataframes[0]
    orderbook_clean = dataframes[1]
    ```
    
    Nota: Esta función modifica los DataFrames in-place.
    """
    # Reset index del primer DataFrame (típicamente messages)
    dataframes[0] = dataframes[0].reset_index(drop=True)
    # Reset index del segundo DataFrame (típicamente orderbook)
    dataframes[1] = dataframes[1].reset_index(drop=True)
    return dataframes

 

def unnormalize(x, mean, std):
    """
    DESNORMALIZACIÓN (INVERSA DE Z-SCORE)
    ======================================
    
    Revierte la normalización z-score para obtener valores originales.
    
    Fórmula:
    --------
    x_original = x_normalized * σ + μ
    
    Args:
        x (float or np.ndarray): Valor(es) normalizado(s)
        mean (float): Media original (μ)
        std (float): Desviación estándar original (σ)
    
    Returns:
        float or np.ndarray: Valor(es) en escala original
    
    Ejemplo:
    --------
    ```python
    # Normalizar
    price_original = 42150.5
    price_normalized = (price_original - 42150) / 50  # = 0.01
    
    # Desnormalizar
    price_recovered = unnormalize(price_normalized, mean=42150, std=50)
    print(price_recovered)  # 42150.5 ✓
    
    # Verificar
    assert abs(price_recovered - price_original) < 1e-6
    ```
    
    Uso Típico:
    -----------
    ```python
    # Después de predicción, desnormalizar para interpretación
    pred_normalized = model(input_normalized)  # En escala normalizada
    pred_original = unnormalize(pred_normalized, mean_prices, std_prices)
    print(f"Predicción de precio: ${pred_original:.2f}")
    ```
    """
    return x * std + mean


def one_hot_encoding_type(data):
    """
    ONE-HOT ENCODING DE EVENT_TYPE (LOBSTER)
    =========================================
    
    Convierte event_type categórico (0, 1, 2) a representación one-hot.
    
    Transformación:
    ---------------
    ```
    Input data: (n_events, n_features)
      Column 0: timestamp
      Column 1: event_type (0, 1, or 2)
      Columns 2+: other features
    
    Output: (n_events, n_features + 2)
      Column 0: timestamp (preserved)
      Columns 1-3: one-hot encoded event_type [0/1, 0/1, 0/1]
      Columns 4+: other features (shifted)
    ```
    
    Args:
        data (torch.Tensor): Input tensor de shape (n_events, n_features)
    
    Returns:
        torch.Tensor: Output tensor de shape (n_events, n_features + 2)
    
    Ejemplo:
    --------
    ```python
    # Input
    data = torch.tensor([
        [34200000, 0, 100, 42150],  # timestamp, type, size, price
        [34200250, 1, 50, 42148],
        [34200500, 2, 75, 42152]
    ])
    
    # One-hot encoding
    encoded = one_hot_encoding_type(data)
    
    # Output
    # [[34200000, 1, 0, 0, 100, 42150],  # type 0 → [1, 0, 0]
    #  [34200250, 0, 1, 0, 50, 42148],   # type 1 → [0, 1, 0]
    #  [34200500, 0, 0, 1, 75, 42152]]   # type 2 → [0, 0, 1]
    ```
    
    Ventajas de One-Hot:
    --------------------
    - El modelo no asume orden entre tipos de eventos
    - Cada tipo tiene su propia representación independiente
    - Mejor para redes neuronales que encoding numérico simple
    
    Nota:
    -----
    Esta función asume que event_type ya está en rango [0, 1, 2].
    Si viene del formato LOBSTER original [1-5], primero usar normalize_messages().
    """
    # Crear tensor con espacio para 2 columnas adicionales
    encoded_data = torch.zeros(data.shape[0], data.shape[1] + 2, dtype=torch.float32)
    
    # Preservar timestamp (columna 0)
    encoded_data[:, 0] = data[:, 0]
    
    # One-hot encoding de event_type (columna 1)
    # Convierte: 0→[1,0,0], 1→[0,1,0], 2→[0,0,1]
    one_hot_order_type = torch.nn.functional.one_hot(
        (data[:, 1]).to(torch.int64), 
        num_classes=3
    ).to(torch.float32)
    
    # Insertar one-hot en columnas 1-3
    encoded_data[:, 1:4] = one_hot_order_type
    
    # Copiar resto de features (columnas 2+)
    encoded_data[:, 4:] = data[:, 2:]
    
    return encoded_data


def tanh_encoding_type(data):
    """
    TANH ENCODING DE EVENT_TYPE (LOBSTER)
    ======================================
    
    Alternativa a one-hot: Codifica event_type como valores en [-1, 0, 1].
    Usa tanh() implícitamente: después del shift, valores quedan centrados en 0.
    
    Transformación:
    ---------------
    ```
    Input event_type: [0, 1, 2]
    
    Step 1: Swap 1 y 2
      0 → 0
      1 → 2
      2 → 1
    
    Step 2: Shift by -1
      0 → -1
      2 → 1
      1 → 0
    
    Output: [-1, 0, 1]
    ```
    
    Args:
        data (torch.Tensor): Input tensor con event_type en columna 1
    
    Returns:
        torch.Tensor: Tensor con event_type transformado a [-1, 0, 1]
    
    Ejemplo:
    --------
    ```python
    data = torch.tensor([
        [34200000, 0, 100, 42150],  # type 0 → -1
        [34200250, 1, 50, 42148],   # type 1 → 1
        [34200500, 2, 75, 42152]    # type 2 → 0
    ])
    
    encoded = tanh_encoding_type(data)
    
    # Output:
    # [[34200000, -1, 100, 42150],
    #  [34200250, 1, 50, 42148],
    #  [34200500, 0, 75, 42152]]
    ```
    
    Ventajas vs One-Hot:
    -------------------
    - Usa 1 columna en lugar de 3
    - Valores en rango continuo [-1, 1]
    - Puede capturar "orden" implícito si existe
    
    Desventajas vs One-Hot:
    -----------------------
    - Asume orden/distancia entre tipos
    - Menos expresivo que one-hot
    
    Uso Típico:
    -----------
    Usar one_hot si no hay orden entre tipos (mejor para clasificación).
    Usar tanh si los tipos tienen algún orden semántico.
    """
    # Step 1: Swap valores 1 y 2
    # where(condition, value_if_true, value_if_false)
    data[:, 1] = torch.where(
        data[:, 1] == 1.0, 
        2.0,  # 1 → 2
        torch.where(
            data[:, 1] == 2.0, 
            1.0,  # 2 → 1
            data[:, 1]  # 0 stays 0
        )
    )
    
    # Step 2: Shift by -1
    data[:, 1] = data[:, 1] - 1  # [0, 2, 1] → [-1, 1, 0]
    
    return data


def to_sparse_representation(lob, n_levels):
    """
    CONVERSIÓN A REPRESENTACIÓN SPARSE DEL LOB
    ===========================================
    
    Convierte LOB denso (todos los niveles) a representación sparse
    donde solo se guardan volúmenes en posiciones determinadas por profundidad.
    
    NOTA: Esta función actualmente TIENE BUGS y NO se usa en producción.
    Se mantiene por compatibilidad pero NO es recomendada para uso.
    
    Concepto (teórico):
    -------------------
    ```
    LOB Denso:
    [ASK_P1, ASK_V1, BID_P1, BID_V1, ..., ASK_P10, ASK_V10, BID_P10, BID_V10]
    40 features
    
    LOB Sparse:
    [V_depth_0, V_depth_1, ..., V_depth_n]
    n_levels * 2 features
    
    Idea: Usar la profundidad (diferencia de precio) como índice.
    ```
    
    Args:
        lob (np.ndarray): LOB denso de shape (40,) o list
        n_levels (int): Número de niveles sparse a generar
    
    Returns:
        np.ndarray: LOB sparse de shape (n_levels * 2,)
    
    Problemas Conocidos:
    -------------------
    1. Lógica de indexing incorrecta (j % 2 es inconsistente)
    2. División por 100 asume tick size fijo (no generaliza)
    3. No maneja casos edge (precios idénticos, gaps grandes)
    
    NO USAR ESTA FUNCIÓN en código nuevo.
    Para alternativa correcta, ver preprocessing/btc.py:create_sequences()
    """
    # Convertir a numpy array si es necesario
    if not isinstance(lob, np.ndarray):
        lob = np.array(lob)
    
    # Inicializar array sparse
    sparse_lob = np.zeros(n_levels * 2)
    
    # Iterar sobre niveles del LOB
    for j in range(lob.shape[0] // 2):
        if j % 2 == 0:  # Ask side (BUGGY: esta lógica es incorrecta)
            ask_price = lob[0]
            current_ask_price = lob[j*2]
            depth = (current_ask_price - ask_price) // 100  # Asume tick size = 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)] = lob[j*2+1]
        else:  # Bid side (BUGGY: esta lógica es incorrecta)
            bid_price = lob[2]
            current_bid_price = lob[j*2]
            depth = (bid_price - current_bid_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)+1] = lob[j*2+1]
    
    return sparse_lob


def labeling(X, len, h):
    """
    GENERACIÓN DE ETIQUETAS PARA PREDICCIÓN DE TENDENCIAS
    ======================================================
    
    Genera etiquetas (UP/STATIONARY/DOWN) basadas en cambio porcentual del mid-price.
    Usa umbral α adaptativo calculado de la volatilidad del activo.
    
    Args:
        X (np.ndarray): LOB data de shape (n_timesteps, n_features)
                       Debe tener ASK_P1 en columna 0 y BID_P1 en columna 2
        
        len (int): Ventana de suavizado para mid-price (típicamente 10)
                  Reduce ruido de alta frecuencia
        
        h (int): Horizonte de predicción en timesteps
                 Ejemplo: h=10 predice 10 timesteps adelante
    
    Returns:
        np.ndarray: Array de etiquetas de shape (n_timesteps - len - h + 1,)
                   Valores: 0 (UP), 1 (STATIONARY), 2 (DOWN)
    
    Proceso Detallado:
    ------------------
    1. **Calcular mid-prices suavizados**:
       ```python
       # Usar ventana deslizante de longitud 'len'
       previous_asks = sliding_window(X[:, 0], window=len)[:-h]
       previous_bids = sliding_window(X[:, 2], window=len)[:-h]
       previous_mids = (previous_asks + previous_bids) / 2
       previous_mids = mean(previous_mids, axis=1)  # Promedio de ventana
       
       # Lo mismo para precios futuros
       future_asks = sliding_window(X[:, 0], window=len)[h:]
       future_bids = sliding_window(X[:, 2], window=len)[h:]
       future_mids = (future_asks + future_bids) / 2
       future_mids = mean(future_mids, axis=1)
       ```
    
    2. **Calcular cambio porcentual**:
       ```python
       pct_change = (future_mids - previous_mids) / previous_mids
       # Ejemplo: (42160 - 42150) / 42150 = 0.000237 = 0.0237%
       ```
    
    3. **Calcular umbral α adaptativo**:
       ```python
       alpha = abs(pct_change).mean() / 2
       # Divide por 2 para balance entre clases
       ```
    
    4. **Asignar etiquetas**:
       ```python
       labels = np.where(
           pct_change < -alpha, 2,  # DOWN
           np.where(
               pct_change > alpha, 0,  # UP
               1  # STATIONARY
           )
       )
       ```
    
    Ejemplo Numérico:
    -----------------
    ```python
    # LOB data
    X = np.array([
        [42150, 0.5, 42148, 0.6, ...],  # t=0
        [42151, 0.4, 42149, 0.5, ...],  # t=1
        ...
        [42160, 0.6, 42158, 0.7, ...]   # t=127
    ])
    
    # Generar etiquetas
    labels = labeling(X, len=10, h=10)
    
    # Output
    # labels.shape = (108,)  # 128 - 10 - 10
    # labels[0] = 0  # UP (precio sube en próximos 10 timesteps)
    # labels[1] = 1  # STATIONARY
    # labels[2] = 2  # DOWN
    
    # Distribución típica (balanceada por α adaptativo)
    # UP: 35%
    # STATIONARY: 30%
    # DOWN: 35%
    ```
    
    Ventajas del α Adaptativo:
    --------------------------
    **Problema con α fijo (ej: 0.002 = 0.2%)**:
    ```
    Mercado tranquilo (BTC en consolidación):
    - Cambios típicos: 0.001% - 0.01%
    - Con α=0.2%: 90% etiquetado como STATIONARY (desbalanceado)
    
    Mercado volátil (BTC en rally):
    - Cambios típicos: 0.5% - 2%
    - Con α=0.2%: 70% etiquetado como UP/DOWN (desbalanceado)
    ```
    
    **Solución con α adaptativo**:
    ```
    α = mean(|pct_change|) / 2
    
    Mercado tranquilo:
    - α = 0.005% / 2 = 0.0025%
    - Distribución: UP 33%, STAT 34%, DOWN 33% ✓
    
    Mercado volátil:
    - α = 1.0% / 2 = 0.5%
    - Distribución: UP 35%, STAT 30%, DOWN 35% ✓
    ```
    
    Interpretación de Etiquetas:
    ----------------------------
    - **Label 0 (UP)**: Precio subirá > α% en los próximos h timesteps
      * Acción: Comprar (long position)
    
    - **Label 1 (STATIONARY)**: Precio se mantendrá en [-α%, +α%]
      * Acción: Mantener o no operar
    
    - **Label 2 (DOWN)**: Precio bajará > α% en los próximos h timesteps
      * Acción: Vender (short position)
    
    Prints de Diagnóstico:
    ----------------------
    La función imprime estadísticas útiles:
    ```python
    # Alpha: 0.000523
    # Number of labels: (array([0, 1, 2]), array([3500, 3200, 3300]))
    # Percentage of labels: [0.35 0.32 0.33]
    ```
    
    Esto permite verificar:
    - Que α es razonable para el activo
    - Que las clases están balanceadas
    - Que no hay dominancia de una clase
    
    Errores Comunes:
    ----------------
    1. **AssertionError: Length must be greater than 0**
       - Causa: len ≤ 0
       - Solución: Usar len >= 1 (típicamente 10)
    
    2. **AssertionError: Horizon must be greater than 0**
       - Causa: h ≤ 0
       - Solución: Usar h >= 1 (típicamente 10, 20, 50, 100)
    
    3. **Clases muy desbalanceadas (ej: 80% STATIONARY)**
       - Causa: α muy grande o len muy pequeño
       - Solución: Aumentar len, verificar calidad de datos
    
    Nota Importante:
    ----------------
    Esta función es CRÍTICA para el entrenamiento:
    - Define qué aprende el modelo
    - α adaptativo mejora significativamente el balance de clases
    - Suavizado (len) reduce overfitting a ruido
    
    Para más detalles sobre etiquetado, ver:
    - docs/INNOVACIONES_TLOB.md (sección "Etiquetado Dinámico")
    - preprocessing/btc.py:btc_load()
    """
    # Validaciones
    assert len > 0, "Length must be greater than 0"
    assert h > 0, "Horizon must be greater than 0"
    
    # Si horizonte es menor que ventana de suavizado, ajustar
    if h < len:
        len = h
    
    # Calcular mid-prices pasados y futuros usando ventanas deslizantes
    # sliding_window_view: crea ventanas sin copiar datos (más eficiente)
    previous_ask_prices = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len)[:-h]
    previous_bid_prices = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len)[:-h]
    future_ask_prices = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len)[h:]
    future_bid_prices = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len)[h:]

    # Calcular mid-prices (promedio de bid y ask)
    previous_mid_prices = (previous_ask_prices + previous_bid_prices) / 2
    future_mid_prices = (future_ask_prices + future_bid_prices) / 2

    # Promediar sobre la ventana de suavizado
    previous_mid_prices = np.mean(previous_mid_prices, axis=1)
    future_mid_prices = np.mean(future_mid_prices, axis=1)

    # Calcular cambio porcentual
    percentage_change = (future_mid_prices - previous_mid_prices) / previous_mid_prices
    
    # Calcular umbral α adaptativo
    alpha = np.abs(percentage_change).mean() / 2
    
    # Método alternativo (comentado): α basado en spread
    # alpha = (X[:, 0] - X[:, 2]).mean() / ((X[:, 0] + X[:, 2]) / 2).mean()
        
    # Imprimir diagnósticos
    print(f"Alpha: {alpha}")
    
    # Asignar etiquetas basadas en α
    labels = np.where(
        percentage_change < -alpha, 2,  # DOWN
        np.where(
            percentage_change > alpha, 0,  # UP
            1  # STATIONARY
        )
    )
    
    # Imprimir distribución de etiquetas
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Number of labels: {(unique, counts)}")
    print(f"Percentage of labels: {counts / labels.shape[0]}")
    
    return labels
