# ============================================================================
# Dockerfile para TLOB - Predicción de Tendencias con Transformers
# ============================================================================
# 
# Este Dockerfile crea un contenedor que ejecuta la aplicación Streamlit
# para inferencia del modelo TLOB sobre series temporales del LOB.
#
# Uso:
#   docker build -t tlob-app .
#   docker run -p 8501:8501 tlob-app
#

# Imagen base: Python 3.12 slim (liviana y moderna)
FROM python:3.12-slim

# Metadata del contenedor
LABEL maintainer="TLOB Team"
LABEL description="TLOB Transformer Model for Price Trend Prediction"
LABEL version="2.0"

# Configurar directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
# - gcc, g++: compiladores para algunas librerías de Python
# - git: para clonar repositorios si es necesario
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requisitos
# (Se hace primero para aprovechar caché de Docker si no cambian)
COPY requirements.txt .

# Instalar dependencias de Python
# --no-cache-dir: no guardar caché para reducir tamaño de imagen
RUN pip install --no-cache-dir -r requirements.txt

# Instalar dependencias adicionales para Streamlit
# Streamlit 1.39+ es compatible con Python 3.12
RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    plotly==5.24.0

# Copiar código fuente de la aplicación
COPY . .

# Crear directorio para datos si no existe
RUN mkdir -p data/BTC/individual_examples data/checkpoints/TLOB

# Exponer puerto 8501 (puerto por defecto de Streamlit)
EXPOSE 8501

# Configurar variables de entorno para Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Healthcheck: verificar que la app está respondiendo
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando para ejecutar la aplicación
# Se ejecuta al hacer docker run
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


