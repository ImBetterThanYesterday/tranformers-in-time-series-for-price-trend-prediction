#!/bin/bash
# ============================================================================
# Script para ejecutar la aplicación TLOB
# ============================================================================

echo "========================================="
echo "TLOB - Predicción de Tendencias"
echo "========================================="
echo ""

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado"
    echo "Instala Python 3.9+ desde https://www.python.org/"
    exit 1
fi

echo "✓ Python encontrado: $(python3 --version)"
echo ""

# Verificar si requirements.txt existe
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt no encontrado"
    exit 1
fi

# Preguntar si desea crear entorno virtual
read -p "¿Crear entorno virtual? (recomendado) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        echo "Creando entorno virtual..."
        python3 -m venv venv
    fi
    
    echo "Activando entorno virtual..."
    source venv/bin/activate
    echo "✓ Entorno virtual activado"
    echo ""
fi

# Instalar dependencias
echo "Instalando dependencias..."
pip install -q -r requirements.txt
pip install -q streamlit plotly seaborn

echo "✓ Dependencias instaladas"
echo ""

# Ejecutar aplicación
echo "========================================="
echo "Iniciando aplicación Streamlit..."
echo "========================================="
echo ""
echo "La aplicación se abrirá en:"
echo "  http://localhost:8501"
echo ""
echo "Presiona Ctrl+C para detener"
echo ""

streamlit run app.py


