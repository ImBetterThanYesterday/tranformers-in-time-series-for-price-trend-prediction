# 🚀 Quick Start - TLOB Streamlit App

## ⚡ Inicio Rápido (3 pasos)

### Opción A: Con Docker (Recomendado) 🐳

```bash
# 1. Construir imagen
docker build -t tlob-app .

# 2. Ejecutar contenedor
docker run -p 8501:8501 tlob-app

# 3. Abrir navegador
# → http://localhost:8501
```

---

### Opción B: Local (Sin Docker) 💻

```bash
# 1. Instalar dependencias
pip install -r requirements_streamlit.txt

# 2. Ejecutar app
streamlit run app.py

# 3. Abrir navegador (se abre automáticamente)
# → http://localhost:8501
```

---

## 📋 Requisitos Previos

- **Python 3.9+**
- **Docker** (solo para Opción A)
- **4GB RAM** mínimo

---

## 🎮 Cómo Usar

1. **Seleccionar ejemplo:** Panel lateral → Elegir `example_1.npy` a `example_5.npy`
2. **Explorar datos:** Pestaña "📊 Datos" → Ver heatmap y series temporales
3. **Predecir:** Pestaña "🎯 Predicción" → Clic en "Ejecutar Predicción"
4. **Ver resultados:** Pestaña "📈 Resultados" → Ver predicción y probabilidades

---

## 🐛 Troubleshooting

### Error: "Module 'streamlit' not found"
```bash
pip install streamlit plotly seaborn
```

### Error: "Port 8501 already in use"
```bash
# Cambiar puerto
streamlit run app.py --server.port 8502
```

### Error: "CUDA not available"
```
No es problema, el modelo funciona en CPU.
```

---

## 📚 Documentación Completa

Ver **README_DEPLOY.md** para instrucciones detalladas.

---

**¡Listo para predecir tendencias! 📈**


