# 🔧 Troubleshooting - TLOB Streamlit App

## Problemas Comunes y Soluciones

### 1. RecursionError al cargar el modelo

**Error:**
```
RecursionError: maximum recursion depth exceeded in comparison
```

**Causa:** Streamlit's `@st.cache_resource` tiene problemas al hashear objetos complejos de PyTorch.

**Solución:**
- ✅ **Ya solucionado** en `app.py`
- Usamos `session_state` en lugar de `@st.cache_resource`
- El modelo se carga una sola vez y se mantiene en memoria

---

### 2. ModuleNotFoundError: No module named 'streamlit'

**Solución:**
```bash
pip install -r requirements_streamlit.txt
```

O instalar manualmente:
```bash
pip install streamlit plotly seaborn torch
```

---

### 3. Port 8501 already in use

**Solución A:** Cambiar de puerto
```bash
streamlit run app.py --server.port 8502
```

**Solución B:** Matar el proceso existente
```bash
# En Mac/Linux
lsof -ti:8501 | xargs kill -9

# En Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

### 4. Docker: permission denied

**Error:**
```
permission denied while trying to connect to the Docker daemon socket
```

**Solución:**
```bash
# En Mac/Linux
sudo docker-compose up

# O añadir usuario al grupo docker
sudo usermod -aG docker $USER
newgrp docker
```

---

### 5. Checkpoint not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/checkpoints/TLOB/...'
```

**Solución:**
Verificar que el checkpoint existe:
```bash
ls -la data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1/pt/
```

Si no existe, verificar la ruta correcta o re-entrenar el modelo.

---

### 6. CUDA not available

**Mensaje:**
```
CUDA not available, using CPU
```

**Esto NO es un error:** El modelo funciona perfectamente en CPU.

Si quieres usar GPU:
```bash
# Instalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 7. Docker build failed

**Solución A:** Limpiar caché y reconstruir
```bash
docker-compose down
docker system prune -a
docker-compose build --no-cache
docker-compose up
```

**Solución B:** Usar Docker normal (sin compose)
```bash
docker build -t tlob-app .
docker run -p 8501:8501 tlob-app
```

---

### 8. Ejemplos no se cargan

**Verificar que existan:**
```bash
ls -la data/BTC/individual_examples/
```

**Debería mostrar:**
```
example_1.npy
example_2.npy
example_3.npy
example_4.npy
example_5.npy
```

**Si no existen, generarlos:**
```bash
python create_individual_examples.py
```

---

### 9. Shape mismatch error

**Error:**
```
Shape incorrecto. Esperado: (128, 40), Recibido: (128, 44)
```

**Solución:**
El archivo tiene metadata extra. El modelo solo usa las primeras 40 features.

Verificar que `load_lob_window()` slice correctamente:
```python
window = np.load(file_path)[:, :40]  # Solo primeras 40 features
```

---

### 10. App se congela al predecir

**Causa posible:** Modelo muy grande para CPU.

**Solución:**
- Esperar unos segundos (primera predicción es lenta)
- Cerrar otras aplicaciones
- Usar GPU si está disponible

---

## Comandos Útiles de Diagnóstico

### Verificar instalación de Streamlit
```bash
streamlit --version
```

### Verificar instalación de PyTorch
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Listar puertos en uso
```bash
# Mac/Linux
netstat -an | grep LISTEN

# Mac específico
lsof -iTCP -sTCP:LISTEN -n -P
```

### Ver logs de Docker en tiempo real
```bash
docker-compose logs -f
```

### Ver recursos de Docker
```bash
docker stats
```

---

## Logs y Debugging

### Activar modo debug de Streamlit

Editar `~/.streamlit/config.toml`:
```toml
[logger]
level = "debug"
```

O ejecutar con flag:
```bash
streamlit run app.py --logger.level=debug
```

### Ver traceback completo

En `app.py`, añadir al inicio:
```python
import traceback
import sys

# Mostrar errores completos
st.set_option('client.showErrorDetails', True)
```

---

## Contacto y Soporte

Si el problema persiste:

1. **Verificar versiones:**
   ```bash
   python --version  # 3.9+
   streamlit --version  # 1.28+
   torch --version  # 2.0+
   ```

2. **Limpiar cache de Streamlit:**
   ```bash
   streamlit cache clear
   ```

3. **Reinstalar dependencias:**
   ```bash
   pip uninstall streamlit torch -y
   pip install -r requirements_streamlit.txt
   ```

4. **Crear issue en GitHub** con:
   - Sistema operativo
   - Versiones de Python y librerías
   - Traceback completo del error
   - Pasos para reproducir

---

**Última actualización:** Noviembre 2025

