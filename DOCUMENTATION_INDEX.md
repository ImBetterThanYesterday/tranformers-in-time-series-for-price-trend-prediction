# 📚 Índice de Documentación - Proyecto TLOB

## 🎯 Documentación Principal

### 1. **Resumen de la Sesión**
📄 [`SESSION_SUMMARY.md`](./SESSION_SUMMARY.md)
- ✨ Resumen completo de todos los logros
- 📊 Aplicación Streamlit + Procesamiento Raw Data
- 🎯 Flujo end-to-end del CSV a la predicción
- 📈 Visualizaciones y resultados

### 2. **Procesamiento de Datos Crudos**
📄 [`SUMMARY_RAW_DATA_PROCESSING.md`](./SUMMARY_RAW_DATA_PROCESSING.md)
- 🔄 Pipeline completo de procesamiento
- 📊 Resultados y estadísticas
- ✅ Verificaciones de calidad
- 💡 Lecciones aprendidas

### 3. **Demo y Tutorial**
📄 [`DEMO_RAW_DATA.md`](./DEMO_RAW_DATA.md)
- 🚀 Tutorial paso a paso
- 📝 Ejemplos de código
- 🔧 Troubleshooting
- 📚 Referencias

### 4. **Documentación Original**
📄 [`docs/knowledge.md`](./docs/knowledge.md)
- 📖 Conocimiento base del proyecto
- 🏗️ Arquitectura del repositorio
- 🔄 Flujo de datos
- 🤖 Detalles de los modelos

---

## 🛠️ Scripts y Herramientas

### Scripts de Procesamiento

| Script | Descripción | Comando |
|--------|-------------|---------|
| **`process_raw_btc_samples.py`** | Procesa CSV crudo | `python3 process_raw_btc_samples.py --num_samples 10` |
| **`compare_raw_vs_processed.py`** | Valida y compara datos | `python3 compare_raw_vs_processed.py` |
| **`create_individual_examples.py`** | Genera ejemplos individuales | `python3 create_individual_examples.py` |

### Scripts de Inferencia

| Script | Descripción | Comando |
|--------|-------------|---------|
| **`inference_pytorch.py`** | Inferencia batch PyTorch | `python3 inference_pytorch.py --examples_path <file>` |
| **`inference_onnx.py`** | Inferencia optimizada ONNX | `python3 inference_onnx.py --examples_path <file>` |
| **`inference_single_file.py`** | Inferencia archivo individual | `python3 inference_single_file.py <file>` |
| **`run_all_inferences.py`** | Inferencia batch automatizada | `python3 run_all_inferences.py` |

### Scripts de Análisis

| Script | Descripción | Comando |
|--------|-------------|---------|
| **`demo_inference.py`** | Demo completo de inferencia | `python3 demo_inference.py` |
| **`extract_examples.py`** | Extrae ejemplos del dataset | `python3 extract_examples.py --num 5` |

### Aplicación Web

| Archivo | Descripción | Comando |
|---------|-------------|---------|
| **`app.py`** | Aplicación Streamlit | `streamlit run app.py` |
| **`docker-compose.yml`** | Orquestación Docker | `docker-compose up` |

---

## 📁 Estructura de Datos

### Datos Originales
```
data/BTC/original_source/
└── 1-09-1-20.csv          # CSV de Kaggle (3.7M filas, 1.1GB)
```

### Datos Preprocesados
```
data/BTC/
├── train.npy              # Training set (2,780,963 × 44)
├── val.npy                # Validation set
└── test.npy               # Test set
```

### Ejemplos para Streamlit
```
data/BTC/individual_examples/
├── example_1.npy          # (128 × 40)
├── example_2.npy
├── example_3.npy
├── example_4.npy
├── example_5.npy
├── summary_all_inferences.txt
└── README.md
```

### Muestras Raw Procesadas
```
data/BTC/raw_samples/
├── raw_sample_1.npy       # (128 × 40)
├── ...
├── raw_sample_10.npy
├── raw_samples_batch.npy  # (10 × 128 × 40)
├── raw_sample_1_result.npy
├── raw_sample_1_result.txt
├── metadata.json
└── README.md
```

---

## 🎯 Flujos de Trabajo

### 1. **Entrenar Modelo**
```bash
# 1. Preprocesar datos (si no está hecho)
python3 main.py --config config/config.py

# 2. Entrenar
python3 main.py \
    --model TLOB \
    --dataset BTC \
    --horizon 10 \
    --seq_size 128
```

### 2. **Procesar Datos Raw**
```bash
# 1. Procesar CSV original
python3 process_raw_btc_samples.py --num_samples 10

# 2. Validar datos
python3 compare_raw_vs_processed.py

# 3. Inferencia
python3 inference_single_file.py data/BTC/raw_samples/raw_sample_1.npy
```

### 3. **Desplegar Aplicación**
```bash
# Opción 1: Docker
docker-compose up

# Opción 2: Local
streamlit run app.py
```

### 4. **Análisis Completo**
```bash
# 1. Generar ejemplos
python3 extract_examples.py --num 5

# 2. Inferencia batch
python3 run_all_inferences.py

# 3. Visualizar en Streamlit
streamlit run app.py
```

---

## 📖 Guías Rápidas

### Quick Start: Inferencia en Nuevos Datos

1. **Descargar CSV** de Kaggle o Binance
2. **Procesar**:
   ```bash
   python3 process_raw_btc_samples.py \
       --csv_path tu_archivo.csv \
       --num_samples 10
   ```
3. **Inferencia**:
   ```bash
   python3 inference_single_file.py \
       data/BTC/raw_samples/raw_sample_1.npy
   ```

### Quick Start: Aplicación Streamlit

1. **Con Docker**:
   ```bash
   docker-compose up
   ```
   → Abrir http://localhost:8501

2. **Local**:
   ```bash
   pip install -r requirements_streamlit.txt
   streamlit run app.py
   ```

### Quick Start: Validar Datos

```bash
python3 compare_raw_vs_processed.py
```

---

## 🔍 Búsqueda Rápida

### Por Tarea:

**Quiero procesar datos nuevos**
→ [`DEMO_RAW_DATA.md`](./DEMO_RAW_DATA.md) - Sección "Ejemplo Completo"

**Quiero entender el modelo**
→ [`docs/knowledge.md`](./docs/knowledge.md) - Sección "5. Modelo TLOB"

**Quiero ver resultados**
→ [`SESSION_SUMMARY.md`](./SESSION_SUMMARY.md) - Sección "Resultados Obtenidos"

**Quiero hacer inferencia**
→ [`DEMO_RAW_DATA.md`](./DEMO_RAW_DATA.md) - Sección "Inferencia Individual"

**Tengo un error**
→ [`DEMO_RAW_DATA.md`](./DEMO_RAW_DATA.md) - Sección "Troubleshooting"

### Por Concepto:

**Normalización Z-score**
→ [`SUMMARY_RAW_DATA_PROCESSING.md`](./SUMMARY_RAW_DATA_PROCESSING.md) - Sección "Detalles Técnicos"

**Estructura del LOB**
→ [`SESSION_SUMMARY.md`](./SESSION_SUMMARY.md) - Sección "Entendimiento del Dataset BTC"

**Pipeline completo**
→ [`SESSION_SUMMARY.md`](./SESSION_SUMMARY.md) - Sección "Flujo Completo"

**Compatibilidad de datos**
→ [`SUMMARY_RAW_DATA_PROCESSING.md`](./SUMMARY_RAW_DATA_PROCESSING.md) - Sección "Verificaciones de Calidad"

---

## 📊 Tablas de Referencia

### Modelos Disponibles

| Modelo | Seq Size | Parámetros | Checkpoint |
|--------|----------|------------|------------|
| TLOB | 128 | 1.1M | `data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_42/` |
| DEEPLOB | 100 | - | `data/checkpoints/DEEPLOB/` |
| BINCTABL | 10 | - | `data/checkpoints/BINCTABL/` |
| MLPLOB | 384 | - | `data/checkpoints/MLPLOB/` |

### Horizontes de Predicción

| Horizon | Timesteps Adelante | Duración (250ms sampling) |
|---------|-------------------|---------------------------|
| h=10 | 10 | 2.5 segundos |
| h=20 | 20 | 5.0 segundos |
| h=50 | 50 | 12.5 segundos |
| h=100 | 100 | 25.0 segundos |

### Features del LOB

| Rango | Descripción | Tipo |
|-------|-------------|------|
| F0-F9 | ASK Price L1-L10 | Precio (sell orders) |
| F10-F19 | ASK Volume L1-L10 | Volumen (sell orders) |
| F20-F29 | BID Price L1-L10 | Precio (buy orders) |
| F30-F39 | BID Volume L1-L10 | Volumen (buy orders) |

---

## 🎓 Recursos Adicionales

### Artículos y Papers
- **TLOB**: Temporal Limit Order Book for Price Trend Prediction
- **FI-2010**: Benchmarking deep order flow imbalance
- **DeepLOB**: Deep convolutional neural networks for limit order books

### Datasets
- **BTC**: [Kaggle - Bitcoin Perpetual LOB](https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data)
- **FI-2010**: Benchmark dataset para LOB prediction

### Repositorios Relacionados
- **Repositorio Original**: [Link al repo original del paper TLOB]
- **Fork del Proyecto**: [Tu repositorio]

---

## 🆘 Soporte y Troubleshooting

### Problemas Comunes

**1. RecursionError en Streamlit**
- **Solución**: Actualizar a Python 3.12 y Streamlit 1.39.0
- **Archivo**: `Dockerfile`, `requirements_streamlit.txt`

**2. Shape mismatch en inferencia**
- **Solución**: Verificar que los datos sean (128, 40)
- **Script**: `compare_raw_vs_processed.py`

**3. CSV con formato diferente**
- **Solución**: Verificar 43 columnas (1 index + 42 datos)
- **Documentación**: `DEMO_RAW_DATA.md` - Estructura del CSV

**4. NaN o Inf en datos**
- **Solución**: Revisar normalización
- **Script**: `compare_raw_vs_processed.py`

### Contacto
- **Issues**: [GitHub Issues de tu repo]
- **Documentación**: Este archivo y los enlaces arriba

---

## 📅 Historial de Cambios

### 2024-11-16
- ✅ Creado pipeline de procesamiento raw data
- ✅ Actualizada aplicación Streamlit (40 features, 128 timesteps)
- ✅ Solucionados RecursionError y ValueError
- ✅ Actualizado a Python 3.12, Streamlit 1.39.0
- ✅ Documentación completa generada

### [Versión Anterior]
- ✅ Implementación inicial del modelo TLOB
- ✅ Preprocesamiento de datos BTC
- ✅ Scripts de inferencia PyTorch y ONNX

---

## ✅ Checklist de Uso

### Para Nuevos Usuarios:
- [ ] Leer [`SESSION_SUMMARY.md`](./SESSION_SUMMARY.md)
- [ ] Seguir [`DEMO_RAW_DATA.md`](./DEMO_RAW_DATA.md)
- [ ] Ejecutar `docker-compose up`
- [ ] Probar inferencia con ejemplos precargados

### Para Desarrollo:
- [ ] Leer [`docs/knowledge.md`](./docs/knowledge.md)
- [ ] Configurar entorno Python 3.12
- [ ] Instalar dependencias (`requirements.txt`)
- [ ] Ejecutar tests de inferencia

### Para Producción:
- [ ] Validar datos con `compare_raw_vs_processed.py`
- [ ] Configurar Docker
- [ ] Probar inferencia en batch
- [ ] Monitorear métricas

---

**Última actualización**: 2024-11-16
**Versión**: 1.0
**Mantenido por**: [Tu nombre/equipo]

