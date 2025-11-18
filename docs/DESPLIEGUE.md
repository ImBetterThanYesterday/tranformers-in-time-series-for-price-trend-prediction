# Guía de Despliegue con Docker
## TLOB - Transformer for Limit Order Book

---

## 📋 **Índice**

1. [Requisitos Previos](#requisitos-previos)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Construcción de la Imagen Docker](#construcción-de-la-imagen-docker)
4. [Ejecución del Contenedor](#ejecución-del-contenedor)
5. [Docker Compose](#docker-compose)
6. [Verificación del Despliegue](#verificación-del-despliegue)
7. [Solución de Problemas](#solución-de-problemas)

---

## ✅ **Requisitos Previos**

### Software Necesario

```bash
# 1. Docker Desktop (incluye Docker Engine y Docker Compose)
# Descargar de: https://www.docker.com/products/docker-desktop

# 2. Git (para clonar el repositorio)
git --version

# 3. Al menos 4GB de RAM disponible
# 4. Al menos 10GB de espacio en disco
```

### Verificar Instalación

```bash
# Verificar Docker
docker --version
# Esperado: Docker version 20.10.x o superior

# Verificar Docker Compose
docker-compose --version
# Esperado: docker-compose version 1.29.x o superior

# Verificar que Docker está corriendo
docker ps
# Esperado: Lista vacía o contenedores existentes (sin errores)
```

---

## 📁 **Estructura del Proyecto**

```
TLOB-main/
├── Dockerfile                    # ⭐ Definición de la imagen Docker
├── docker-compose.yml            # ⭐ Orquestación del contenedor
├── requirements.txt              # Dependencias Python
├── app.py                        # Aplicación Streamlit
├── src/
│   ├── data/
│   │   ├── checkpoints/         # Pesos preentrenados del modelo
│   │   └── BTC/                 # Datos de ejemplo
│   ├── models/                  # Arquitectura del modelo
│   ├── preprocessing/           # Preprocesamiento de datos
│   └── utils/                   # Utilidades
├── inference/                   # Scripts de inferencia
├── config/                      # Configuración
└── docs/                        # Documentación
```

---

## 🏗️ **Construcción de la Imagen Docker**

### Dockerfile Explicado

```dockerfile
# Imagen base oficial de Python
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del proyecto
COPY . .

# Exponer el puerto de Streamlit (8501)
EXPOSE 8501

# Verificar instalación
RUN python -c "import torch; import streamlit; print('✓ Dependencies OK')"

# Comando de inicio: ejecutar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Construir la Imagen

```bash
# Navegar al directorio del proyecto
cd /ruta/a/TLOB-main

# Construir la imagen (puede tardar 5-10 minutos)
docker build -t tlob-app:latest .

# Verificar que la imagen se creó
docker images | grep tlob
# Esperado: tlob-app  latest  abc123  2 minutes ago  2.5GB
```

#### Opciones de Build

```bash
# Build con caché (más rápido en builds subsecuentes)
docker build -t tlob-app:latest .

# Build sin caché (desde cero)
docker build --no-cache -t tlob-app:latest .

# Build con nombre de versión
docker build -t tlob-app:v1.0 .

# Build silencioso (sin output detallado)
docker build -q -t tlob-app:latest .
```

---

## 🚀 **Ejecución del Contenedor**

### Método 1: Docker Run (Comando Simple)

```bash
# Ejecutar el contenedor
docker run -d \
  --name tlob-container \
  -p 8501:8501 \
  -v $(pwd)/src/data:/app/src/data \
  tlob-app:latest

# Verificar que está corriendo
docker ps | grep tlob

# Ver logs en tiempo real
docker logs -f tlob-container
```

#### Explicación de Parámetros

| Parámetro | Descripción |
|-----------|-------------|
| `-d` | Ejecutar en modo detached (segundo plano) |
| `--name tlob-container` | Nombre del contenedor |
| `-p 8501:8501` | Mapear puerto 8501 (host:contenedor) |
| `-v $(pwd)/src/data:/app/src/data` | Montar directorio de datos |
| `tlob-app:latest` | Imagen a ejecutar |

### Método 2: Docker Compose (Recomendado)

```bash
# Iniciar el servicio
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener el servicio
docker-compose down

# Reiniciar el servicio
docker-compose restart
```

---

## 🎼 **Docker Compose**

### docker-compose.yml Explicado

```yaml
version: '3.8'

services:
  tlob-app:
    # Nombre del contenedor
    container_name: tlob-streamlit
    
    # Construir desde Dockerfile local
    build:
      context: .
      dockerfile: Dockerfile
    
    # Mapeo de puertos
    ports:
      - "8501:8501"  # Streamlit
    
    # Volúmenes montados
    volumes:
      - ./src/data:/app/src/data:ro  # Datos (read-only)
      - ./logs:/app/logs             # Logs (escritura)
    
    # Variables de entorno
    environment:
      - PYTHONUNBUFFERED=1
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    
    # Política de reinicio
    restart: unless-stopped
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Comandos Útiles de Docker Compose

```bash
# Iniciar servicios
docker-compose up -d

# Ver estado
docker-compose ps

# Ver logs
docker-compose logs -f tlob-app

# Ejecutar comando en el contenedor
docker-compose exec tlob-app bash

# Reconstruir y reiniciar
docker-compose up -d --build

# Detener y eliminar contenedores
docker-compose down

# Detener, eliminar y limpiar volúmenes
docker-compose down -v
```

---

## ✅ **Verificación del Despliegue**

### 1. Verificar que el Contenedor está Corriendo

```bash
docker ps --filter name=tlob
```

**Output esperado:**
```
CONTAINER ID   IMAGE              COMMAND                  STATUS         PORTS
abc123def456   tlob-app:latest    "streamlit run app.py"   Up 2 minutes   0.0.0.0:8501->8501/tcp
```

### 2. Verificar Logs

```bash
docker logs tlob-container --tail 50
```

**Output esperado:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.17.0.2:8501
```

### 3. Acceder a la Aplicación

Abrir navegador en:
```
http://localhost:8501
```

**Debe mostrarse:**
- ✅ Interfaz de Streamlit
- ✅ Título: "TLOB - Predicción de Tendencias en Limit Order Book"
- ✅ Selector de modelos
- ✅ Dropdown de ejemplos

### 4. Probar Predicción

1. Seleccionar modelo: "TLOB"
2. Cargar ejemplo: "CSV Example 1"
3. Clic en "🔮 Realizar Predicción"

**Resultado esperado:**
```
✅ Predicción completada

Predicción: UP
Confianza: 78.5%

Probabilidades:
- DOWN: 10.2%
- STATIONARY: 11.3%
- UP: 78.5%
```

---

## 🔧 **Comandos de Mantenimiento**

### Entrar al Contenedor

```bash
# Con docker run
docker exec -it tlob-container bash

# Con docker-compose
docker-compose exec tlob-app bash

# Dentro del contenedor
ls -la
python --version
streamlit --version
```

### Ver Uso de Recursos

```bash
# Uso de CPU y memoria
docker stats tlob-container

# Inspeccionar contenedor
docker inspect tlob-container

# Ver procesos dentro del contenedor
docker top tlob-container
```

### Copiar Archivos

```bash
# Desde host a contenedor
docker cp archivo.txt tlob-container:/app/

# Desde contenedor a host
docker cp tlob-container:/app/logs/app.log ./
```

---

## 🐛 **Solución de Problemas**

### Problema 1: Puerto 8501 Ocupado

**Error:**
```
Error starting userland proxy: listen tcp4 0.0.0.0:8501: bind: address already in use
```

**Solución:**
```bash
# Encontrar proceso usando el puerto
lsof -i :8501

# Matar el proceso
kill -9 <PID>

# O usar otro puerto
docker run -p 8502:8501 tlob-app:latest
```

### Problema 2: Contenedor se Detiene Inmediatamente

**Diagnóstico:**
```bash
# Ver logs completos
docker logs tlob-container

# Ver últimas 100 líneas
docker logs --tail 100 tlob-container
```

**Causas comunes:**
- Dependencias faltantes → Revisar `requirements.txt`
- Error en `app.py` → Ver traceback en logs
- Pesos del modelo no encontrados → Verificar `src/data/checkpoints/`

### Problema 3: Dependencias no se Instalan

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torch==2.0.1
```

**Solución:**
```bash
# Actualizar pip en el Dockerfile
RUN pip install --upgrade pip setuptools wheel

# O usar versiones compatibles
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Problema 4: Modelo no se Carga

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'src/data/checkpoints/TLOB/...'
```

**Solución:**
```bash
# Verificar que los checkpoints están en el volumen
docker exec tlob-container ls -la src/data/checkpoints/TLOB/

# Si faltan, copiarlos:
docker cp src/data/checkpoints tlob-container:/app/src/data/
```

### Problema 5: Aplicación Lenta

**Diagnóstico:**
```bash
# Ver uso de recursos
docker stats tlob-container
```

**Optimizaciones:**
```yaml
# En docker-compose.yml, aumentar recursos:
services:
  tlob-app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          memory: 2G
```

---

## 🔄 **Actualización y Redeploy**

### Actualizar el Código

```bash
# 1. Detener contenedor
docker-compose down

# 2. Hacer cambios en el código
git pull origin main
# o editar archivos localmente

# 3. Reconstruir imagen
docker-compose build

# 4. Reiniciar con nueva versión
docker-compose up -d

# 5. Verificar logs
docker-compose logs -f
```

### Limpiar Imágenes Antiguas

```bash
# Ver imágenes
docker images

# Eliminar imagen específica
docker rmi tlob-app:v1.0

# Limpiar imágenes no usadas
docker image prune -a

# Limpiar todo (contenedores, imágenes, volúmenes)
docker system prune -a --volumes
```

---

## 📊 **Monitoreo**

### Logs en Producción

```bash
# Guardar logs en archivo
docker logs tlob-container > logs/app_$(date +%Y%m%d).log 2>&1

# Ver logs en tiempo real con timestamp
docker logs -f --timestamps tlob-container

# Filtrar logs por palabra clave
docker logs tlob-container 2>&1 | grep ERROR
```

### Health Checks

```bash
# Verificar salud del contenedor
docker inspect --format='{{.State.Health.Status}}' tlob-container

# Ver historial de health checks
docker inspect --format='{{json .State.Health}}' tlob-container | jq
```

---

## 🌐 **Despliegue en Producción**

### Consideraciones

1. **Variables de Entorno Secretas**
```yaml
environment:
  - SECRET_KEY=${SECRET_KEY}
env_file:
  - .env
```

2. **HTTPS/SSL**
```yaml
# Usar reverse proxy (Nginx, Traefik)
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.tlob.rule=Host(`tlob.example.com`)"
```

3. **Persistencia de Datos**
```yaml
volumes:
  - tlob_data:/app/src/data
volumes:
  tlob_data:
```

4. **Escalabilidad**
```yaml
deploy:
  replicas: 3
  update_config:
    parallelism: 1
    delay: 10s
```

---

## 📚 **Referencias**

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Deployment](https://docs.streamlit.io/knowledge-base/tutorials/deploy)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

---

**Última actualización**: Noviembre 2025  
**Autor**: Proyecto Final - Análisis de Series Temporales con Transformers


