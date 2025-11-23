# Guía de Deployment en Render

Esta guía te ayudará a deployar la API en Render sin usar Docker.

## Pre-requisitos

1. Cuenta en [Render](https://render.com) (gratis)
2. Repositorio en GitHub con tu proyecto
3. Los modelos (.cbm) y label_encoding.json en el repositorio

## Paso 1: Preparar el Repositorio

Asegúrate de que tu repositorio tenga los siguientes archivos críticos:

```
simulacion-tweets-metro/
├── models/
│   ├── modelo_clasificacion_falla.cbm  ✅ Debe estar en Git
│   └── modelo_deteccion_falla.cbm
├── data/
│   └── processed/
│       └── label_encoding.json  ✅ Debe estar en Git
├── src/
│   └── api/
│       └── main.py
├── requirements.txt
└── .env.example
```

### Verificar que los archivos están en Git:

```bash
git status
git add models/*.cbm data/processed/label_encoding.json
git commit -m "Add models and label encoding for deployment"
git push origin main
```

## Paso 2: Crear Web Service en Render

1. Ve a [Render Dashboard](https://dashboard.render.com/)
2. Haz clic en **"New +"** → **"Web Service"**
3. Conecta tu repositorio de GitHub
4. Selecciona el repositorio `simulacion-tweets-metro`

## Paso 3: Configurar el Web Service

### Build & Deploy Settings:

| Campo | Valor |
|-------|-------|
| **Name** | `metro-cdmx-api` (o el nombre que prefieras) |
| **Environment** | `Python 3` |
| **Region** | `Oregon (US West)` (o el más cercano) |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` |

**IMPORTANTE**: Render usa la variable `$PORT` automáticamente, NO pongas un puerto fijo.

## Paso 4: Configurar Variables de Entorno

En la sección **Environment Variables**, agrega:

| Key | Value |
|-----|-------|
| `PORT` | (Render lo asigna automáticamente, NO agregues esta) |
| `HOST` | `0.0.0.0` |
| `ALLOWED_ORIGINS` | `*` (o tu dominio específico) |
| `UMBRAL_ALERTA` | `80.0` |
| `MIN_TWEETS_PER_ITERATION` | `1` |
| `MAX_TWEETS_PER_ITERATION` | `3` |
| `EMBEDDING_MODEL` | `xlm-roberta-base` |
| `ENVIRONMENT` | `production` |
| `MODEL_CLASIFICACION_PATH` | `models/modelo_clasificacion_falla.cbm` |
| `MODEL_DETECCION_PATH` | `models/modelo_deteccion_falla.cbm` |
| `LABEL_ENCODING_PATH` | `data/processed/label_encoding.json` |

**Opcional - Para mejor seguridad CORS:**
- `ALLOWED_ORIGINS` → `https://tu-frontend.com,https://otro-dominio.com`

## Paso 5: Configurar Plan

- **Free Plan**: Suficiente para pruebas (se apaga después de 15 min de inactividad)
- **Starter Plan ($7/mes)**: Recomendado para producción (siempre activo)

## Paso 6: Deploy

1. Haz clic en **"Create Web Service"**
2. Render comenzará a construir tu aplicación
3. Espera de 5-10 minutos (descarga modelos de Hugging Face)

### Monitorear el Deploy:

En los logs verás:
```
🚀 Iniciando API...
📁 Directorio base: /opt/render/project/src
📦 Cargando modelos...
📂 Cargando modelo desde: /opt/render/project/src/models/modelo_clasificacion_falla.cbm
✅ Modelo CatBoost cargado
📂 Cargando modelo de embeddings: xlm-roberta-base
✅ Modelo de embeddings cargado
📂 Cargando mapeo de etiquetas desde: /opt/render/project/src/data/processed/label_encoding.json
✅ Mapeo de etiquetas cargado: {0: 'Normal', 1: 'Humo/Incendio', ...}
✅ Estado de estaciones inicializado
🎉 API lista para recibir peticiones en 0.0.0.0:10000!
```

## Paso 7: Verificar el Deployment

Tu API estará disponible en: `https://metro-cdmx-api.onrender.com`

### Prueba los endpoints:

```bash
# Health check
curl https://metro-cdmx-api.onrender.com/health

# Info de la API
curl https://metro-cdmx-api.onrender.com/

# Ejecutar iteración
curl https://metro-cdmx-api.onrender.com/iteracion

# Ver estado
curl https://metro-cdmx-api.onrender.com/estado
```

## Paso 8: Configurar Health Check (Opcional)

En **Settings** → **Health Check Path**:
- Path: `/health`
- Esto reiniciará automáticamente si la API falla

## Troubleshooting

### Error: "No se encontró el modelo"
- Verifica que los archivos .cbm están en Git:
  ```bash
  git ls-files models/
  ```
- Si no aparecen, agrégalos:
  ```bash
  git add models/*.cbm
  git commit -m "Add models"
  git push
  ```

### Error: "ModuleNotFoundError"
- Verifica que todas las dependencias están en `requirements.txt`
- Render usa exactamente lo que está en tu archivo

### La API se apaga después de 15 minutos (Free Plan)
- Es normal en el plan gratuito
- Se reactiva automáticamente con la primera petición (demora ~30 segundos)
- Para mantenerla siempre activa, usa el plan Starter

### Timeout durante Build
- Los modelos de Hugging Face tardan en descargarse la primera vez
- Es normal que tarde 5-10 minutos
- Si falla, haz un redeploy manual: **Manual Deploy** → **Deploy latest commit**

### Memoria insuficiente
- El plan Free tiene 512MB RAM
- Si necesitas más, usa el plan Starter (2GB RAM)

## Actualizar el Deployment

Cada vez que hagas push a `main`, Render automáticamente:
1. Detecta los cambios
2. Hace rebuild
3. Redeploy automático

Para forzar un redeploy sin cambios:
- **Manual Deploy** → **Clear build cache & deploy**

## URLs Importantes

- **API URL**: `https://tu-servicio.onrender.com`
- **Logs**: Dashboard → tu servicio → Logs
- **Metrics**: Dashboard → tu servicio → Metrics
- **Settings**: Dashboard → tu servicio → Settings

## Docs Interactivas

Una vez deployado, accede a:
- Swagger UI: `https://tu-servicio.onrender.com/docs`
- ReDoc: `https://tu-servicio.onrender.com/redoc`

## Costos Estimados

- **Free Plan**: $0/mes (con limitaciones)
- **Starter Plan**: $7/mes (recomendado)
- **Sin cargos ocultos por tráfico**

## Notas Finales

- Render usa Python 3.11 por defecto
- Los archivos estáticos (modelos) persisten entre deploys
- Los logs se mantienen por 7 días
- Backups automáticos no incluidos (haz backup manual de modelos)
