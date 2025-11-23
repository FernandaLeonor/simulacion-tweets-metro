# Sistema de Detección de Fallas del Metro CDMX mediante Análisis de Tweets

Sistema inteligente de monitoreo en tiempo real que detecta y clasifica fallas en el Metro de la Ciudad de México (Línea 1) mediante análisis de tweets usando Machine Learning y procesamiento de lenguaje natural.

## Descripción del Proyecto

Este proyecto utiliza modelos de Machine Learning (CatBoost) y embeddings de lenguaje natural (XLM-RoBERTa) para analizar tweets en tiempo real y detectar posibles fallas en las estaciones del Metro CDMX. El sistema puede:

- **Detectar fallas** (clasificación binaria: Normal vs Falla)
- **Clasificar tipo de falla** (multiclase): Normal, Humo/Incendio, Agua/Inundación, Eléctrica, Mecánica
- **Monitorear 20 estaciones** de la Línea 1 en tiempo real
- **Generar alertas críticas** cuando la certeza del modelo supera el 80%
- **Exponer API REST** con FastAPI para integración con frontends

## Estructura del Proyecto

```
simulacion-tweets-metro/
├── src/                              # Código fuente
│   ├── api/                          # API REST con FastAPI
│   │   └── main.py                   # Endpoints y lógica de la API
│   ├── data_generation/              # Generación de datos sintéticos
│   │   ├── fake_data_simple.py       # Generador básico de datos
│   │   ├── fake_data_coherent.py     # Generador de datos coherentes
│   │   └── realistic_tweet_generator.py  # Generador de tweets realistas
│   ├── features/                     # Procesamiento de características
│   │   └── feature_processor.py      # Genera embeddings XLM-RoBERTa
│   ├── training/                     # Entrenamiento de modelos
│   │   ├── train_binary_model.py     # Entrena modelo de detección
│   │   └── train_multiclass_models.py  # Entrena modelos binario + multiclase
│   └── simulation/                   # Simulación en tiempo real
│       ├── binary_simulator.py       # Simulador de detección binaria
│       └── multiclass_simulator.py   # Simulador de clasificación multiclase
│
├── models/                           # Modelos entrenados (.cbm)
├── data/                             # Datos del proyecto
│   └── processed/                    # Datos procesados
│       ├── features_embeddings.csv   # Dataset con embeddings
│       ├── features.json             # Features en formato JSON
│       └── label_encoding.json       # Mapeo de etiquetas
├── logs/                             # Logs de entrenamiento
├── notebooks/                        # Notebooks para exploración
├── tests/                            # Tests unitarios
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Este archivo
```

## Requisitos

- Python 3.8+
- pip o conda para gestión de paquetes

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/simulacion-tweets-metro.git
cd simulacion-tweets-metro
```

2. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Generar Datos de Entrenamiento

Genera tweets sintéticos realistas para entrenar los modelos:

```bash
python -m src.data_generation.realistic_tweet_generator
```

### 2. Procesar Features y Embeddings

Convierte los tweets en embeddings usando XLM-RoBERTa:

```bash
python -m src.features.feature_processor
```

### 3. Entrenar Modelos

Entrena los modelos de detección y clasificación:

```bash
python -m src.training.train_multiclass_models
```

Esto genera:
- `models/modelo_deteccion_falla.cbm` (clasificación binaria)
- `models/modelo_clasificacion_falla.cbm` (clasificación multiclase)
- `data/processed/label_encoding.json` (mapeo de etiquetas)

### 4. Ejecutar Simulación en Terminal

**Simulación con clasificación multiclase:**
```bash
python -m src.simulation.multiclass_simulator
```

**Simulación binaria (detección de fallas):**
```bash
python -m src.simulation.binary_simulator
```

### 5. Ejecutar API REST

Inicia el servidor FastAPI:

```bash
python -m src.api.main
```

La API estará disponible en `http://localhost:8000`

**Endpoints disponibles:**
- `GET /` - Información de la API
- `GET /iteracion` - Ejecuta una iteración de simulación
- `GET /estado` - Obtiene el estado actual de las estaciones
- `POST /reset` - Reinicia el estado de todas las estaciones

**Documentación interactiva:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Arquitectura del Sistema

### Flujo de Datos

```
Tweets → Embeddings (XLM-RoBERTa) → Features → Modelos CatBoost → Predicciones
```

### Modelos de ML

1. **Modelo de Detección (Binario)**
   - Objetivo: Detectar si hay falla o no
   - Clases: 0 (Normal), 1 (Falla)
   - Algoritmo: CatBoostClassifier

2. **Modelo de Clasificación (Multiclase)**
   - Objetivo: Clasificar el tipo de falla
   - Clases: Normal, Humo/Incendio, Agua/Inundación, Eléctrica, Mecánica
   - Algoritmo: CatBoostClassifier

### Features Utilizadas

- **Embeddings XLM-RoBERTa**: 384 dimensiones del texto del tweet
- **Estación**: Estación del Metro donde ocurre el evento (categórica)
- **Temperatura**: Temperatura ambiente simulada
- **Humedad**: Humedad relativa simulada
- **Precipitación**: Nivel de precipitación en mm
- **Nivel de tráfico**: Escala de 0-5

## Estaciones Monitoreadas (Línea 1)

```
Observatorio, Tacubaya, Juanacatlán, Chapultepec, Sevilla, Insurgentes,
Cuauhtémoc, Balderas, Salto del Agua, Isabel la Católica, Pino Suárez,
Merced, Candelaria, San Lázaro, Moctezuma, Balbuena, Boulevard Puerto Aéreo,
Gómez Farías, Zaragoza, Pantitlán
```

## Tipos de Fallas Detectadas

1. **Normal**: Operación sin problemas (puede incluir saturación)
2. **Humo/Incendio**: Detecta palabras clave como "humo", "quemado", "flama"
3. **Agua/Inundación**: Detecta "agua", "inundada", "filtración"
4. **Eléctrica**: Detecta "chispazo", "apagón", "luces parpadeando"
5. **Mecánica**: Detecta "freno", "puertas", "ruido mecánico"

## Configuración

### Umbrales y Parámetros

Puedes ajustar estos valores en los archivos de simulación:

```python
UMBRAL_ALERTA = 80.0  # Porcentaje para activar alertas críticas
INTERVALO = 5         # Segundos entre iteraciones
N_TWEETS = (1, 3)     # Rango de tweets a generar por iteración
```

## Tecnologías Utilizadas

- **FastAPI**: Framework web moderno y rápido
- **CatBoost**: Algoritmo de gradient boosting
- **Sentence Transformers**: Generación de embeddings
- **XLM-RoBERTa**: Modelo multilingüe de embeddings
- **Pandas/NumPy**: Procesamiento de datos
- **Scikit-learn**: Utilidades de ML

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## Autor

Aaron - Proyecto de Simulación de Tweets del Metro CDMX

## Agradecimientos

- Sistema de Transporte Colectivo Metro de la Ciudad de México
- Comunidad de usuarios del Metro que reportan incidentes en redes sociales
- Hugging Face por los modelos pre-entrenados de NLP
