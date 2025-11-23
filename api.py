from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import random
from datetime import datetime
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
import json
import pandas as pd
from generador_tweets_realistas import generar_tweet_simulado

# ================= MODELOS PYDANTIC =================
class EstacionEstado(BaseModel):
    estacion: str
    hora: str
    probabilidades: Dict[int, float]  # {clase_id: probabilidad}
    no_falla_prob: float
    falla_mas_probable: str
    falla_mas_probable_prob: float
    alerta: bool  # Si supera el umbral

class TweetProcesado(BaseModel):
    texto: str
    estacion: str
    clase_predicha: str
    probabilidad_clase: float
    timestamp: str

class AlertaCritica(BaseModel):
    estacion: str
    tipo_falla: str
    certeza: float
    tweet: str
    timestamp: str

class IteracionResponse(BaseModel):
    timestamp: str
    tweets_procesados: List[TweetProcesado]
    estados_estaciones: List[EstacionEstado]
    alertas_criticas: List[AlertaCritica]
    numero_tweets: int

# ================= CONFIG =================
UMBRAL_ALERTA = 80.0  # % para activar alarma
N_TWEETS = (1, 3)

estaciones_L1 = [
    "Observatorio", "Tacubaya", "Juanacatlán", "Chapultepec", "Sevilla",
    "Insurgentes", "Cuauhtémoc", "Balderas", "Salto del Agua", "Isabel la Católica",
    "Pino Suárez", "Merced", "Candelaria", "San Lázaro", "Moctezuma", "Balbuena",
    "Boulevard Puerto Aéreo", "Gómez Farías", "Zaragoza", "Pantitlán"
]

# ================= VARIABLES GLOBALES =================
app = FastAPI(
    title="API Simulación Metro CDMX",
    description="API para simular y clasificar tweets del Metro CDMX Línea 1",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para modelos y estado
model_cb = None
embed_model = None
label_mapping = {}
estatus_estaciones = {}

# ================= FUNCIONES AUXILIARES =================
def get_initial_probs():
    """Inicializa probabilidades para una estación"""
    initial_probs = {i: 0.0 for i in label_mapping.keys()}
    if 0 in initial_probs:
        initial_probs[0] = 100.0
    else:
        num_classes = len(label_mapping)
        for k in initial_probs:
            initial_probs[k] = 100.0 / num_classes
    initial_probs['hora'] = '-'
    return initial_probs

def inicializar_estaciones():
    """Inicializa el estado de todas las estaciones"""
    global estatus_estaciones
    estatus_estaciones = {est: get_initial_probs() for est in estaciones_L1}

# ================= EVENTOS DE INICIO =================
@app.on_event("startup")
async def load_models():
    """Carga los modelos al iniciar la aplicación"""
    global model_cb, embed_model, label_mapping

    print("🚀 Iniciando API...")
    print("📦 Cargando modelos...")

    # Cargar modelo CatBoost
    model_cb = CatBoostClassifier()
    model_cb.load_model("modelo_clasificacion_falla.cbm")
    print("✅ Modelo CatBoost cargado")

    # Cargar modelo de embeddings
    embed_model = SentenceTransformer('xlm-roberta-base')
    print("✅ Modelo de embeddings cargado")

    # Cargar mapeo de etiquetas
    try:
        with open('label_encoding.json', 'r', encoding='utf-8') as f:
            label_mapping_raw = json.load(f)
        label_mapping = {int(k): v for k, v in label_mapping_raw.items()}
        print(f"✅ Mapeo de etiquetas cargado: {label_mapping}")
    except FileNotFoundError:
        print("❌ Error: No se encontró 'label_encoding.json'")
        raise

    # Inicializar estado de estaciones
    inicializar_estaciones()
    print("✅ Estado de estaciones inicializado")
    print("🎉 API lista para recibir peticiones!")

# ================= ENDPOINTS =================
@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API Simulación Metro CDMX - Línea 1",
        "version": "1.0.0",
        "endpoints": {
            "/iteracion": "Ejecuta una iteración de la simulación",
            "/estado": "Obtiene el estado actual de todas las estaciones",
            "/reset": "Reinicia el estado de todas las estaciones"
        }
    }

@app.get("/iteracion", response_model=IteracionResponse)
async def ejecutar_iteracion():
    """
    Ejecuta una iteración de la simulación:
    - Genera tweets aleatorios
    - Los clasifica con el modelo
    - Actualiza el estado de las estaciones
    - Retorna los resultados
    """
    timestamp_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Generar tweets
    num_tweets_a_generar = random.randint(N_TWEETS[0], N_TWEETS[1])
    tweets_generados = generar_tweet_simulado(num_tweets=num_tweets_a_generar)

    tweets_procesados = []
    alertas_criticas = []

    for tweet_data in tweets_generados:
        tweet_text = tweet_data['text']

        # Extraer estación del tweet
        try:
            estacion_match = tweet_text.split('**')[1]
            estacion = estacion_match.split(',')[0].strip()
            if estacion not in estaciones_L1:
                estacion = random.choice(estaciones_L1)
        except:
            estacion = random.choice(estaciones_L1)

        # Generar datos aleatorios para features
        temp = random.uniform(15.0, 35.0)
        humidity = random.uniform(40.0, 95.0)
        precip_mm = random.choices([0.0, random.uniform(0.1, 10.0)], weights=[0.8, 0.2], k=1)[0]
        traffic_jam_level = random.randint(0, 5)

        # Vector embedding
        vector = embed_model.encode(tweet_text).tolist()

        # Preparar features para el modelo
        features_dict = {
            'station': estacion,
            'temp': temp,
            'humidity': humidity,
            'precip_mm': precip_mm,
            'traffic_jam_level': traffic_jam_level,
        }
        for i, val in enumerate(vector):
            features_dict[f"embedding_{i}"] = val

        # Crear DataFrame con el orden correcto de features
        model_feature_names = model_cb.feature_names_
        X_input = pd.DataFrame([features_dict], columns=model_feature_names)

        # Predicción
        probabilidades_raw = model_cb.predict_proba(X_input)[0]
        probabilidades_dict = {i: prob * 100 for i, prob in enumerate(probabilidades_raw)}

        pred_clase_idx = model_cb.predict(X_input)[0]
        pred_clase_label = label_mapping[int(pred_clase_idx)]
        prob_falla_display = probabilidades_dict[int(pred_clase_idx)]

        # Actualizar estado de la estación
        estatus_estaciones[estacion].update(probabilidades_dict)
        estatus_estaciones[estacion]['hora'] = datetime.now().strftime('%H:%M')

        # Agregar a tweets procesados
        tweets_procesados.append(TweetProcesado(
            texto=tweet_text,
            estacion=estacion,
            clase_predicha=pred_clase_label,
            probabilidad_clase=prob_falla_display,
            timestamp=timestamp_actual
        ))

        # Verificar si hay alerta crítica
        if int(pred_clase_idx) != 0 and prob_falla_display > UMBRAL_ALERTA:
            alertas_criticas.append(AlertaCritica(
                estacion=estacion,
                tipo_falla=pred_clase_label,
                certeza=prob_falla_display,
                tweet=tweet_text,
                timestamp=timestamp_actual
            ))

    # Construir estados de todas las estaciones
    estados = []
    for estacion in estaciones_L1:
        datos = estatus_estaciones[estacion]

        # Probabilidad de No Falla (clase 0)
        prob_no_falla = datos.get(0, 0.0)

        # Encontrar la falla más probable (excluyendo clase 0)
        max_falla_prob = 0.0
        max_falla_nombre = "N/A"

        for class_idx in label_mapping.keys():
            if class_idx != 0:
                prob = datos.get(class_idx, 0.0)
                if prob > max_falla_prob:
                    max_falla_prob = prob
                    max_falla_nombre = label_mapping[class_idx]

        # Determinar si hay alerta
        tiene_alerta = max_falla_prob > UMBRAL_ALERTA

        # Extraer solo las probabilidades (sin 'hora')
        probabilidades_limpias = {k: v for k, v in datos.items() if k != 'hora'}

        estados.append(EstacionEstado(
            estacion=estacion,
            hora=datos['hora'],
            probabilidades=probabilidades_limpias,
            no_falla_prob=prob_no_falla,
            falla_mas_probable=max_falla_nombre,
            falla_mas_probable_prob=max_falla_prob,
            alerta=tiene_alerta
        ))

    return IteracionResponse(
        timestamp=timestamp_actual,
        tweets_procesados=tweets_procesados,
        estados_estaciones=estados,
        alertas_criticas=alertas_criticas,
        numero_tweets=num_tweets_a_generar
    )

@app.get("/estado")
async def obtener_estado():
    """Obtiene el estado actual de todas las estaciones sin ejecutar una nueva iteración"""
    estados = []
    for estacion in estaciones_L1:
        datos = estatus_estaciones[estacion]

        prob_no_falla = datos.get(0, 0.0)

        max_falla_prob = 0.0
        max_falla_nombre = "N/A"

        for class_idx in label_mapping.keys():
            if class_idx != 0:
                prob = datos.get(class_idx, 0.0)
                if prob > max_falla_prob:
                    max_falla_prob = prob
                    max_falla_nombre = label_mapping[class_idx]

        tiene_alerta = max_falla_prob > UMBRAL_ALERTA
        probabilidades_limpias = {k: v for k, v in datos.items() if k != 'hora'}

        estados.append(EstacionEstado(
            estacion=estacion,
            hora=datos['hora'],
            probabilidades=probabilidades_limpias,
            no_falla_prob=prob_no_falla,
            falla_mas_probable=max_falla_nombre,
            falla_mas_probable_prob=max_falla_prob,
            alerta=tiene_alerta
        ))

    return {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "estados_estaciones": estados
    }

@app.post("/reset")
async def reiniciar_estado():
    """Reinicia el estado de todas las estaciones a sus valores iniciales"""
    inicializar_estaciones()
    return {
        "message": "Estado de estaciones reiniciado correctamente",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ================= EJECUCIÓN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
