import random
import time
import pandas as pd
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
import json 
from generador_tweets_realistas import generar_tweet_simulado 

# ================= CONFIG =================
INTERVALO = 5  # Más rápido para ver las alertas (5 segundos)
UMBRAL_ALERTA = 80.0  # % para activar alarma
N_TWEETS = (1, 3) 

# ================= DATOS =================
estaciones_L1 = [
    "Observatorio", "Tacubaya", "Juanacatlán", "Chapultepec", "Sevilla",
    "Insurgentes", "Cuauhtémoc", "Balderas", "Salto del Agua", "Isabel la Católica",
    "Pino Suárez", "Merced", "Candelaria", "San Lázaro", "Moctezuma", "Balbuena",
    "Boulevard Puerto Aéreo", "Gómez Farías", "Zaragoza", "Pantitlán"
]
dias_semana = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']

# ================= CARGAR MODELOS =================
print("Cargando cerebro...")
model_cb = CatBoostClassifier()
model_cb.load_model("modelo_clasificacion_falla.cbm") # Usar el modelo de clasificación de falla
embed_model = SentenceTransformer('xlm-roberta-base')

# Cargar el mapeo de etiquetas
try:
    with open('label_encoding.json', 'r', encoding='utf-8') as f:
        label_mapping_raw = json.load(f)
    # Convertir claves de string a int
    label_mapping = {int(k): v for k, v in label_mapping_raw.items()}
    print(f"Mapeo de etiquetas cargado: {label_mapping}")
except FileNotFoundError:
    print("Error: No se encontró 'label_encoding.json'. Asegúrate de que el modelo de clasificación haya sido entrenado y guardado.")
    exit()

# Definiciones de fallas para mostrar en el tablero
TIPOS_FALLA = label_mapping
NOMBRES_CORTOS_FALLA = {k: v[:4].upper() for k, v in label_mapping.items()} # Tomar las primeras 4 letras en mayúsculas

# ================= ESTADO DEL SISTEMA MODIFICADO =================
# Diccionario para guardar el último estado de CADA PROBABILIDAD para cada estación
def get_initial_probs():
    # Iniciamos con 100% en la primera clase (asumimos "Normal" o similar) y 0% en el resto
    initial_probs = {i: 0.0 for i in label_mapping.keys()}
    # Assuming class 0 is 'Normal' or 'No Falla' if it exists. Otherwise, distribute evenly.
    if 0 in initial_probs:
        initial_probs[0] = 100.0
    else: # If no class 0, distribute evenly or handle as per desired initial state for multiclass
        num_classes = len(label_mapping)
        for k in initial_probs:
            initial_probs[k] = 100.0 / num_classes # Even distribution
    initial_probs['hora'] = '-'
    return initial_probs

estatus_estaciones = {est: get_initial_probs() for est in estaciones_L1}

print("✅ Sistemas listos. Iniciando monitoreo...")

# ================= FUNCIONES =================
def limpiar_consola():
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_tablero():
    print("\n" + "="*85)
    print(f"   MONITOREO LÍNEA 1 - {datetime.now().strftime('%H:%M:%S')}")
    print("="*85)
    
    # 1. Cabecera con un resumen de fallas
    header = f"{'ESTACIÓN':<20} | {'HORA':<4} | {'NO FALLA':<8} | {'FALLA MÁS PROBABLE':<20}"
    print(header)
    print("-" * 85)
    
    # 2. Iterar sobre TODAS las estaciones
    for estacion in estaciones_L1:
        datos = estatus_estaciones[estacion]
        
        # Probabilidad de No Falla (clase 0)
        prob_no_falla = datos.get(0, 0.0)
        
        # Encontrar la falla más probable entre las clases > 0
        max_falla_prob = 0.0
        max_falla_nombre = "N/A"
        
        for class_idx in sorted(TIPOS_FALLA.keys()):
            if class_idx != 0: # Excluir la clase 'No Falla'
                prob = datos.get(class_idx, 0.0)
                if prob > max_falla_prob:
                    max_falla_prob = prob
                    max_falla_nombre = TIPOS_FALLA[class_idx]

        # Formatear las probabilidades
        prob_no_falla_str = f"{prob_no_falla:.1f}%"
        falla_mas_probable_str = f"{max_falla_nombre[:12]:<12} ({max_falla_prob:.1f}%)" # Truncate falla name if too long

        # Colores para 'No Falla'
        color_no_falla = "\033[92m" # Verde
        reset = "\033[0m"

        # Colores para 'Falla Más Probable'
        color_falla = "\033[92m" # Verde (default, si es baja)
        if max_falla_prob > UMBRAL_ALERTA:
            color_falla = "\033[91m" # Rojo (Alerta)
        elif max_falla_prob > 50:
            color_falla = "\033[93m" # Amarillo (Advertencia)

        # 5. Imprimir la fila
        print(f"{estacion:<20} | {datos['hora']:<4} | {color_no_falla}{prob_no_falla_str:<8}{reset} | {color_falla}{falla_mas_probable_str:<20}{reset}")
    
    print("="*85)

# ================= LOOP PRINCIPAL =================
try:
    while True:
        # CAMBIO PRINCIPAL: Usar tu generador mejorado en lugar de generar_tweet()
        num_tweets_a_generar = random.randint(N_TWEETS[0], N_TWEETS[1])
        tweets_generados = generar_tweet_simulado(num_tweets=num_tweets_a_generar)
        
        nuevos_reportes = []

        for tweet_data in tweets_generados:
            tweet_text = tweet_data['text']
            
            # Extraer estación del tweet (formato: @MetroCDMX en **Estacion**, ...)
            try:
                estacion_match = tweet_text.split('**')[1] 
                estacion = estacion_match.split(',')[0].strip()
                if estacion not in estaciones_L1:
                    estacion = random.choice(estaciones_L1)
            except:
                estacion = random.choice(estaciones_L1)
            
            # Generar datos aleatorios para las nuevas features
            temp = random.uniform(15.0, 35.0)
            humidity = random.uniform(40.0, 95.0)
            precip_mm = random.choices([0.0, random.uniform(0.1, 10.0)], weights=[0.8, 0.2], k=1)[0] # Mostly no rain
            traffic_jam_level = random.randint(0, 5) # Scale of 0-5
            
            # 1. Vector embedding
            vector = embed_model.encode(tweet_text).tolist()

            # 2. Preparar todas las features para el modelo
            features_dict = {
                'station': estacion,
                'temp': temp,
                'humidity': humidity,
                'precip_mm': precip_mm,
                'traffic_jam_level': traffic_jam_level,
            }
            # Add embedding features
            for i, val in enumerate(vector):
                features_dict[f"embedding_{i}"] = val
            
            # Crear DataFrame con todas las features, asegurando el orden correcto
            model_feature_names = model_cb.feature_names_
            X_input = pd.DataFrame([features_dict], columns=model_feature_names)

            # 3. Predicción de probabilidades (predict_proba)
            probabilidades_raw = model_cb.predict_proba(X_input)[0]
            
            # 4. Procesar probabilidades
            probabilidades_dict = {i: prob * 100 for i, prob in enumerate(probabilidades_raw)}
            
            # 5. Identificar la clase más probable para log
            pred_clase_idx = model_cb.predict(X_input)[0] # Predict the class index
            pred_clase_label = label_mapping[int(pred_clase_idx)] # Map index to actual label
            
            # Find the highest probability among failure types (excluding class 0 if it exists and means "No Falla")
            max_prob_falla = 0.0
            nombre_falla_max_prob = ""
            for class_idx, prob_val in probabilidades_dict.items():
                if class_idx != 0: # Assuming class 0 is 'No Falla'
                    if prob_val > max_prob_falla:
                        max_prob_falla = prob_val
                        nombre_falla_max_prob = label_mapping[class_idx]
            
            # If no non-zero class had a significant probability, default to the predicted class label
            if nombre_falla_max_prob == "":
                nombre_falla_max_prob = pred_clase_label
            
            prob_falla_display = probabilidades_dict[int(pred_clase_idx)]
            nombre_falla_display = pred_clase_label

            # 6. ACTUALIZAR EL MAPA DE RIESGO para la estación del tweet
            estatus_estaciones[estacion].update(probabilidades_dict)
            estatus_estaciones[estacion]['hora'] = datetime.now().strftime('%H:%M')

            nuevos_reportes.append(f"Tweet en {estacion}: '{tweet_text}' -> {nombre_falla_display} ({prob_falla_display:.1f}%)")

            # 7. ALERTA CRÍTICA
            # Trigger alert if the predicted class is not the "No Falla" class (assuming 0) and its probability is high
            if int(pred_clase_idx) != 0 and prob_falla_display > UMBRAL_ALERTA:
                print(f"\n🚨🚨 ALERTA CRÍTICA EN {estacion.upper()} 🚨🚨")
                print(f"   Tipo de Falla: {nombre_falla_display}")
                print(f"   Certeza del modelo: {prob_falla_display:.2f}%")
                print(f"   Tweet origen: \"{tweet_text}\"\n")
                time.sleep(2)

        # Mostrar tablero actualizado
        # limpiar_consola() # Descomenta si quieres que se limpie la pantalla
        mostrar_tablero()
        
        print("\nLog reciente:")
        for r in nuevos_reportes:
            print(" >", r)

        time.sleep(INTERVALO)

except KeyboardInterrupt:
    print("\nFin del monitoreo.")
