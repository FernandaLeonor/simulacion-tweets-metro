import random
import time
import pandas as pd
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
# Importar tu generador mejorado
from src.data_generation.realistic_tweet_generator import generar_tweet_simulado 

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

# Definiciones de fallas para mostrar en el tablero (para el modelo binario)
TIPOS_FALLA = {0:"No Falla", 1:"Falla Detectada"}
NOMBRES_CORTOS_FALLA = {0:"NO FALLA", 1:"FALLA"} # Para la tabla

# ================= ESTADO DEL SISTEMA MODIFICADO =================
# Diccionario para guardar el último estado de CADA PROBABILIDAD para cada estación
def get_initial_probs():
    # Iniciamos con 100% No Falla y 0% Falla Detectada
    return {0: 100.0, 1: 0.0, 'hora': '-'}

estatus_estaciones = {est: get_initial_probs() for est in estaciones_L1}

# ================= CARGAR MODELOS =================
print("Cargando cerebro...")
model_cb = CatBoostClassifier()
model_cb.load_model("models/modelo_deteccion_falla.cbm") # Usar el modelo de detección de falla
embed_model = SentenceTransformer('xlm-roberta-base')
print("✅ Sistemas listos. Iniciando monitoreo...")

# ================= FUNCIONES =================
def limpiar_consola():
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_tablero():

    print("\n" + "="*85)

    print(f"   MONITOREO LÍNEA 1 - {datetime.now().strftime('%H:%M:%S')}")

    print("="*85)

    

    # 1. Cabecera con todos los tipos de falla

    header = f"{'ESTACIÓN':<20} | {'HORA':<4} | "

    prob_headers = [NOMBRES_CORTOS_FALLA[i] for i in range(len(TIPOS_FALLA))] # Use len(TIPOS_FALLA)

    header += " | ".join(f"{h:<8}" for h in prob_headers) # Adjusted spacing for 2 classes

    print(header)

    print("-" * 85)

    

    # 2. Iterar sobre TODAS las estaciones (porque deben estar fijas)

    for estacion in estaciones_L1:

        datos = estatus_estaciones[estacion]

        

        # 3. Formatear las probabilidades

        prob_strings = []

        for i in range(len(TIPOS_FALLA)): # Use len(TIPOS_FALLA)

            prob = datos[i]

            

            # 4. Colores basados en la probabilidad de falla (solo para 'Falla Detectada' que es i=1)

            color = "\033[92m" # Verde (default o No Falla)

            if i == 1 and prob > UMBRAL_ALERTA: # Only for "Falla Detectada" (class 1)

                color = "\033[91m" # Rojo Intenso (Alerta)

            elif i == 1 and prob > 50:

                color = "\033[93m" # Amarillo (Advertencia)

                

            reset = "\033[0m"

            

            # Formatear el porcentaje

            prob_str = f"{prob:.1f}%"

            prob_strings.append(f"{color}{prob_str:<8}{reset}") # Adjusted spacing

        

        # 5. Imprimir la fila

        prob_display = " | ".join(prob_strings)

        print(f"{estacion:<20} | {datos['hora']:<4} | {prob_display}")

    

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
            # El orden de las columnas debe coincidir con el del entrenamiento del modelo
            model_feature_names = model_cb.feature_names_
            X_input = pd.DataFrame([features_dict], columns=model_feature_names)

            # 3. Predicción de probabilidades (predict_proba)
            probabilidades_raw = model_cb.predict_proba(X_input)[0]
            
            # 4. Procesar probabilidades
            probabilidades_dict = {i: prob * 100 for i, prob in enumerate(probabilidades_raw)}
            
            # 5. Identificar la clase más probable para log
            pred_clase = model_cb.predict(X_input)[0] # Predict the class (0 or 1) directly
            prob_falla_max = probabilidades_dict[1] # Probability of "Falla Detectada" (class 1)
            nombre_falla = TIPOS_FALLA[pred_clase]

            # 6. ACTUALIZAR EL MAPA DE RIESGO para la estación del tweet
            # Actualizamos las probabilidades completas para la estación que reportó
            estatus_estaciones[estacion].update(probabilidades_dict)
            estatus_estaciones[estacion]['hora'] = datetime.now().strftime('%H:%M') # Update to current time

            nuevos_reportes.append(f"Tweet en {estacion}: '{tweet_text}' -> {nombre_falla} ({prob_falla_max:.1f}%)")

            # 7. ALERTA CRÍTICA
            if pred_clase == 1 and prob_falla_max > UMBRAL_ALERTA: # Alert only if predicted class is 1 (Falla Detectada)
                print(f"\n🚨🚨 ALERTA CRÍTICA EN {estacion.upper()} 🚨🚨")
                print(f"   Reporte: {nombre_falla}")
                print(f"   Certeza del modelo: {prob_falla_max:.2f}%")
                print(f"   Tweet origen: \"{tweet_text}\"\n")
                time.sleep(2) # Pausa dramática para ver la alerta

        # Mostrar tablero actualizado
        # limpiar_consola() # Descomenta si quieres que se limpie la pantalla
        mostrar_tablero()
        
        print("\nLog reciente:")
        for r in nuevos_reportes:
            print(" >", r)

        time.sleep(INTERVALO)

except KeyboardInterrupt:
    print("\nFin del monitoreo.")
