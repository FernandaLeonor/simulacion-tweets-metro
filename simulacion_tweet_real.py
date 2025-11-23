import random
import time
import pandas as pd
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier

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

# Definiciones de fallas para mostrar en el tablero
TIPOS_FALLA = {0:"Normal",1:"Humo",2:"Agua",3:"Eléctrica",4:"Mecánica"}
NOMBRES_CORTOS_FALLA = {0:"NORM",1:"HUMO",2:"AGUA",3:"ELEC",4:"MEC"} # Para la tabla

# Tus templates
tweets_templates = {
    'normal': ["Servicio fluido sin retrasos","Todo normal en la estación","Tren pasando sin problemas"],
    'humo': ["Huele a quemado en el vagón","Sale humo del tren","Olor extraño a quemado"],
    'agua': ["Inundación en la estación","Mucha agua en los andenes","Goteras por todos lados"],
    'electrico': ["Se fue la luz en la estación","Luces parpadeando","Apagón total en el metro"],
    'mecanico': ["Puerta atascada, no abre","Tren detenido por falla mecánica","Problema con los frenos"]
}
# Probabilidades de generación (Simulamos que a veces falla algo)
probabilidades_tweets = {
    'normal': 0.6, 'humo': 0.1, 'agua': 0.1, 'electrico': 0.1, 'mecanico': 0.1
}

# ================= ESTADO DEL SISTEMA MODIFICADO =================
# Diccionario para guardar el último estado de CADA PROBABILIDAD para cada estación
def get_initial_probs():
    # Iniciamos con 100% Normal y 0% en el resto
    return {0: 100.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 'hora': '-'}

estatus_estaciones = {est: get_initial_probs() for est in estaciones_L1}

# ================= CARGAR MODELOS =================
print("Cargando cerebro...")
model_cb = CatBoostClassifier()
model_cb.load_model("modelo_final.cbm") # Asegúrate que sea el RECIÉN entrenado
embed_model = SentenceTransformer('xlm-roberta-base')
print("✅ Sistemas listos. Iniciando monitoreo...")

# ================= FUNCIONES =================
def generar_tweet():
    tipo = random.choices(list(probabilidades_tweets.keys()), weights=list(probabilidades_tweets.values()))[0]
    return random.choice(tweets_templates[tipo]), tipo

def limpiar_consola():
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_tablero():
    print("\n" + "="*85)
    print(f"   MONITOREO LÍNEA 1 - {datetime.now().strftime('%H:%M:%S')}")
    print("="*85)
    
    # 1. Cabecera con todos los tipos de falla
    header = f"{'ESTACIÓN':<20} | {'HORA':<4} | "
    prob_headers = [NOMBRES_CORTOS_FALLA[i] for i in range(5)]
    header += " | ".join(f"{h:<6}" for h in prob_headers)
    print(header)
    print("-" * 85)
    
    # 2. Iterar sobre TODAS las estaciones (porque deben estar fijas)
    for estacion in estaciones_L1:
        datos = estatus_estaciones[estacion]
        
        # 3. Formatear las probabilidades
        prob_strings = []
        for i in range(5):
            prob = datos[i]
            
            # 4. Colores basados en la probabilidad de falla (solo para las fallas 1-4)
            color = "\033[92m" # Verde (default o Normal)
            if i != 0 and prob > UMBRAL_ALERTA:
                color = "\033[91m" # Rojo Intenso (Alerta)
            elif i != 0 and prob > 50:
                color = "\033[93m" # Amarillo (Advertencia)
                
            reset = "\033[0m"
            
            # Formatear el porcentaje
            # Lo hacemos un poco más pequeño para que quepa bien
            prob_str = f"{prob:.1f}%"
            prob_strings.append(f"{color}{prob_str:<6}{reset}")

        # 5. Imprimir la fila
        prob_display = " | ".join(prob_strings)
        print(f"{estacion:<20} | {datos['hora']:<4} | {prob_display}")
    
    print("="*85)

# ================= LOOP PRINCIPAL =================
try:
    while True:
        # Generamos tweets aleatorios
        n = random.randint(*N_TWEETS)
        
        nuevos_reportes = []

        for _ in range(n):
            tweet_text, tipo_real = generar_tweet()
            estacion = random.choice(estaciones_L1) # Estación aleatoria
            hora = random.randint(6, 22)
            
            # 1. Vector embedding
            vector = embed_model.encode(tweet_text).tolist()

            # 2. DataFrame (CON NOMBRES DE COLUMNAS FORZADOS)
            df_vect = pd.DataFrame([vector])
            df_vect.columns = [f"v_{i}" for i in range(df_vect.shape[1])]

            df_meta = pd.DataFrame([{'estacion': estacion, 'hora': hora, 'dia': random.choice(dias_semana)}])
            X_input = pd.concat([df_vect, df_meta], axis=1)

            # 3. Predicción de probabilidades (predict_proba)
            probabilidades_raw = model_cb.predict_proba(X_input)[0]
            
            # 4. Procesar probabilidades
            probabilidades_dict = {i: prob * 100 for i, prob in enumerate(probabilidades_raw)}
            
            # 5. Identificar la clase más probable para log
            pred_clase = max(probabilidades_dict, key=probabilidades_dict.get)
            prob_falla_max = probabilidades_dict[pred_clase]
            nombre_falla = TIPOS_FALLA[pred_clase]

            # 6. ACTUALIZAR EL MAPA DE RIESGO para la estación del tweet
            # Actualizamos las probabilidades completas para la estación que reportó
            estatus_estaciones[estacion].update(probabilidades_dict)
            estatus_estaciones[estacion]['hora'] = f"{hora:02}:00"
            
            nuevos_reportes.append(f"Tweet en {estacion}: '{tweet_text}' -> {nombre_falla} ({prob_falla_max:.1f}%)")

            # 7. ALERTA CRÍTICA
            if prob_falla_max > UMBRAL_ALERTA and pred_clase != 0:
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