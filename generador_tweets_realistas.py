import random
import json
import re

# ================= 1. COMPONENTES EXTRAÍDOS DEL DATASET REAL (DATOS_CRUDOS) =================

# A. ESTACIONES REALES DE LA LÍNEA 1
estaciones_L1 = [
    "Observatorio", "Tacubaya", "Juanacatlán", "Chapultepec", "Sevilla",
    "Insurgentes", "Cuauhtémoc", "Balderas", "Salto del Agua", "Isabel la Católica",
    "Pino Suárez", "Merced", "Candelaria", "San Lázaro", "Moctezuma", "Balbuena",
    "Boulevard Puerto Aéreo", "Gómez Farías", "Zaragoza", "Pantitlán"
]

# B. EMOCIONES / RUIDO (Comentarios sobre congestión, quejas y vida diaria)
emociones_ruido = [
    "Ya quiero llegar a mi casa. 😭", 
    "¡Manden trenes vacíos! Es urgente.", 
    "Estoy sudando horrible. El calor es insoportable.", 
    "¿Por qué siempre pasa esto a esta hora?", 
    "¡Qué frustrante y qué mal servicio!",
    "No es posible el retraso de hoy.",
    "Todo el mundo empujando horrible.",
    "El internet no funciona aquí."
]

# C. REPORTE DE FALLA (Núcleo del problema, usando solo 1 o 2 palabras clave fuertes)
# La clave (0 a 4) debe coincidir con tu TIPOS_FALLA: 0:Normal, 1:Humo, 2:Agua, 3:Eléctrica, 4:Mecánica
reportes_falla = {
    # 0: Normal/Saturación (Simula la clase Normal)
    0: [
        "está imposible, la fila para entrar llega hasta las escaleras de la calle.",
        "llevamos 15 minutos parados sin avanzar. La saturación es brutal.",
        "hay un mar de gente y el tren no llega desde hace rato.",
        "el servicio va lentísimo, ya es hora pico y es un infierno."
    ],
    # 1: Humo (Clave: humo, quemado)
    1: [
        "hay un olor a **humo** muy fuerte, como a cable quemado.",
        "se vio una flama y ahora está saliendo **humo** denso de las vías.",
        "el vagón huele a **quemado** y la gente tose. @MetroCDMX"
    ],
    # 2: Agua (Clave: agua, filtración, inundada)
    2: [
        "la estación está **inundada**, el agua llega al andén. Es terrible.",
        "hay una **filtración** gigante en el techo, parece que va a llover adentro. 💧",
        "están cayendo chorros de **agua** cerca de las taquillas."
    ],
    # 3: Eléctrica (Clave: chispazo, apagón, parpadeando)
    3: [
        "hubo un **chispazo** y las luces del tren se **apagaron**. Fue un susto.",
        "el motor hizo un **estallido** y ahora va dando tirones. Pienso en la luz.",
        "las luces están **parpadeando** y huele a algo raro, eléctrico."
    ],
    # 4: Mecánica (Clave: freno, puertas, ruido, detenido)
    4: [
        "el tren hizo un **frenado** de emergencia muy violento. Nos caímos todos.",
        "las **puertas** están fallando y se abrieron solas. ¡Qué peligro!",
        "el tren lleva un ruido **mecánico** horrible y se detiene cada 5 metros."
    ]
}

# D. Tipos de Usuario (Para el campo 'user')
tipos_usuario = [
    "Godin_Atorado", "Usuario_Cansado", "Metro_Queja_L1", "Reporte_Fallas", 
    "Alerta_Metro_CDMX", "Ciudadano_Molesto", "Reporte_Diario"
]

# ================= NUEVO: COMPONENTES DEL JSON =================

def cargar_frases_json(json_path="features.json"):
    """
    Carga y extrae frases reales del JSON para enriquecer los reportes
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        frases_por_tipo = {
            0: [],  # Normal/Saturación
            1: [],  # Humo
            2: [],  # Agua  
            3: [],  # Eléctrica
            4: []   # Mecánica
        }
        
        for item in datos:
            resumen = item['X_features']['resumen_para_roberta']
            
            # Extraer frases cortas y clasificarlas
            oraciones = re.split(r'[.!?]', resumen)
            for oracion in oraciones:
                oracion = oracion.strip()
                if len(oracion) < 20 or len(oracion) > 150:
                    continue
                    
                # Clasificar por palabras clave (igual que tu lógica actual)
                oracion_lower = oracion.lower()
                
                if any(palabra in oracion_lower for palabra in ['humo', 'quemado', 'flama', 'llama']):
                    frases_por_tipo[1].append(oracion)
                elif any(palabra in oracion_lower for palabra in ['agua', 'inundada', 'filtración', 'inundación', 'lluvia', 'charco']):
                    frases_por_tipo[2].append(oracion)
                elif any(palabra in oracion_lower for palabra in ['eléctric', 'chispazo', 'apagón', 'luz', 'voltaje', 'corto']):
                    frases_por_tipo[3].append(oracion)
                elif any(palabra in oracion_lower for palabra in ['mecánic', 'freno', 'puerta', 'ruido', 'motor', 'frenado']):
                    frases_por_tipo[4].append(oracion)
                elif any(palabra in oracion_lower for palabra in ['saturación', 'lleno', 'gente', 'fila', 'espera', 'retraso']):
                    frases_por_tipo[0].append(oracion)
        
        # Limpiar frases duplicadas
        for tipo in frases_por_tipo:
            frases_por_tipo[tipo] = list(set(frases_por_tipo[tipo]))
            
        print(f"✅ Cargadas {sum(len(frases) for frases in frases_por_tipo.values())} frases reales del JSON")
        return frases_por_tipo
        
    except FileNotFoundError:
        print("⚠️  No se encontró features.json. Usando solo frases sintéticas.")
        return None

# Cargar frases del JSON al inicio
frases_json = cargar_frases_json()

def obtener_reporte_mejorado(clase_falla):
    """
    Obtiene un reporte que puede venir del JSON o de las frases sintéticas
    """
    # 60% de probabilidad de usar frases del JSON si están disponibles
    if frases_json and random.random() < 0.6 and frases_json[clase_falla]:
        frase_json = random.choice(frases_json[clase_falla])
        
        # Acortar y adaptar frases largas del JSON
        if len(frase_json) > 120:
            palabras = frase_json.split()
            if len(palabras) > 15:
                frase_json = ' '.join(palabras[:12]) + "..."
        
        # Hacer más coloquial (como tus frases sintéticas)
        mejoras = {
            "usuarios reportan": "reportan",
            "se ha detectado": "hay", 
            "se encuentra": "está",
            "se están realizando": "hacen"
        }
        
        for formal, coloquial in mejoras.items():
            frase_json = frase_json.replace(formal, coloquial)
            
        return frase_json
    else:
        # Fallback a frases sintéticas originales
        return random.choice(reportes_falla[clase_falla])

# ================= 2. LÓGICA DE COMBINACIÓN (MEJORADA) =================

def generar_tweet_simulado(num_tweets=15):
    """
    Genera una lista de tweets simulados combinando componentes aleatorios 
    de estaciones, reportes de falla y ruido emocional.
    """
    tweets_simulados = []
    
    # Garantizamos al menos 2 reportes de cada clase crítica (1, 2, 3, 4)
    # y el resto (num_tweets - 8) serán de clase Normal (0) para simular ruido.
    base_clases = [1, 1, 2, 2, 3, 3, 4, 4] 
    
    clases_extra = [0] * (num_tweets - len(base_clases))
    clases_a_generar = base_clases + clases_extra
    random.shuffle(clases_a_generar)

    for i in range(num_tweets):
        # 1. Seleccionar la clase de falla y el reporte base (C)
        clase_falla = clases_a_generar[i] if i < len(clases_a_generar) else random.randint(0, 4)
        reporte_base = obtener_reporte_mejorado(clase_falla)
        
        # 2. Seleccionar componentes aleatorios
        estacion = random.choice(estaciones_L1) # A
        ruido = random.choice(emociones_ruido)  # B
        usuario = random.choice(tipos_usuario)  # D
        
        # 3. Construir el texto final (Formato típico de reporte)
        tweet_text = f"@MetroCDMX en **{estacion}**, {reporte_base}. {ruido}"
        
        # 4. Construir el JSON simulado (Añadiendo la etiqueta 'clase_real' para validación)
        tweet_json = {
            "source": "Twitter",
            "user": f"{usuario}_{random.randint(100, 999)}",
            "text": tweet_text,
            "geo_enabled": random.choice([True, False, False]),
            # ESTO ES SOLO PARA VALIDACIÓN, NO SE LO PASES AL MODELO EN PRODUCCIÓN:
            "clase_real": clase_falla
        }
        tweets_simulados.append(tweet_json)
        
    return tweets_simulados

# ================= 3. EJEMPLO DE USO =================
if __name__ == '__main__':
    N_EJEMPLOS = 15
    nuevos_tweets = generar_tweet_simulado(num_tweets=N_EJEMPLOS)

    print(f"### {N_EJEMPLOS} Tweets Simulados Mejorados con JSON ###\n---")
    
    for tweet in nuevos_tweets:
        clase_str = {0:'NORMAL', 1:'HUMO', 2:'AGUA', 3:'ELEC', 4:'MEC'}.get(tweet['clase_real'], 'N/A')
        origen = "JSON" if any(palabra in tweet['text'].lower() for palabra in [
            'reportan', 'saturación', 'filtración', 'detenido', 'operación'
        ]) and not any(palabra in tweet['text'] for palabra in ['**humo**', '**agua**', '**chispazo**', '**frenado**']) else "SINTÉTICO"
        
        print(f"[{clase_str}] [{origen:8}] [{tweet['user']}] -> {tweet['text']}")

    print("\n---")
    print("JSON COMPLETO (Ideal para inyectar en tu simulación):")
    # Imprime en JSON con indentación y caracteres especiales correctos
    print(json.dumps(nuevos_tweets, indent=2, ensure_ascii=False))