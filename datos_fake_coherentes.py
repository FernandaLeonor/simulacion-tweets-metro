import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Cargar modelo
print("Cargando modelo de embeddings...")
model = SentenceTransformer('xlm-roberta-base')

# 2. SEMILLAS MEJORADAS Y AMPLIADAS
# Incluyen tus templates originales, los de la simulación y ejemplos reales de Twitter
semillas = {
    0: [ # Normal
        "El servicio es fluido", "Todo bien en la linea", "Avanza rápido", 
        "Sin novedades", "Llegué a tiempo", "Vagón vacío y limpio",
        "Metro operando normal", "Buen servicio hoy",
        "Servicio fluido sin retrasos", "Todo normal en la estación", 
        "Tren pasando sin problemas", "Avance continuo de trenes",
        "La afluencia es moderada", "Llegamos súper rápido a Pantitlán",
        "Sorprendentemente hoy el metro va bien"
    ],
    1: [ # Humo/Incendio
        "Huele a quemado", "Sale humo de las llantas", "Se está quemando algo",
        "Hay fuego en las vías", "Mucho humo en el andén", "Olor a caucho quemado",
        "Huele a quemado en el vagón", "Sale humo del tren", "Olor extraño a quemado",
        "Desalojaron el tren por humo", "Humo denso en la estación Balderas",
        "Se prendieron las zapatas", "Conato de incendio en vagón"
    ],
    2: [ # Agua/Inundación
        "Se está metiendo el agua", "Inundación en las escaleras", "Goteras en el techo",
        "Piso muy mojado y resbaloso", "Cascadas en la estación", "Andén inundado",
        "Inundación en la estación", "Mucha agua en los andenes", "Goteras por todos lados",
        "Cierran accesos por lluvia intensa", "El transbordo parece alberca",
        "Filtraciones de agua en el túnel", "Entrada de agua del exterior"
    ],
    3: [ # Eléctrico
        "Se fue la luz", "Vagón a oscuras", "Fallas de energía", 
        "Las lámparas parpadean", "Apagón en el túnel", "Sin electricidad",
        "Se fue la luz en la estación", "Luces parpadeando", "Apagón total en el metro",
        "Corto circuito en vías", "Suspensión del servicio por revisión eléctrica",
        "Tren detenido sin energía", "Bajón de luz y el tren se paró"
    ],
    4: [ # Mecánico
        "El tren hace ruidos raros", "Las puertas no cierran", "El tren se jalonea",
        "Estamos detenidos hace 10 min", "Falla en el motor", "Rechinido muy fuerte",
        "Puerta atascada, no abre", "Tren detenido por falla mecánica", "Problema con los frenos",
        "Desalojo por avería de tren", "Tren con marcha lenta por seguridad",
        "Se escuchó un golpe fuerte y nos paramos", "Problemas en el cierre de puertas"
    ]
}

estaciones = ["Observatorio", "Tacubaya", "Balderas", "Pino Suárez", "Pantitlán", "Merced", "Zaragoza"]
dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

datos_coherentes = []
N_REGISTROS = 1000  # Aumentamos a 1000 para que aprenda mejor la variabilidad

print(f"Generando {N_REGISTROS} registros con DATA AUGMENTATION...")

for _ in range(N_REGISTROS):
    # 1. Elegir categoría y frase base
    target = random.randint(0, 4)
    frase_base = random.choice(semillas[target])
    
    # 2. Generar vector base
    vector = model.encode(frase_base).tolist()
    
    # 3. DATA AUGMENTATION (El secreto para que funcione)
    # Añadimos un poco de ruido aleatorio al vector.
    # Esto simula que la frase es "ligeramente diferente" (sinónimos, typos, etc.)
    # Ruido gaussiano: media 0, desviación 0.02
    ruido = np.random.normal(0, 0.02, size=len(vector))
    vector_con_ruido = (np.array(vector) + ruido).tolist()
    
    dato = {
        "vector_tweet": vector_con_ruido, # Usamos el vector "sucio" para entrenar robustez
        "estacion_tweet": random.choice(estaciones),
        "hora_tweet": random.randint(5, 23),
        "dia_semana": random.choice(dias),
        "target_falla": target
    }
    datos_coherentes.append(dato)

# Guardar
with open('datos_entrenamiento.json', 'w', encoding='utf-8') as f:
    json.dump(datos_coherentes, f)

print("✅ Archivo 'datos_entrenamiento.json' mejorado generado.")
print("   Ahora entrena tu modelo de nuevo. La precisión bajará del 100% (quizás al 95%),")
print("   pero funcionará MUCHO mejor en la simulación real.")