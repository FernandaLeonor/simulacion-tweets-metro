import json
import numpy as np
import random

# Generamos 1000 datos falsos
datos_fake = []
estaciones = ["Tacubaya", "Polanco", "Mixcoac", "Barranca", "Zapata"]
dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"]

for i in range(1000):
    # Simulamos vector de 768 dimensiones
    vector = np.random.rand(768).tolist() 
    
    dato = {
        "vector_tweet": vector,
        "estacion_tweet": random.choice(estaciones),
        "hora_tweet": random.randint(5, 23),
        "dia_semana": random.choice(dias),
        "target_falla": random.randint(0, 4) # 5 clases de falla
    }
    datos_fake.append(dato)

with open('datos_entrenamiento.json', 'w') as f:
    json.dump(datos_fake, f)

print("✅ Archivo 'datos_entrenamiento.json' generado con 1000 registros pesados.")