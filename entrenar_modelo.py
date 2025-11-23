import json
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. CARGAR DATOS 
print("Cargando dataset...")
try:
    with open('datos_entrenamiento.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f" Se cargaron {len(data)} registros.")
except FileNotFoundError:
    print("  ERROR: No encuentro 'datos_entrenamiento.json'. Créalo o pídelo.")
    exit()

# 2. PREPARAR LA MATRIZ DE CARACTERÍSTICAS (X)
print("⚙️  Procesando vectores y columnas...")

# A. Desglosamos el vector de 768 números en 768 columnas
# Esto convierte la lista [0.1, 0.2...] en columnas col_0, col_1...
lista_vectores = [d['vector_tweet'] for d in data]
df_vectores = pd.DataFrame(lista_vectores)

df_vectores.columns = [f"v_{i}" for i in range(df_vectores.shape[1])]


# B. Sacamos los datos de contexto (DÓNDE y CUÁNDO)
lista_meta = []
for d in data:
    lista_meta.append({
        'estacion': d['estacion_tweet'],  # TEXTO (Importante para el "Dónde")
        'hora': d['hora_tweet'],          # NÚMERO
        'dia': d['dia_semana']            # TEXTO
    })
df_meta = pd.DataFrame(lista_meta)

# C. Unimos todo en una sola tabla gigante
# X tendrá: 768 cols de vector + 1 col estacion + 1 col hora + 1 col dia
X = pd.concat([df_vectores, df_meta], axis=1)

# 3. PREPARAR EL OBJETIVO (Y) - QUÉ FALLA
# Esto es lo que aprendemos del Excel de Siemens
y = [d['target_falla'] for d in data]

# 4. CONFIGURAR CATBOOST
# Le decimos cuáles columnas son texto para que él las maneje internamente
# Las columnas del vector son números (0...767), las de texto son 'estacion' y 'dia'
cat_features_indices = ['estacion', 'dia']

print("🧠 Iniciando entrenamiento del modelo...")

# Definimos el modelo
model = CatBoostClassifier(
    iterations=500,           # Cuántas veces repasa los datos
    learning_rate=0.1,        # Qué tan rápido aprende
    depth=6,                  # Profundidad del árbol
    loss_function='MultiClass', # ¡IMPORTANTE! Para distinguir Fuego vs Agua vs Electricidad
    eval_metric='Accuracy',   # Queremos maximizar la precisión
    verbose=50                # Nos imprime info cada 50 pasos
)

# 5. ENTRENAR (FIT)
# Separamos un cachito (20%) para validar que aprendió bien
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(
    X_train, y_train,
    cat_features=cat_features_indices, # Aquí le pasamos las columnas de texto
    eval_set=(X_test, y_test),         # Para ver si mejora
    early_stopping_rounds=50           # Si deja de mejorar, se detiene
)

# 6. VALIDACIÓN RÁPIDA
print("\nReporte de Clasificación (¿Qué tan bien aprendió?):")
predicciones = model.predict(X_test)
print(classification_report(y_test, predicciones))

# 7. GUARDAR EL MODELO
nombre_archivo = "modelo_final.cbm"
model.save_model(nombre_archivo)
print(f"\n¡Modelo guardado como '{nombre_archivo}'.")
print("   Ahora puedes usar este archivo para predecir en el mapa.")