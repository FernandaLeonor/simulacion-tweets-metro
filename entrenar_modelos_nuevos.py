
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import json

print("Iniciando el proceso de entrenamiento de modelos...")

# 1. Cargar los datos
try:
    df = pd.read_csv('features_embeddings.csv')
    print("CSV 'features_embeddings.csv' cargado correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'features_embeddings.csv'. Asegúrate de generarlo primero.")
    exit()

# 2. Preparación de datos
print("Preparando los datos...")

# La columna 'station' es categórica
if 'station' in df.columns:
    df['station'] = df['station'].astype('category')

# Separar features (X) de las variables objetivo (y)
y1 = df['falla_detectada']
y2 = df['target_falla']

# Definir columnas de features
feature_columns = [col for col in df.columns if col not in ['falla_detectada', 'target_falla']]
X = df[feature_columns]

# Identificar columnas categóricas y numéricas para el preprocesamiento
categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype.name == 'category']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Imputar valores nulos en columnas numéricas
imputer = SimpleImputer(strategy='mean')
X[numerical_features] = imputer.fit_transform(X[numerical_features])
print("Valores nulos en características numéricas rellenados con la media.")

# --- Modelo 1: Detección de Falla (Binario) ---
print("\n--- Entrenando Modelo 1: Detección de Falla (Sí/No) ---")

# Convertir booleano a 0/1
y1_binary = y1.astype(int)

# División de datos para el modelo 1
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1_binary, test_size=0.25, random_state=42, stratify=y1_binary)

# Crear y entrenar el clasificador CatBoost
print("Entrenando CatBoostClassifier para detección...")
model1 = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    verbose=100,
    cat_features=categorical_features_indices,
    early_stopping_rounds=50
)
model1.fit(X_train1, y_train1, eval_set=(X_test1, y_test1))

# Evaluar el modelo 1
print("\nEvaluación del Modelo 1:")
y_pred1 = model1.predict(X_test1)
accuracy1 = accuracy_score(y_test1, y_pred1)
print(f"Precisión (Accuracy): {accuracy1:.4f}")
print("Reporte de Clasificación:")
print(classification_report(y_test1, y_pred1))

# Guardar el modelo 1
model1.save_model('modelo_deteccion_falla.cbm')
print("✅ Modelo 1 guardado como 'modelo_deteccion_falla.cbm'")


# --- Modelo 2: Clasificación del Tipo de Falla (Multiclase) ---
print("\n--- Entrenando Modelo 2: Clasificación del Tipo de Falla ---")

# Filtrar el dataset para incluir solo los casos donde hubo una falla
df_fallas = df[df['falla_detectada'] == True].copy()
df_fallas = df_fallas.dropna(subset=['target_falla']) # Eliminar filas si el target es nulo

if not df_fallas.empty:
    X_fallas = df_fallas[feature_columns]
    y_fallas_raw = df_fallas['target_falla']

    # Imputar de nuevo por si acaso el subconjunto tiene diferentes nulos
    X_fallas[numerical_features] = imputer.fit_transform(X_fallas[numerical_features])

    # Codificar las etiquetas de texto del target a números
    le = LabelEncoder()
    y_fallas_encoded = le.fit_transform(y_fallas_raw)
    
    # Guardar el mapeo de etiquetas
    label_mapping = {index: label for index, label in enumerate(le.classes_)}
    with open('label_encoding.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)
    print(f"Mapeo de etiquetas para el Modelo 2 guardado en 'label_encoding.json'. Clases: {le.classes_}")

    # Debido al bajo número de muestras con fallas, no se creará un conjunto de prueba.
    # El modelo se entrenará con todos los datos de fallas disponibles.
    print("Advertencia: El número de muestras de fallas es muy bajo. El Modelo 2 se entrenará con todos los datos disponibles (sin división de prueba).")
    
    X_train2 = X_fallas
    y_train2 = y_fallas_encoded

    # Crear y entrenar el clasificador CatBoost
    print("\nEntrenando CatBoostClassifier para clasificación de tipo de falla...")
    model2 = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=4, # Reducir complejidad para dataset pequeño
        loss_function='MultiClass',
        verbose=100,
        cat_features=[X_fallas.columns.get_loc(col) for col in X_fallas.columns if X_fallas[col].dtype.name == 'category']
    )
    # No se usa eval_set ya que no hay conjunto de prueba
    model2.fit(X_train2, y_train2)

    # No hay evaluación posible sin un conjunto de prueba.
    print("\nEvaluación del Modelo 2 omitida debido a la falta de un conjunto de prueba.")

    # Guardar el modelo 2
    model2.save_model('modelo_clasificacion_falla.cbm')
    print("✅ Modelo 2 guardado como 'modelo_clasificacion_falla.cbm'")

else:
    print("⚠️ No hay suficientes datos de fallas para entrenar el segundo modelo.")

print("\nProceso de entrenamiento finalizado.")
