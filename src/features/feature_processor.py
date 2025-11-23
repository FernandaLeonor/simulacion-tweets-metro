
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Cargar modelo de embeddings
print("Cargando modelo de embeddings...")
model = SentenceTransformer('xlm-roberta-base')

# 2. Cargar datos del archivo JSON
print("Cargando features.json...")
with open('data/processed/features.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. Procesar datos y generar embeddings
processed_data = []
print(f"Procesando {len(data)} registros...")

for entry in data:
    # Ignorar batch_meta
    if 'X_features' in entry:
        x_features = entry['X_features']
        
        # Extraer el texto a procesar
        texto_a_procesar = x_features.get('resumen_para_roberta', '')
        
        # Generar embedding
        embedding = model.encode(texto_a_procesar).tolist()
        
        # Preparar los datos para el CSV
        record = {}
        
        # Añadir features de X_features
        record['station'] = x_features.get('station')
        record.update(x_features.get('features_numericas_promedio', {}))

        # Añadir el embedding como columnas separadas
        embedding_dict = {f'embedding_{i}': val for i, val in enumerate(embedding)}
        record.update(embedding_dict)
        
        # Añadir Y_labels
        if 'Y_labels' in entry:
            record.update(entry['Y_labels'])

        processed_data.append(record)

# 4. Convertir a DataFrame de pandas
print("Convirtiendo datos a DataFrame...")
df = pd.DataFrame(processed_data)

# 5. Guardar en un archivo CSV
output_filename = 'data/processed/features_embeddings.csv'
print(f"Guardando datos en {output_filename}...")
df.to_csv(output_filename, index=False, encoding='utf-8')

print(f"✅ Proceso completado. Archivo '{output_filename}' generado con éxito.")
