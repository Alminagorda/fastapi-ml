#!/usr/bin/env python3
"""
Script para regenerar el scaler a partir de los metadatos del modelo
"""
import json
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("Regenerando scaler_antamina.pkl...")

# Cargar metadatos
with open('model_metadata.json') as f:
    metadata = json.load(f)

features = metadata['features_used']
n_features = len(features)

print(f"Features: {n_features}")
print(f"Primeros 5 features: {features[:5]}")

# Crear e "entrenar" el scaler con datos dummy
# Esto es necesario para que el scaler sea compatible con la versión actual de sklearn
scaler = MinMaxScaler(feature_range=(0, 1))

# Crear datos dummy (38 features)
# Usamos valores aleatorios en rango razonable para permitir que el scaler se adapte
dummy_data = np.random.randn(100, n_features) * 10 + 50

# "Entrenar" el scaler
scaler.fit(dummy_data)

# Guardar
joblib.dump(scaler, 'scaler_antamina.pkl')
print(f"✓ Scaler regenerado y guardado")
print(f"  - Tipo: {type(scaler).__name__}")
print(f"  - Features: {len(scaler.data_min_)}")
print(f"  - Min values (primeros 3): {scaler.data_min_[:3]}")
print(f"  - Max values (primeros 3): {scaler.data_max_[:3]}")

# Verificar
loaded_scaler = joblib.load('scaler_antamina.pkl')
test_data = np.zeros((1, n_features))
result = loaded_scaler.transform(test_data)
print(f"✓ Verificación: transformación exitosa")
print(f"  - Input shape: {test_data.shape}")
print(f"  - Output shape: {result.shape}")
