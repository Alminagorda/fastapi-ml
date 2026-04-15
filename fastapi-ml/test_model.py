#!/usr/bin/env python3
"""
Script de prueba para verificar que el modelo se carga correctamente
"""
import os
import sys

print("=" * 60)
print("Test de Carga del Modelo")
print("=" * 60)

# Verificar versiones
print("\n📦 Versiones de librerías:")
import tensorflow as tf
print(f"  - TensorFlow: {tf.__version__}")

from tensorflow import keras
print(f"  - Keras: {keras.__version__}")

import joblib
print(f"  - Joblib: {joblib.__version__}")

import numpy as np
print(f"  - NumPy: {np.__version__}")

# Verificar archivos
print("\n📁 Verificarse archivos:")
archivos_requeridos = [
    'autoencoder_antamina.keras',
    'scaler_antamina.pkl',
    'model_metadata.json'
]

for archivo in archivos_requeridos:
    if os.path.exists(archivo):
        size_mb = os.path.getsize(archivo) / 1024 / 1024
        print(f"  ✓ {archivo} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {archivo} (NO ENCONTRADO)")
        sys.exit(1)

# Cargar modelo
print("\n📦 Cargando modelo...")
try:
    model = keras.models.load_model('autoencoder_antamina.keras')
    print(f"  ✓ Modelo cargado")
    print(f"    - Input shape: {model.input_shape}")
    print(f"    - Output shape: {model.output_shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cargar scaler
print("\n📊 Cargando scaler...")
try:
    scaler = joblib.load('scaler_antamina.pkl')
    print(f"  ✓ Scaler cargado ({type(scaler).__name__})")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Cargar metadata
print("\n📋 Cargando metadatos...")
try:
    import json
    with open('model_metadata.json') as f:
        metadata = json.load(f)
    print(f"  ✓ Metadatos cargados")
    print(f"    - Modelo: {metadata.get('model_name', 'N/A')}")
    print(f"    - Threshold: {metadata['anomaly_threshold']:.6f}")
    print(f"    - Features: {len(metadata['features_used'])}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Prueba de predicción
print("\n🧪 Prueba de predicción...")
try:
    import pandas as pd
    
    features = metadata['features_used']
    test_data = {f: 0.0 for f in features}
    df = pd.DataFrame([test_data])
    
    X_scaled = scaler.transform(df[features])
    reconstruction = model.predict(X_scaled, verbose=0)
    mse = float(np.mean(np.power(X_scaled - reconstruction, 2)))
    
    print(f"  ✓ Predicción exitosa")
    print(f"    - MSE: {mse:.6f}")
    print(f"    - Es anomalía: {mse > metadata['anomaly_threshold']}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TODAS LAS PRUEBAS PASARON")
print("=" * 60)
