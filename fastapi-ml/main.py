# ============================================================
# main.py — FastAPI Microservicio ML
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
from typing import Dict, Any, List

app = FastAPI(title="Antamina ML Service", version="1.0")

# CORS para que Spring Boot pueda llamarlo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CARGA EL MODELO UNA SOLA VEZ AL INICIAR
# ============================================================
print("="*60)
print("Cargando componentes del modelo...")
print("="*60)

try:
    cwd = os.getcwd()
    print(f"📁 Directorio actual: {cwd}")
    
    files = os.listdir('.')
    model_files = [f for f in files if f in ['autoencoder_antamina.keras', 'scaler_antamina.pkl', 'model_metadata.json']]
    print(f"✓ Archivos de modelo encontrados: {len(model_files)}/3")
    for f in model_files:
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  - {f}: {size:.2f} MB")
    
    print("\n📦 Cargando modelo Keras...")
    model = keras.models.load_model('autoencoder_antamina.keras')
    print(f"✓ Modelo cargado exitosamente")
    print(f"  - Inputs: {model.input_shape}")
    print(f"  - Outputs: {model.output_shape}")
    
    print("\n📊 Cargando scaler...")
    scaler = joblib.load('scaler_antamina.pkl')
    print(f"✓ Scaler cargado: {type(scaler).__name__}")
    
    print("\n📋 Cargando metadatos...")
    with open('model_metadata.json') as f:
        metadata = json.load(f)
    print(f"✓ Metadatos cargados")
    
    THRESHOLD = metadata['anomaly_threshold']
    FEATURES  = metadata['features_used']
    print(f"\n✅ Modelo completamente inicializado")
    print(f"  - Threshold: {THRESHOLD:.6f}")
    print(f"  - Features: {len(FEATURES)}")
    print("="*60)
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Archivo no encontrado")
    print(f"   {e}")
    print(f"\n   Archivos disponibles en {os.getcwd()}:")
    for f in sorted(os.listdir('.')):
        print(f"     - {f}")
    raise
except Exception as e:
    print(f"\n❌ ERROR al cargar modelo:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "threshold": THRESHOLD,
        "features":  len(FEATURES),
        "version":   metadata['model_name']
    }

# ----------------------------------------------------------
# PREDICCIÓN — recibe features y devuelve si es anomalía
# ----------------------------------------------------------
@app.post("/predict")
def predict(data: Dict[str, Any]):
    try:
        # Construye DataFrame con el orden correcto de features
        df = pd.DataFrame([data])

        # Agrega columnas faltantes con 0
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0

        df = df[FEATURES]

        # Escala y predice
        X_scaled     = scaler.transform(df)
        reconstruction = model.predict(X_scaled, verbose=0)
        mse          = float(np.mean(np.power(X_scaled - reconstruction, 2)))
        is_anomaly   = mse > THRESHOLD
        confidence   = min(float(mse / THRESHOLD), 1.0)

        return {
            "is_anomaly":           is_anomaly,
            "reconstruction_error": round(mse, 6),
            "threshold":            round(THRESHOLD, 6),
            "confidence":           round(confidence, 4),
            "severity": (
                "critical" if mse > THRESHOLD * 2.0 else
                "high"     if mse > THRESHOLD * 1.5 else
                "medium"   if mse > THRESHOLD       else
                "low"
            )
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------------------------------------------------
# PREDICCIÓN EN LOTE — para el dashboard en tiempo real
# ----------------------------------------------------------
@app.post("/predict/batch")
def predict_batch(data: List[Dict[str, Any]]):
    try:
        df           = pd.DataFrame(data)
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0
        df           = df[FEATURES]

        X_scaled       = scaler.transform(df)
        reconstruction = model.predict(X_scaled, verbose=0)
        errors         = np.mean(
            np.power(X_scaled - reconstruction, 2), axis=1)

        results = []
        for i, mse in enumerate(errors):
            mse = float(mse)
            results.append({
                "index":                i,
                "is_anomaly":           mse > THRESHOLD,
                "reconstruction_error": round(mse, 6),
                "confidence":           round(
                    min(mse / THRESHOLD, 1.0), 4),
                "severity": (
                    "critical" if mse > THRESHOLD * 2.0 else
                    "high"     if mse > THRESHOLD * 1.5 else
                    "medium"   if mse > THRESHOLD       else
                    "low"
                )
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------------------------------------------------
# SIMULAR ATAQUE — genera features del ataque y predice
# ----------------------------------------------------------
@app.post("/simulate/{attack_type}")
def simulate_attack(attack_type: str, fase: str = "reconocimiento"):
    try:
        # Features base normales
        features = {col: 0.0 for col in FEATURES}

        # Modifica según el tipo de ataque
        if attack_type == "ransomware" and fase == "cifrado":
            features['bytes_out']        = 150000
            features['duracion_conn']    = 40000
            features['cpu_uso']          = 95
            features['ram_uso']          = 90
            features['escrituras_disco'] = 30000
            features['ip_publica_flag']  = 1

        elif attack_type == "ransomware":
            features['num_conexiones']   = 150
            features['duracion_conn']    = 0.5
            features['bytes_out']        = 10
            features['puerto']           = 8443

        elif attack_type == "brute_force":
            features['intentos_login']      = 35
            features['accesos_ultimo_hora'] = 80

        elif attack_type == "phishing":
            features['hora']         = 3
            features['metodo_auth']  = 0
            features['bytes_out']    = 200

        elif attack_type == "exfiltracion":
            features['bytes_out']        = 80000
            features['escrituras_disco'] = 3000
            features['duracion_conn']    = 15000

        elif attack_type == "ddos":
            features['num_conexiones']  = 500
            features['bytes_in']        = 30000
            features['intentos_login']  = 800

        elif attack_type == "plc_injection":
            features['zona']      = 1
            features['protocolo'] = 2
            features['hora']      = 12

        elif attack_type == "vpn_unauthorized":
            features['ip_publica_flag'] = 1
            features['distancia']       = 8500

        # Predice
        df         = pd.DataFrame([features])
        df         = df[FEATURES]
        X_scaled   = scaler.transform(df)
        recon      = model.predict(X_scaled, verbose=0)
        mse        = float(np.mean(np.power(X_scaled - recon, 2)))

        return {
            "attack_type":          attack_type,
            "fase":                 fase,
            "is_anomaly":           mse > THRESHOLD,
            "reconstruction_error": round(mse, 6),
            "threshold":            round(THRESHOLD, 6),
            "confidence":           round(min(mse / THRESHOLD, 1.0), 4),
            "features_used":        features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------------------------------------------------
# INFO DEL MODELO
# ----------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Antamina ML Service está en línea", "docs": "/docs"}

@app.get("/model/info")
def model_info():
    return {
        "version":        metadata.get('model_version', 'unknown'),
        "threshold":      round(THRESHOLD, 6),
        "features_count": len(FEATURES),
        "features":       FEATURES,
        "training_date":  metadata.get('training_date', 'unknown'),
        "val_loss":       metadata.get('val_loss', 'unknown'),
        "epochs_trained": metadata.get('epochs_trained', 'unknown')
    }