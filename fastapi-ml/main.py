# ============================================================
# main.py — FastAPI Microservicio ML
# ============================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
from typing import Dict, Any

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
print("Cargando modelo...")
model = keras.models.load_model('autoencoder_antamina.keras')
scaler   = joblib.load('scaler_antamina.pkl')

with open('model_metadata.json') as f:
    metadata = json.load(f)

THRESHOLD = metadata['anomaly_threshold']
FEATURES  = metadata['features_used']
print(f"✓ Modelo cargado — threshold: {THRESHOLD:.6f}")

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
def predict_batch(data: list):
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
@app.get("/model/info")
def model_info():
    return {
        "version":        metadata['model_version'],
        "threshold":      THRESHOLD,
        "features_count": len(FEATURES),
        "features":       FEATURES,
        "training_date":  metadata['training_date'],
        "val_loss":       metadata['val_loss'],
        "epochs_trained": metadata['epochs_trained']
    }