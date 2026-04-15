# FastAPI ML Microservice - Antamina

Microservicio FastAPI para detección de anomalías usando un modelo autoencoder entrenado en TensorFlow.

## Requisitos

- Python 3.11+
- Ver `requirements.txt` para dependencias

## Instalación Local

```bash
pip install -r requirements.txt
```

## Ejecutar Localmente

```bash
uvicorn main:app --reload --port 8000
```

La API estará disponible en: http://localhost:8000

## Endpoints

### 1. Health Check
```bash
GET /health
```
Retorna estado del servicio y metadatos del modelo.

### 2. Predicción Simple
```bash
POST /predict
Content-Type: application/json

{
  "feature1": 0.5,
  "feature2": 1.2,
  ...
}
```

### 3. Predicción en Lote
```bash
POST /predict/batch

[
  {"feature1": 0.5, "feature2": 1.2},
  {"feature1": 0.7, "feature2": 0.8}
]
```

### 4. Simular Ataque
```bash
POST /simulate/{attack_type}?fase={fase}

Tipos de ataque soportados:
- ransomware
- brute_force
- phishing
- exfiltracion
- ddos
- plc_injection
- vpn_unauthorized
```

### 5. Info del Modelo
```bash
GET /model/info
```

## Deployment en Render

1. Push del código a GitHub ✓
2. Conectar repositorio a Render
3. Crear Web Service
4. Environment: Python 3.11
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn main:app --host 0.0.0.0 --port 10000`

## Archivos Necesarios

- `main.py` - Código principal de la API
- `requirements.txt` - Dependencias de Python
- `autoencoder_antamina.keras` - Modelo entrenado
- `scaler_antamina.pkl` - Escalador de datos
- `model_metadata.json` - Metadatos del modelo

## Variables de Entorno

Actualmente ninguna requerida. El servicio carga el modelo al iniciar.

## CORS

El servicio tiene CORS habilitado para todas las origins, métodos y headers para facilitar integración con clientes.
