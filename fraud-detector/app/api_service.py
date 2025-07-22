from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, validator
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

# Настройка логирования
logging.basicConfig(
    filename='logs/api_requests.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Настройка путей
sys.path.append(str(Path(__file__).parent.parent))

# Импорт функций
from feature_engineering import prepare_features, get_feature_names, map_location_to_country

app = FastAPI(
    title="AML Fraud Detection API",
    description="API для определения подозрительных транзакций",
    version="1.0.0"
)

# Загрузка модели
try:
    model_path = Path(__file__).parent.parent / "model" / "aml_model.joblib"
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    sender: str
    location: str
    recipient: str
    timestamp: str

    @validator('location')
    def validate_location(cls, v):
        mapped_loc = map_location_to_country(v)
        if not mapped_loc or len(mapped_loc) != 2 or not mapped_loc.isalpha():
            raise ValueError("Location must be valid 2-letter country code (e.g. 'US', 'RU')")
        return mapped_loc

def get_top_features(model, features: List[str], data: pd.DataFrame, top_n: int = 3) -> List[Dict]:
    """Возвращает топ-N наиболее значимых фичей для предсказания"""
    if not hasattr(model, 'feature_importances_'):
        return []
    
    feature_importance = []
    for feature in features:
        if feature in model.feature_names_:
            idx = list(model.feature_names_).index(feature)
            importance = model.feature_importances_[idx]
            feature_importance.append((feature, importance))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    return [
        {"feature": feat, "importance": float(imp)}
        for feat, imp in feature_importance[:top_n]
    ]

def simplify_features(features: List[str], data: pd.DataFrame) -> Dict[str, Any]:
    """Группирует фичи по категориям для читаемого вывода"""
    result = {
        "amount": {
            "value": float(data['amount'].iloc[0]),
            "features": [f for f in features if f in [
                'amount', 'log_amount', 'is_micro',
                'is_small', 'is_round'
            ]]
        },
        "time": {
            "hour": int(data['hour'].iloc[0]) if 'hour' in data.columns else None,
            "features": [f for f in features if f in [
                'is_night', 'is_weekend'
            ]]
        },
        "currency": {
            "type": data['currency'].iloc[0],
            "features": [f for f in features if f in [
                'is_rub', 'is_usd', 'is_eur'
            ]]
        },
        "risk": {
            "suspicious_recipient": bool(data['is_suspicious_recipient'].iloc[0]),
            "features": [f for f in features if f in [
                'is_suspicious_recipient', 'sender_count',
                'recipient_count', 'is_recurring', 'is_split'
            ]]
        },
        "location": {
            "code": data['location'].iloc[0] if 'location' in data.columns else None,
            "active_feature": next(
                (f for f in features if f.startswith('loc_') and data[f].iloc[0] == 1),
                None
            )
        }
    }
    
    return result

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования запросов"""
    response = await call_next(request)
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
    }
    
    if request.url.path == "/predict" and request.method == "POST":
        try:
            body = await request.json()
            log_data["transaction_id"] = body.get("transaction_id")
            log_data["amount"] = body.get("amount")
            log_data["location"] = body.get("location")
        except:
            pass
    
    logging.info(json.dumps(log_data))
    return response

@app.post("/predict", response_model=Dict[str, Any])
async def detect_fraud(transaction: Transaction, request: Request) -> Dict[str, Any]:
    try:
        # Преобразование и обработка данных
        data = pd.DataFrame([transaction.dict()])
        data = prepare_features(data)
        
        # Выбор фичей
        available_features = [f for f in get_feature_names() if f in data.columns]
        features = [f for f in available_features if f in model.feature_names_]
        
        if not features:
            raise ValueError("No matching features between model and input data")
        
        # Предсказание
        fraud_prob = float(model.predict_proba(data[features])[0][1])
        top_risk_factors = get_top_features(model, features, data)
        
        # Логирование предсказания
        prediction_log = {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": round(fraud_prob, 4),
            "is_fraud": fraud_prob > 0.5,
            "top_risk_factors": top_risk_factors,
            "timestamp": datetime.now().isoformat()
        }
        logging.info(f"PREDICTION: {json.dumps(prediction_log)}")
        
        return {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": round(fraud_prob, 4),
            "is_fraud": fraud_prob > 0.5,
            "top_risk_factors": top_risk_factors,
            "feature_analysis": simplify_features(features, data),
            "model_version": "1.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as ve:
        logging.error(f"ERROR: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "model_loaded": True,
        "service": "AML Fraud Detector",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/features")
async def get_model_features():
    return {
        "features": model.feature_names_,
        "count": len(model.feature_names_),
        "timestamp": datetime.now().isoformat()
    }