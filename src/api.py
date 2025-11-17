from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Generator, List
import pandas as pd
import numpy as np
import logging
from mlflow_manage import MLflowManager
from schemas import PredictionRequest, PredictionResponse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Price Prediction API", version="1.0.0")

# Глобальные переменные для модели
MODEL_NAME = "price_predict"
MODEL_ALIAS = "production"
EXPERIMENT_NAME = "technopark-test-task"

class ModelLoader:
    def __init__(self):
        self.mlflow_manager = None
        self.model = None

        self.load_production_model()
    
    def load_production_model(self):
        """Загрузка production модели из MLflow"""
        try:
            self.mlflow_manager = MLflowManager(experiment_name=EXPERIMENT_NAME, model_name=MODEL_NAME)
            # Загрузка модели
            self.model = self.mlflow_manager.load_model(model_name=MODEL_NAME, alias=MODEL_ALIAS)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

model_loader = ModelLoader()

async def lifespan(app: FastAPI):
    try:
        logger.info("Starting up Price Prediction API...")
        
        if model_loader.model is None:
            logger.error("Failed to load model on startup. Service will not be able to serve predictions.")
        
        logger.info("API startup completed")
        yield
    finally:
        logger.info("Shutting down Price Prediction API...")
        
        logger.info("API shutdown completed")

app.lifespan = lifespan

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Эндпоинт для предсказания цены"""
    try:
        if model_loader.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        input_df = pd.DataFrame([request.dict()])
        prediction = model_loader.model.predict(input_df)
        
        return PredictionResponse(
            prediction=float(prediction[0]),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check эндпоинт"""

    model_status = "loaded" if model_loader.model is not None else "not_loaded"
    
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)