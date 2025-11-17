import os
import logging
from dotenv import load_dotenv
import mlflow
import mlflow.catboost
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd

# Загрузка переменных окружения
load_dotenv()

logger = logging.getLogger('technopark-test-task')

class MLflowManager:
    def __init__(self, experiment_name=None, model_name=None):
        
        self.tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.s3_bucket = os.getenv('S3_BUCKET')
        
        if not all([self.model_name, self.experiment_name]):
            raise ValueError("Model name and experiment name must be provided")
        
        # Настройка MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        # Настройка S3 credentials
        self._setup_s3_credentials()
        
        logger.info(f"MLflow Manager initialized: {self.tracking_uri}")
    
    def _setup_s3_credentials(self):
        """Настройка учетных данных для S3"""
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        s3_endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL')
        
        if aws_access_key and aws_secret_key:
            os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
            
        if s3_endpoint:
            os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint
    
    def start_experiment(self, run_name=None):
        """Запуск эксперимента MLflow"""
        return mlflow.start_run(run_name=run_name)
    
    def log_parameters(self, params):
        """Логирование параметров"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        """Логирование метрик"""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, signature=None, input_example=None, model_name=None):
        """Логирование модели в MLflow Model Registry"""
        model_name = model_name or self.model_name
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example
        )
        logger.info(f"Model registered as: {model_name}")
    
    def log_artifact(self, file_path):
        """Логирование артефактов"""
        mlflow.log_artifact(file_path)
    
    def end_run(self):
        """Завершение run"""
        mlflow.end_run()
    
    def load_model(self, model_name=None, alias="production"):
        """Загрузка модели из MLflow Model Registry"""
        model_name = model_name or self.model_name
        
        try:
            model_uri = f"models:/{model_name}@{alias}"
            logger.info(f"Loading model from: {model_uri}")
            
            model = mlflow.catboost.load_model(model_uri)
            logger.info(f"Successfully loaded model: {model_name} (stage: {alias})")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from MLflow: {str(e)}")
            raise
    