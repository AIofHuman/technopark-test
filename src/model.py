import pandas as pd
import os
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import logging

from data_proccessing import DataProcessor

logger = logging.getLogger('technopark-test-task')

def mape_metric(y_true, y_pred):
    """Кастомная метрика MAPE"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class PricePredictor:
    def __init__(self):
        self.model = None
        self.data_processor = None
        self.cat_features = None
        self.id_feature = None
        
    def train(self, mlflow_manager, df, cat_features, id_feature = 'rfq_id', target='target_unit_price_rub', test_size=0.2, random_state=42):
        """Обучение модели"""
        # Подготовка данных
        self.data_processor = DataProcessor(cat_features, target, id_feature)
        df_clean = self.data_processor.validate_and_clean(df, 'cleaned_dataset.csv')
        
        # Разделение на признаки и целевую переменную
        X = df_clean.drop(columns=[target])
        y = df_clean[target]
        
        # self.feature_names = X.columns.tolist()
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Настройка MLflow
        # mlflow.set_experiment("price_prediction")
        
        # with mlflow.start_run():
            # Параметры модели
        model_params = {
            'iterations': 5000,
            'learning_rate': 0.01,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': random_state,
            'verbose': False
        }

        mlflow_manager.start_experiment()
            
        # Логирование параметров
        mlflow_manager.log_parameters({
            **model_params,
            'cat_features': self.data_processor.cat_features,
            'num_features': self.data_processor.num_features,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'target_column': target,
            'id_column': id_feature
        })
            
            # Инициализация модели
        self.model = CatBoostRegressor(**model_params)
            
        # Обучение с категориальными признаками
        logger.info(f"Catboost start with params {model_params}")
        self.model.fit(
            X_train, y_train,
            cat_features=self.data_processor.cat_features,
            eval_set=(X_test, y_test),
            verbose=500
        )
            
        # Считаем метрики
        y_pred = self.model.predict(X_test)
        mape = mape_metric(y_test.values, y_pred)
        rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
        metrics = {
            'mape': mape,
            'rmse': rmse
        }

        mlflow_manager.log_metrics(metrics)
            
        # Создание сигнатуры модели
        signature = infer_signature(X_train, self.model.predict(X_train))
            
        # Логирование модели
        mlflow_manager.log_model(
            model=self.model,
            signature=signature,
            input_example=X_train.iloc[:5]
        )
            
        mlflow_manager.end_run()
            
            # Логирование в MLflow
            # mlflow.log_params(model_params)
            # mlflow.log_metrics({
            #     'mape': mape,
            #     'rmse': rmse
            # })
            # mlflow.catboost.log_model(self.model, "model")
            
        logger.info(f"Model trained. MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")
            
        return self.model
    
    def predict(self, X):
        """Предсказание"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)