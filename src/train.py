import pandas as pd
import argparse
import os
import logging
from model import PricePredictor
from mlflow_manage import MLflowManager

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

# Настройка логирования
logger = logging.getLogger('technopark-test-task')
logger.setLevel(logging.DEBUG)

# Проверка на дублирование обработчиков
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)

def main(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, default="target_unit_price_rub", help="Target column name") 
    parser.add_argument("--id_feature", type=str, default="rfq_id", help="ID column name")
    parser.add_argument("--model_name", type=str, default="price_predict", help="Model name")
    args = parser.parse_args(args_list)
    
    try:
        # Проверка существования файла
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Файл не найден: {args.data_path}")
            
        logger.info(f"Загрузка данных из {args.data_path}")
        df = pd.read_csv(args.data_path)
        logger.info(f"Данные загружены: {df.shape}")
        
        logger.info("Начало обучения...")
        mlflow_manager = MLflowManager(experiment_name="technopark-test-task", model_name=args.model_name)
        predictor = PricePredictor()
        cat_features = ['customer_tier', 'material', 'route', 'tolerance', 'surface_finish', 'coating']
        model = predictor.train(mlflow_manager, df, cat_features, id_feature=args.id_feature, target=args.target)
        
        # Создание директории для моделей
        models_dir = os.path.join(PROJECT_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"{args.model_name}.cbm")
        model.save_model(model_path)
        logger.info(f"Модель сохранена в {model_path}")
        
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")
        raise

if __name__ == "__main__":
    try:
        data_path = os.path.join(PROJECT_DIR, "data", "mvp_quotes.csv")
        main(["--data_path", data_path, 
              "--target", "target_unit_price_rub", 
              "--id_feature", "rfq_id", 
              "--model_name", "price_predict"])
    except Exception as e:
        logger.error(f"Ошибка при выполнении: {e}")
