import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

REGRESSION_TEST_DATA = [
    {
        "customer_tier": "A",
        "material": "steel",
        "thickness_mm": 2.5,
        "length_mm": 1000.0,
        "width_mm": 500.0,
        "holes_count": 10,
        "bends_count": 5,
        "weld_length_mm": 200.0,
        "cut_length_mm": 1500.0,
        "route": "laser_cut",
        "tolerance": "standard",
        "surface_finish": "paint",
        "coating": "powder",
        "qty": 100,
        "due_days": 14,
        "engineer_score": 7.5,
        "part_weight_kg": 15.5
    },
    {
        "customer_tier": "B",
        "material": "aluminum",
        "thickness_mm": 1.5,
        "length_mm": 800.0,
        "width_mm": 400.0,
        "holes_count": 5,
        "bends_count": 2,
        "weld_length_mm": 100.0,
        "cut_length_mm": 1200.0,
        "route": "waterjet",
        "tolerance": "precise",
        "surface_finish": "raw",
        "coating": "none",
        "qty": 50,
        "due_days": 7,
        "engineer_score": 6.0,
        "part_weight_kg": 8.2
    }
]

REFERENCE_PREDICTIONS = [157, 153]

def test_no_regression():
    """
    Тест на отсутствие регрессии.
    Проверяет, что предсказания модели не изменились значимо
    по сравнению с предыдущими запусками.
    """
    # Если модель не загружена, пропускаем тест
    health_response = client.get("/health")
    if not health_response.json().get("model_loaded", False):
        pytest.skip("Модель не загружена, пропускаем тест на регрессию")
    
    current_predictions = []
    
    # Получаем текущие предсказания для тестовых данных
    for test_data in REGRESSION_TEST_DATA:
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200, f"Ошибка при предсказании: {response.text}"
        
        prediction_data = response.json()
        current_predictions.append(prediction_data["prediction"])
    
        
    # Проверяем, что предсказания не изменились (допустимое отклонение 5%)
    tolerance = 0.05  # 1%
    
    for i, (current, reference) in enumerate(zip(current_predictions, REFERENCE_PREDICTIONS)):
        deviation = abs(current - reference) / abs(reference)
        
        assert deviation <= tolerance, (
            f"Регрессия обнаружена для примера {i+1}: "
            f"отклонение {deviation:.4f} ({deviation*100:.2f}%) "
            f"превышает допустимое {tolerance*100:.1f}%. "
            f"Было: {reference:.2f}, стало: {current:.2f}"
        )
    
    print(f"Регрессия не обнаружена. Предсказания: {current_predictions}")

def test_health_check():
    """Тест health check эндпоинта"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_input_output():
    """Тест корректности формата входа/выхода"""
    # Тестовые данные с актуальными полями модели
    test_data = {
        "customer_tier": "A",
        "material": "steel",
        "thickness_mm": 2.5,
        "length_mm": 1000.0,
        "width_mm": 500.0,
        "holes_count": 10,
        "bends_count": 5,
        "weld_length_mm": 200.0,
        "cut_length_mm": 500.0,
        "route": "laser_cut",
        "tolerance": "standard",
        "surface_finish": "paint",
        "coating": "zinc",
        "qty": 100,
        "due_days": 14,
        "engineer_score": 7.5,
        "part_weight_kg": 15.5
    }
    
    response = client.post("/predict", json=test_data)
    
    # Проверка формата ответа
    assert response.status_code in [200, 503]  # 503 если модель не загружена
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "status" in data
        assert isinstance(data["prediction"], float)
        assert data["prediction"] >= 0  # Цена не может быть отрицательной

def test_inference():
    """Тест инференса с реальными данными"""
    test_data = {
        "customer_tier": "B",
        "material": "aluminum",
        "thickness_mm": 1.5,
        "length_mm": 800.0,
        "width_mm": 400.0,
        "holes_count": 5,
        "bends_count": 2,
        "weld_length_mm": 100.0,
        "cut_length_mm": 200.0,
        "route": "laser_cut",
        "tolerance": "precise",
        "surface_finish": "none",
        "coating": "none",
        "qty": 50,
        "due_days": 7,
        "engineer_score": 6.0,
        "part_weight_kg": 8.2
    }
    
    response = client.post("/predict", json=test_data)
    
    # Проверяем различные возможные статусы ответа
    assert response.status_code in [200, 503, 422]
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "prediction" in data

def test_validation_errors():
    """Тест валидации входных данных"""
    # Тест с отрицательным значением
    invalid_data = {
        "customer_tier": "C",
        "material": "steel",
        "thickness_mm": -2.5,  # Отрицательное значение!
        "length_mm": 1000.0,
        "width_mm": 500.0,
        "holes_count": 10,
        "bends_count": 5,
        "weld_length_mm": 200.0,
        "cut_length_mm": 1500.0,
        "route": "laser_cut",
        "tolerance": "standard",
        "surface_finish": "paint",
        "coating": "powder",
        "qty": 100,
        "due_days": 14,
        "engineer_score": 7.5,
        "part_weight_kg": 15.5
    }
    
    response = client.post("/predict", json=invalid_data)
    # Должна быть ошибка валидации
    assert response.status_code == 422
    
    # Проверяем что в ответе есть информация об ошибке
    error_data = response.json()
    assert "detail" in error_data

def test_missing_fields():
    """Тест отсутствия обязательных полей"""
    # Неполные данные (отсутствует material)
    incomplete_data = {
        "customer_tier": "A",
        "thickness_mm": 2.5,
        "length_mm": 1000.0,
        "width_mm": 500.0,
        "holes_count": 10,
        # пропущен material
        "bends_count": 5,
        "weld_length_mm": 200.0,
        "cut_length_mm": 1500.0,
        "route": "laser_cut",
        "tolerance": "standard",
        "surface_finish": "paint",
        "coating": "powder",
        "qty": 100,
        "due_days": 14,
        "engineer_score": 7.5,
        "part_weight_kg": 15.5
    }
    
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422  # Ошибка валидации


# if __name__ == "__main__":
#     # import uvicorn
#     # uvicorn.run(app, host="0.0.0.0", port=8000)
#     test_no_regression()