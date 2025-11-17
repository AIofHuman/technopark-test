from pydantic import BaseModel, field_validator

class PredictionRequest(BaseModel):
    """Схема для предсказания"""
    customer_tier: str
    material: str
    thickness_mm: float
    length_mm: float
    width_mm: float
    holes_count: int
    bends_count: int
    weld_length_mm: float
    cut_length_mm: float
    route: str
    tolerance: str
    surface_finish: str
    coating: str
    qty: int
    due_days: int
    engineer_score: float
    part_weight_kg: float

    @field_validator(
        'thickness_mm', 'length_mm', 'width_mm', 'weld_length_mm', 
        'cut_length_mm', 'part_weight_kg', 'engineer_score',
        'holes_count', 'bends_count', 'qty', 'due_days'
    )
    @classmethod
    def validate_non_negative(cls, v):
        """Общая проверка для всех числовых полей"""
        if v < 0:
            raise ValueError('Значение не может быть отрицательным')
        return v

class PredictionResponse(BaseModel):
    prediction: float
    status: str = "success"

    @field_validator('prediction')
    @classmethod
    def validate_price(cls, v):
        if v < 0:
            raise ValueError('Цена не может быть отрицательной')
        return v

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class ErrorResponse(BaseModel):
    detail: str