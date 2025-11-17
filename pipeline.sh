
set -e  # Прерывать выполнение при любой ошибке

echo "🚀 Запуск ML пайплайна..."
# Подготовка данных и обучение модели
python train.py --data_path "./data/mvp_quotes.csv" \
    --target "target_unit_price_rub" \
    --id_feature "rfq_id" \
    --model_name "price_predict"

echo "🧪 Запуск тестов..."
pytest -v

echo "🐳 Сборка Docker образа..."
docker build -t price-prediction-api .

echo "🔧 Запуск контейнера..."
docker run -p 8000:8000 price-prediction-api
echo "✅ Пайплайн успешно завершен!"
echo "📡 API доступно по адресу: http://localhost:8000"