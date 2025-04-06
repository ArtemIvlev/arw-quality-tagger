import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import logging
import concurrent.futures
import traceback
import time

# Настройка логгера
logger = logging.getLogger(__name__)

# Глобальные переменные для моделей
models = {
    'efficientnet': None,
    'quality': None
}

# Функция для загрузки моделей
def init_models():
    """Инициализирует все необходимые модели"""
    global models
    
    # Загружаем EfficientNet
    try:
        models['efficientnet'] = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        logger.info("EfficientNet model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load EfficientNet model: {e}")
        models['efficientnet'] = None
    
    # Загружаем модель качества
    try:
        model_path = os.path.join('models', 'photo_quality_model.joblib')
        if os.path.exists(model_path):
            models['quality'] = joblib.load(model_path)
            logger.info("Quality model loaded successfully")
        else:
            logger.warning(f"Quality model file not found: {model_path}")
            models['quality'] = None
    except Exception as e:
        logger.error(f"Failed to load quality model: {e}")
        models['quality'] = None

def predict_quality(image):
    """
    Предсказывает качество изображения используя EfficientNet и модель качества
    """
    if models['efficientnet'] is None or models['quality'] is None:
        logger.warning("Models not initialized")
        return None
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Подготовка изображения
            img = cv2.resize(image, (224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            
            # Получаем признаки от EfficientNet
            features = models['efficientnet'].predict(img, verbose=0)
            
            # Предсказываем качество
            score = models['quality'].predict(features)[0]
            
            # Нормализуем оценку от 0 до 1
            normalized_score = min(max(score, 0), 1)
            return normalized_score
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании качества (попытка {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None
    
    return None

if __name__ == "__main__":
    # Тест на одном изображении
    init_models()
    test_image = Image.open("test.jpg")
    score = predict_quality(test_image)
    print(f"Predicted quality score: {score:.3f}") 