import os
import cv2
import time
import rawpy
import numpy as np
import logging
import argparse
import multiprocessing
import subprocess
import shlex
import traceback
import signal
import concurrent.futures
import sys
from pathlib import Path
from functools import partial
from tqdm import tqdm
import imageio
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from predict_quality import predict_quality, init_models
import joblib

# Инициализация базового логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Изменено с ERROR на INFO

# Настройка логирования
def setup_logging(log_level):
    """
    Настраивает логирование с указанным уровнем
    """
    print(f"Setting up logging with level: {log_level}")  # Отладочное сообщение
    
    # Настраиваем формат логов
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Настраиваем вывод в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Настраиваем вывод в файл
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(formatter)
    
    # Настраиваем логгер
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Удаляем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Добавляем наши обработчики
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    print("Logging setup complete")  # Отладочное сообщение
    return logger

# Глобальные переменные для управления процессами
interrupted = False
pool = None
current_process = None
model_queue = None
model_thread = None

# Инициализация CLIP
CLIP_AVAILABLE = False
clip_model = None
clip_processor = None

print("Starting model initialization...")  # Отладочное сообщение

try:
    print("Loading CLIP model...")  # Отладочное сообщение
    logger.info("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    logger.info("CLIP model loaded successfully")
    CLIP_AVAILABLE = True
    print("CLIP model loaded successfully")  # Отладочное сообщение
except Exception as e:
    logger.error(f"Failed to load CLIP model: {e}")
    print(f"Failed to load CLIP model: {e}")  # Отладочное сообщение
    CLIP_AVAILABLE = False

# Инициализация моделей
MODELS_AVAILABLE = {
    'efficientnet': False
}

models = {
    'efficientnet': None
}

print("Initializing TensorFlow models...")  # Отладочное сообщение

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.preprocessing.image import img_to_array
    
    # Отключаем предупреждения TensorFlow
    tf.get_logger().setLevel('ERROR')
    
    # Явно указываем использовать только CPU
    tf.config.set_visible_devices([], 'GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Инициализация модели с таймаутом
    def load_models():
        try:
            print("Loading EfficientNet model...")  # Отладочное сообщение
            models['efficientnet'] = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            MODELS_AVAILABLE['efficientnet'] = True
            logger.info("EfficientNet model loaded successfully (CPU mode)")
            print("EfficientNet model loaded successfully")  # Отладочное сообщение
            return True
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model: {e}")
            print(f"Failed to load EfficientNet model: {e}")  # Отладочное сообщение
            return False
    
    # Запускаем загрузку модели в отдельном потоке с таймаутом
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(load_models)
        try:
            success = future.result(timeout=30)  # Таймаут 30 секунд
            if not success:
                logger.error("Failed to load EfficientNet model")
                print("Failed to load EfficientNet model")  # Отладочное сообщение
        except concurrent.futures.TimeoutError:
            logger.error("Model loading timed out")
            print("Model loading timed out")  # Отладочное сообщение
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")  # Отладочное сообщение

except ImportError:
    logger.warning("TensorFlow not available. AI scoring will be disabled.")
    print("TensorFlow not available. AI scoring will be disabled.")  # Отладочное сообщение
except Exception as e:
    logger.error(f"Unexpected error during TensorFlow initialization: {e}")
    print(f"Unexpected error during TensorFlow initialization: {e}")  # Отладочное сообщение

print("Model initialization complete")  # Отладочное сообщение

def force_quit():
    """Принудительное завершение всех процессов"""
    global pool, current_process
    logger.warning("Force quitting all processes...")
    
    if pool:
        try:
            pool.terminate()
            pool.join(timeout=1)
        except:
            pass
    
    if current_process:
        try:
            current_process.terminate()
            current_process.join(timeout=1)
        except:
            pass
    
    sys.exit(1)

def signal_handler(signum, frame):
    """Обработчик сигналов прерывания"""
    global interrupted
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    interrupted = True
    force_quit()

# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, 'SIGQUIT'):
    signal.signal(signal.SIGQUIT, signal_handler)

def estimate_nima_score(image):
    if not MODELS_AVAILABLE['efficientnet'] or models['efficientnet'] is None:
        return None
    
    max_retries = 3  # Максимальное количество попыток
    retry_delay = 2  # Задержка между попытками в секундах
    
    for attempt in range(max_retries):
        try:
            # Подготовка изображения с таймаутом
            def prepare_image():
                try:
                    img = cv2.resize(image, (224, 224))
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = efficientnet_preprocess(img)
                    return img
                except Exception as e:
                    logger.error(f"Error preparing image: {e}")
                    return None
            
            # Запускаем подготовку изображения в отдельном потоке с таймаутом
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(prepare_image)
                try:
                    img = future.result(timeout=10)  # Увеличенный таймаут до 10 секунд
                    if img is None:
                        continue
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Image preparation timed out (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
            
            # Получение предсказания с таймаутом
            def predict():
                try:
                    return models['efficientnet'].predict(img, verbose=0)
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    return None
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(predict)
                try:
                    features = future.result(timeout=30)  # Увеличенный таймаут до 30 секунд
                    if features is None:
                        continue
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Model prediction timed out (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
            
            # Преобразование в оценку от 1 до 10
            score = float(np.mean(features)) * 10
            return min(max(score, 1), 10)
            
        except Exception as e:
            logger.error(f"Error in quality estimation (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None
    
    logger.error("All attempts to estimate NIMA score failed")
    return None

def estimate_focus_score(image_path):
    """Оценивает резкость изображения"""
    try:
        # Загружаем RAW файл
        with rawpy.imread(str(image_path)) as raw:
            # Преобразуем в RGB
            image = raw.postprocess()
            # Преобразуем в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            return float(lap.var())
    except Exception as e:
        logger.error(f"Error in estimate_focus_score: {e}")
        return 0.0

def estimate_contrast(image_path):
    """Оценивает контраст изображения"""
    try:
        # Загружаем RAW файл
        with rawpy.imread(str(image_path)) as raw:
            # Преобразуем в RGB
            image = raw.postprocess()
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Вычисляем стандартное отклонение яркости
            return float(np.std(gray))
    except Exception as e:
        logger.error(f"Error in estimate_contrast: {e}")
        return 0.0

def estimate_exposure(image_path):
    """Оценивает экспозицию изображения"""
    try:
        # Загружаем RAW файл
        with rawpy.imread(str(image_path)) as raw:
            # Преобразуем в RGB
            image = raw.postprocess()
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Вычисляем среднюю яркость
            mean_brightness = np.mean(gray)
            
            # Определяем статус экспозиции
            if mean_brightness < 50:
                return "Under"
            elif mean_brightness > 200:
                return "Over"
            else:
                return "OK"
    except Exception as e:
        logger.error(f"Error in estimate_exposure: {e}")
        return "Unknown"

def estimate_noise(image_path):
    """Оценивает уровень шума в изображении"""
    try:
        # Загружаем RAW файл
        with rawpy.imread(str(image_path)) as raw:
            # Преобразуем в RGB
            image = raw.postprocess()
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Вычисляем шум в тенях (область с низкой яркостью)
            shadow_mask = gray < 50
            if np.sum(shadow_mask) > 0:
                shadow_noise = np.std(gray[shadow_mask])
                if shadow_noise < 10:
                    return "Low"
                elif shadow_noise < 20:
                    return "Medium"
                else:
                    return "High"
            else:
                return "Unknown"
    except Exception as e:
        logger.error(f"Error in estimate_noise: {e}")
        return "Unknown"

def get_xmp_path(image_path):
    xmp_path = image_path.with_suffix('.xmp')
    # Если XMP файл не существует, создаем его
    if not xmp_path.exists():
        try:
            # Создаем пустой XMP файл
            xmp_path.touch()
            logger.info(f"Created new XMP file: {xmp_path}")
        except Exception as e:
            logger.error(f"Failed to create XMP file: {e}")
            raise
    return xmp_path

def write_xmp_tag(xmp_path, description_parts):
    description_text = " | ".join(description_parts)
    tag_arg = f"-XMP:Description={description_text}"
    
    # Проверяем права доступа к файлу
    if not os.access(xmp_path, os.W_OK):
        logger.error(f"No write access to XMP file: {xmp_path}")
        raise PermissionError(f"No write access to XMP file: {xmp_path}")
    
    # Извлекаем количество звёзд из description_parts
    stars = None
    for part in description_parts:
        if part.startswith("Stars: "):
            stars = int(part.split(": ")[1])
            break
    
    # Формируем команду с рейтингом, если он есть
    args = ['exiftool', '-overwrite_original']
    if stars is not None:
        args.extend(['-XMP:Rating=' + str(stars)])
    args.extend([tag_arg, str(xmp_path)])
    
    logger.info(f"Running exiftool with args: {' '.join(shlex.quote(a) for a in args)}")
    
    try:
        # Добавляем таймаут в 30 секунд и проверяем процесс
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Ждем завершения процесса с таймаутом
        try:
            stdout, stderr = process.communicate(timeout=30)
            if process.returncode != 0:
                logger.error(f"exiftool failed with return code {process.returncode}")
                logger.error(f"stderr: {stderr.strip()}")
                raise subprocess.CalledProcessError(process.returncode, args, stdout, stderr)
            logger.info(f"exiftool output: {stdout.strip()}")
            if stderr:
                logger.warning(f"exiftool warning: {stderr.strip()}")
        except subprocess.TimeoutExpired:
            logger.error(f"exiftool timed out after 30 seconds for file: {xmp_path}")
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
            # Пробуем принудительно завершить все процессы exiftool
            try:
                subprocess.run(['pkill', '-f', 'exiftool'], check=False)
            except:
                pass
            raise
    except Exception as e:
        logger.error(f"Error writing XMP: {e}")
        raise

def estimate_aesthetic_score(image_path):
    """Оценивает эстетическое качество изображения"""
    try:
        # Загружаем RAW файл
        with rawpy.imread(str(image_path)) as raw:
            # Преобразуем в RGB
            image = raw.postprocess()
            
            # Используем нашу обученную модель
            score = predict_quality(image)
            if score is not None:
                # Нормализуем скор в диапазон 0-10
                normalized_score = min(10, max(0, score * 2))
                return normalized_score
            
            # Если наша модель не сработала, используем базовые метрики
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Оцениваем контраст
            contrast = np.std(gray) / 128.0  # нормализуем к 1
            
            # Оцениваем экспозицию (насколько близко к среднему значению)
            mean = np.mean(gray)
            exposure_score = 1.0 - abs(mean - 128) / 128.0
            
            # Оцениваем резкость
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian) / 128.0
            
            # Комбинируем метрики
            score = (contrast + exposure_score + sharpness) / 3.0
            
            # Нормализуем к диапазону 0-10
            return score * 10
        
    except Exception as e:
        logger.error(f"Error in estimate_aesthetic_score: {e}")
        return 5.0  # возвращаем среднее значение в случае ошибки

def calculate_star_rating(focus_score, contrast, exposure, noise, aesthetic_score=None):
    """
    Переводит технические оценки в систему звёзд (1-5)
    """
    # Нормализуем focus_score (типичный диапазон 100-500)
    # Считаем, что резкость от 200 уже хорошая
    normalized_focus = max(0, min(1, (focus_score - 200) / 300))  # 0-1
    
    # Нормализуем contrast (0-100)
    # Считаем, что контраст от 30 уже нормальный
    normalized_contrast = max(0, min(1, (contrast - 30) / 70))
    
    # Нормализуем exposure (Under/OK/Over -> 0.0/0.5/1.0)
    exposure_values = {
        "Under": 0.3,    # Слишком темное, но не критично
        "OK": 0.8,       # Нормальная
        "Over": 0.4      # Слишком светлое, но не критично
    }
    normalized_exposure = exposure_values.get(exposure, 0.5)
    
    # Нормализуем noise (Low/Medium/High -> 1.0/0.5/0.0)
    noise_values = {
        "Low": 1.0,      # Хорошо
        "Medium": 0.7,   # Средне, но приемлемо
        "High": 0.3,     # Плохо, но не критично
        "Unknown": 0.5   # По умолчанию
    }
    normalized_noise = noise_values.get(noise, 0.5)
    
    # Если есть оценка эстетики, используем её
    if aesthetic_score is not None:
        # Считаем, что оценка от 5 уже хорошая
        normalized_aesthetic = max(0, min(1, (aesthetic_score - 5) / 5))
    else:
        normalized_aesthetic = 0.5  # Нейтральная оценка
    
    # Веса для разных компонентов
    weights = {
        'focus': 0.25,     # Резкость важна, но не критична
        'contrast': 0.15,  # Контраст добавляет выразительности
        'exposure': 0.15,  # Экспозиция важна для качества
        'noise': 0.15,     # Шум влияет на качество
        'aesthetic': 0.3   # Эстетика важна
    }
    
    # Вычисляем взвешенную сумму
    weighted_sum = (
        normalized_focus * weights['focus'] +
        normalized_contrast * weights['contrast'] +
        normalized_exposure * weights['exposure'] +
        normalized_noise * weights['noise'] +
        normalized_aesthetic * weights['aesthetic']
    )
    
    # Переводим в звёзды (1-5)
    # Смещаем диапазон, чтобы средние оценки давали 3 звезды
    stars = 1 + (weighted_sum * 4)
    
    # Округляем до ближайшего целого
    return round(stars)

def estimate_ai_scores(image_path):
    """
    Оценивает качество изображения с помощью EfficientNet
    """
    if not MODELS_AVAILABLE['efficientnet'] or models['efficientnet'] is None:
        return None
    
    try:
        # Загружаем RAW файл
        with rawpy.imread(str(image_path)) as raw:
            # Преобразуем в RGB
            image = raw.postprocess()
            
            # Подготовка изображения
            img = cv2.resize(image, (224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = efficientnet_preprocess(img)
            
            # Получаем предсказание
            features = models['efficientnet'].predict(img, verbose=0)
            
            # Преобразуем в оценку от 1 до 10
            score = float(np.mean(features)) * 10
            return {'efficientnet': min(max(score, 1), 10)}
        
    except Exception as e:
        logger.error(f"Error in AI scoring: {e}")
        return None

# Путь к модели
MODEL_PATH = Path("models/photo_quality_model.joblib")

# Глобальные переменные для модели
quality_model = None

# Загрузка модели
def load_quality_model():
    global quality_model
    try:
        logger.info(f"Пытаемся загрузить модель из: {MODEL_PATH.absolute()}")
        if MODEL_PATH.exists():
            logger.info("Файл модели найден, начинаем загрузку...")
            quality_model = joblib.load(MODEL_PATH)
            logger.info("Модель качества фотографий успешно загружена")
            return True
        else:
            logger.error(f"Файл модели не найден: {MODEL_PATH.absolute()}")
            return False
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}\n{traceback.format_exc()}")
        return False

# Загружаем модель при запуске
logger.info("Начинаем загрузку модели качества...")
load_quality_model()

def estimate_quality_score(image_path):
    """
    Оценивает качество изображения с помощью обученной модели
    """
    try:
        # Загружаем RAW файл
        with rawpy.imread(str(image_path)) as raw:
            # Преобразуем в RGB
            image = raw.postprocess()
            # Используем функцию predict_quality из модуля predict_quality
            score = predict_quality(image)
            if score is not None:
                # Нормализуем оценку от 1 до 10
                normalized_score = min(10, max(1, score * 10))
                return normalized_score
            return None
    except Exception as e:
        logger.error(f"Ошибка при оценке качества: {e}\n{traceback.format_exc()}")
        return None

def extract_features(image_path):
    """
    Извлекает признаки из изображения для модели используя CLIP и EfficientNet
    """
    try:
        # Загружаем RAW файл
        with rawpy.imread(str(image_path)) as raw:
            # Преобразуем в RGB
            image = raw.postprocess()
            
            # CLIP features
            inputs = clip_processor(images=image, return_tensors="pt")
            image_features = clip_model.get_image_features(**inputs)
            clip_features = image_features.detach().numpy()[0]
            
            # EfficientNet features
            img = cv2.resize(image, (224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = efficientnet_preprocess(img)
            efficientnet_features = models['efficientnet'].predict(img, verbose=0)[0]
            
            # Базовые характеристики изображения
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Контраст
            contrast = np.std(gray)
            
            # Экспозиция
            exposure = np.mean(gray)
            
            # Шум
            noise = estimate_noise(image_path)
            
            # Объединяем все признаки
            features = np.concatenate([
                clip_features,
                efficientnet_features,
                [contrast, exposure, noise]
            ])
            
            return features
        
    except Exception as e:
        logger.error(f"Ошибка при извлечении признаков: {e}\n{traceback.format_exc()}")
        return None

def process_arw_file(arw_path, output_dir, enable_ai_scoring=True, enable_aesthetic_scoring=True):
    """
    Обрабатывает один ARW файл
    """
    try:
        print(f"\nНачинаем обработку файла: {arw_path}")  # Отладочное сообщение
        
        # Проверяем существование файла
        if not os.path.exists(arw_path):
            print(f"Файл не существует: {arw_path}")  # Отладочное сообщение
            return
        
        # Создаем выходной каталог, если его нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Получаем имя файла без расширения
        base_name = os.path.splitext(os.path.basename(arw_path))[0]
        
        # Путь для XMP файла
        xmp_path = os.path.join(output_dir, f"{base_name}.xmp")
        
        print(f"Создаем XMP файл: {xmp_path}")  # Отладочное сообщение
        
        # Создаем XMP файл
        with open(xmp_path, 'w') as f:
            f.write('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n')
            f.write('<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 6.0">\n')
            f.write('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n')
            f.write('<rdf:Description rdf:about=""\n')
            f.write('  xmlns:tiff="http://ns.adobe.com/tiff/1.0/"\n')
            f.write('  xmlns:exif="http://ns.adobe.com/exif/1.0/"\n')
            f.write('  xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n')
            f.write('  xmlns:dc="http://purl.org/dc/elements/1.1/"\n')
            f.write('  xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/">\n')
            
            # Добавляем базовые метаданные
            f.write('  <tiff:ImageDescription>ARW Quality Tagger</tiff:ImageDescription>\n')
            f.write('  <xmp:ModifyDate>' + time.strftime('%Y-%m-%dT%H:%M:%S') + '</xmp:ModifyDate>\n')
            
            # Добавляем технические метрики
            print("Извлекаем технические метрики...")  # Отладочное сообщение
            focus_score = estimate_focus_score(arw_path)
            contrast = estimate_contrast(arw_path)
            exposure = estimate_exposure(arw_path)
            noise = estimate_noise(arw_path)
            
            print(f"Технические метрики: Focus={focus_score:.2f}, Contrast={contrast:.2f}, Exposure={exposure}, Noise={noise}")  # Отладочное сообщение
            
            # Добавляем AI оценки
            if enable_ai_scoring:
                print("Оцениваем качество с помощью AI...")  # Отладочное сообщение
                quality_score = estimate_quality_score(arw_path)
                print(f"AI оценка качества: {quality_score:.2f}")  # Отладочное сообщение
            else:
                quality_score = None
                print("AI оценка качества отключена")  # Отладочное сообщение
            
            if enable_aesthetic_scoring and CLIP_AVAILABLE:
                print("Оцениваем эстетику с помощью CLIP...")  # Отладочное сообщение
                aesthetic_score = estimate_aesthetic_score(arw_path)
                print(f"CLIP оценка эстетики: {aesthetic_score:.2f}")  # Отладочное сообщение
            else:
                aesthetic_score = None
                print("CLIP оценка эстетики отключена")  # Отладочное сообщение
            
            # Вычисляем общий рейтинг
            print("Вычисляем общий рейтинг...")  # Отладочное сообщение
            stars = calculate_star_rating(focus_score, contrast, exposure, noise, aesthetic_score)
            print(f"Общий рейтинг: {stars} звезд")  # Отладочное сообщение
            
            # Добавляем все метрики в XMP
            f.write('  <photoshop:History>')  # Используем History для хранения наших метрик
            metrics = []
            if focus_score is not None:
                metrics.append(f"FocusScore: {focus_score:.2f}")
            if contrast is not None:
                metrics.append(f"Contrast: {contrast:.2f}")
            if exposure is not None:
                metrics.append(f"Exposure: {exposure}")
            if noise is not None:
                metrics.append(f"Noise: {noise}")
            if quality_score is not None:
                metrics.append(f"Quality: {quality_score:.2f}")
            if aesthetic_score is not None:
                metrics.append(f"Aesthetic: {aesthetic_score:.2f}")
            if stars is not None:
                metrics.append(f"Stars: {stars}")
            
            f.write(" | ".join(metrics))
            f.write('</photoshop:History>\n')
            
            # Закрываем теги
            f.write('  </rdf:Description>\n')
            f.write('</rdf:RDF>\n')
            f.write('</x:xmpmeta>\n')
            f.write('<?xpacket end="w"?>\n')
        
        print(f"XMP файл успешно создан: {xmp_path}")  # Отладочное сообщение
        
    except Exception as e:
        print(f"Ошибка при обработке файла {arw_path}: {str(e)}")  # Отладочное сообщение
        print(f"Traceback: {traceback.format_exc()}")  # Отладочное сообщение
        logger.error(f"Error processing {arw_path}: {str(e)}")
        logger.error(traceback.format_exc())

def main(folder_path, max_processes=None, use_nima=False, log_level=logging.ERROR):
    """Основная функция обработки"""
    global pool, interrupted, model_queue, model_thread
    
    # Инициализируем модели
    init_models()
    
    # Настраиваем логирование
    logger = setup_logging(log_level)
    
    try:
        arw_files = list(Path(folder_path).rglob("*.ARW"))
        if not arw_files:
            logger.warning("No .ARW files found in the specified folder.")
            return
        
        # Определяем количество процессов
        if max_processes is None:
            num_processes = min(multiprocessing.cpu_count(), 8)
        else:
            num_processes = min(multiprocessing.cpu_count(), max_processes)
        
        if log_level <= logging.INFO:
            logger.info(f"Using {num_processes} processes for parallel processing")
            logger.info(f"Found {len(arw_files)} files to process")
            if use_nima and MODELS_AVAILABLE['efficientnet']:
                logger.info("AI scoring enabled")
            else:
                logger.info("AI scoring disabled")
            if CLIP_AVAILABLE:
                logger.info("Aesthetic scoring enabled")
            else:
                logger.info("Aesthetic scoring disabled")
        
        # Создаем пул процессов
        pool = multiprocessing.Pool(processes=num_processes)
        
        try:
            # Запускаем обработку файлов параллельно с прогресс-баром
            results = []
            with tqdm(total=len(arw_files), desc="Processing files", unit="file") as pbar:
                for result in pool.imap_unordered(
                    partial(process_arw_file, output_dir=folder_path, enable_ai_scoring=use_nima, enable_aesthetic_scoring=CLIP_AVAILABLE),
                    arw_files
                ):
                    if interrupted:
                        logger.info("Interrupting processing...")
                        break
                    results.append(result)
                    pbar.update(1)
            
            # Анализируем результаты
            successful = sum(1 for r in results if r is not None)
            failed = len(results) - successful
            
            if log_level <= logging.INFO:
                logger.info(f"Processing completed. Success: {successful}, Failed: {failed}")
            
            if failed > 0:
                logger.warning("Some files failed to process")
        finally:
            # Гарантированное закрытие пула
            try:
                pool.terminate()
                pool.join(timeout=1)
            except:
                pass
                    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
        interrupted = True
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
    finally:
        if interrupted:
            force_quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ARW files and analyze image quality')
    parser.add_argument('folder', help='Path to folder containing ARW files')
    parser.add_argument('--processes', type=int, help='Maximum number of processes to use')
    parser.add_argument('--use-nima', action='store_true', help='Enable AI scoring')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='ERROR', help='Set the logging level')
    args = parser.parse_args()
    
    # Преобразуем строку уровня логирования в константу
    log_level = getattr(logging, args.log_level.upper())
    
    try:
        main(args.folder, args.processes, args.use_nima, log_level)
    except KeyboardInterrupt:
        logger.info("Main process interrupted. Shutting down...")
        force_quit()
