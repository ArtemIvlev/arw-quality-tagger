import rawpy
import imageio
import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys
import shlex
import multiprocessing
from functools import partial
from tqdm import tqdm
import traceback
import logging
import signal
import time
import argparse
import os
import threading
from queue import Queue
import concurrent.futures
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

try:
    logger.info("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    logger.info("CLIP model loaded successfully")
    CLIP_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to load CLIP model: {e}")
    CLIP_AVAILABLE = False

def force_quit():
    """Принудительное завершение всех процессов"""
    global pool, current_process, model_queue, model_thread
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
    
    if model_queue:
        try:
            model_queue.put(None)  # Сигнал для завершения потока модели
        except:
            pass
    
    if model_thread and model_thread.is_alive():
        try:
            model_thread.join(timeout=1)
        except:
            pass
    
    # Принудительное завершение всех дочерних процессов
    try:
        if os.name == 'posix':  # Linux/Unix
            os.system('pkill -f "python.*main.py"')
        elif os.name == 'nt':   # Windows
            os.system('taskkill /F /IM python.exe')
    except:
        pass
    
    sys.exit(1)

def signal_handler(signum, frame):
    """Обработчик сигналов прерывания"""
    global interrupted
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    interrupted = True
    
    # Если сигнал получен повторно, принудительно завершаем
    if signal.getsignal(signum) == signal_handler:
        logger.warning("Received second interrupt signal. Force quitting...")
        force_quit()

# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGQUIT, signal_handler)

# Инициализация TensorFlow (если доступен)
TENSORFLOW_AVAILABLE = False
nima_model = None

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    
    # Отключаем предупреждения TensorFlow
    tf.get_logger().setLevel('ERROR')
    
    # Явно указываем использовать только CPU
    tf.config.set_visible_devices([], 'GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Инициализация модели EfficientNet с таймаутом
    def load_model():
        try:
            model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            logger.info("EfficientNet model loaded successfully (CPU mode)")
            return model
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model: {e}")
            return None
    
    # Запускаем загрузку модели в отдельном потоке с таймаутом
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(load_model)
        try:
            nima_model = future.result(timeout=30)  # Таймаут 30 секунд на загрузку модели
            if nima_model is not None:
                TENSORFLOW_AVAILABLE = True
        except concurrent.futures.TimeoutError:
            logger.error("Model loading timed out")
            nima_model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            nima_model = None
except ImportError:
    logger.warning("TensorFlow not available. NIMA scoring will be disabled.")
except Exception as e:
    logger.error(f"Unexpected error during TensorFlow initialization: {e}")

def estimate_nima_score(image):
    if not TENSORFLOW_AVAILABLE or nima_model is None:
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
                    img = preprocess_input(img)
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
                    return nima_model.predict(img, verbose=0)
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

def estimate_focus_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def estimate_contrast(image):
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Вычисляем стандартное отклонение яркости
    return float(np.std(gray))

def estimate_exposure(image):
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

def estimate_noise(image):
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
    
    # Теперь записываем новый тег
    args = ['exiftool', '-overwrite_original', tag_arg, str(xmp_path)]
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

def estimate_aesthetic_score(image):
    if not CLIP_AVAILABLE or clip_model is None or clip_processor is None:
        return None
    
    try:
        # Проверяем, не пустое ли изображение
        if np.all(image == 0) or np.all(image == 255):
            logger.warning("Empty or solid color image detected")
            return 1.0
            
        # Проверяем минимальную яркость
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if np.mean(gray) < 10:  # Слишком темное изображение
            logger.warning("Image is too dark")
            return 1.0
            
        # Конвертируем numpy array в PIL Image
        pil_image = Image.fromarray(image)
        
        # Набор промптов для оценки
        prompts = [
            "This is a high-quality artistic nude photo.",
            "This is a low-quality or poorly composed photo.",
            "This is a professional photography with good composition.",
            "This is an amateur or snapshot photo."
        ]
        
        # Получаем оценки для каждого промпта
        scores = []
        for text in prompts:
            inputs = clip_processor(
                images=pil_image,
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            scores.append(float(logits_per_image.item()))
        
        # Вычисляем взвешенную оценку
        weights = [1.0, -1.0, 0.8, -0.8]  # Веса для каждого промпта
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        # Нормализуем оценку
        normalized_score = (weighted_score + 4) / 8  # Приводим к диапазону [0, 1]
        
        # Преобразуем в оценку от 1 до 10
        final_score = normalized_score * 9 + 1  # Масштабируем в диапазон [1, 10]
        
        # Ограничиваем оценку
        return min(max(final_score, 1), 10)
        
    except Exception as e:
        logger.error(f"Error in aesthetic estimation: {e}")
        return None

def process_arw_file(arw_path, use_nima=False):
    global current_process
    current_process = multiprocessing.current_process()
    
    if interrupted:
        return False, "Processing interrupted"
        
    try:
        logger.info(f"Processing: {arw_path.name}")
        with rawpy.imread(str(arw_path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
        
        # Вычисляем все метрики
        focus_score = estimate_focus_score(rgb)
        contrast = estimate_contrast(rgb)
        exposure = estimate_exposure(rgb)
        noise = estimate_noise(rgb)
        
        # Выводим результаты
        logger.info(f"Focus score: {focus_score:.2f}")
        logger.info(f"Contrast: {contrast:.1f}")
        logger.info(f"Exposure: {exposure}")
        logger.info(f"Noise: {noise}")
        
        # Записываем в XMP
        xmp_path = get_xmp_path(arw_path)
        logger.info(f"Preparing to write XMP to: {xmp_path}")
        
        description_parts = [
            f"FocusScore: {focus_score:.2f}",
            f"Contrast: {contrast:.1f}",
            f"Exposure: {exposure}",
            f"Noise: {noise}"
        ]
        
        # Добавляем NIMA оценку только если включена и доступна
        if use_nima and TENSORFLOW_AVAILABLE:
            try:
                nima_score = estimate_nima_score(rgb)
                if nima_score is not None:
                    logger.info(f"NIMA score: {nima_score:.1f}")
                    description_parts.append(f"NIMA: {nima_score:.1f}")
            except Exception as e:
                logger.error(f"Error during NIMA scoring: {e}")
        
        # Добавляем оценку эстетики через CLIP
        if CLIP_AVAILABLE:
            try:
                aesthetic_score = estimate_aesthetic_score(rgb)
                if aesthetic_score is not None:
                    logger.info(f"Aesthetic score: {aesthetic_score:.1f}")
                    description_parts.append(f"Aesthetic: {aesthetic_score:.1f}")
            except Exception as e:
                logger.error(f"Error during aesthetic scoring: {e}")
        
        logger.info(f"Writing XMP with description: {' | '.join(description_parts)}")
        write_xmp_tag(xmp_path, description_parts)
        logger.info(f"Successfully wrote XMP to: {xmp_path}")
        
        return True, None
    except Exception as e:
        error_msg = f"Error processing {arw_path.name}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return False, error_msg
    finally:
        current_process = None

def main(folder_path, max_processes=None, use_nima=False):
    global interrupted, pool
    
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
        
        logger.info(f"Using {num_processes} processes for parallel processing")
        logger.info(f"Found {len(arw_files)} files to process")
        if use_nima and TENSORFLOW_AVAILABLE:
            logger.info("NIMA scoring enabled")
        else:
            logger.info("NIMA scoring disabled")
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
                    partial(process_arw_file, use_nima=use_nima),
                    arw_files
                ):
                    if interrupted:
                        logger.info("Interrupting processing...")
                        break
                    results.append(result)
                    pbar.update(1)
            
            # Анализируем результаты
            successful = sum(1 for success, _ in results if success)
            failed = len(results) - successful
            
            logger.info(f"Processing completed. Success: {successful}, Failed: {failed}")
            
            if failed > 0:
                logger.warning("Failed files:")
                for success, error in results:
                    if not success:
                        logger.warning(error)
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
    parser.add_argument('--use-nima', action='store_true', help='Enable NIMA scoring')
    args = parser.parse_args()
    
    try:
        main(args.folder, args.processes, args.use_nima)
    except KeyboardInterrupt:
        logger.info("Main process interrupted. Shutting down...")
        force_quit()
