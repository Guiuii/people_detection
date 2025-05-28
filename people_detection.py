import cv2
from ultralytics import YOLO
from tqdm import tqdm


def process_video(config: dict):
    """
    Обрабатывает видеофайл, детектируя людей с помощью YOLOv8, 
    используя параметры из конфигурационного словаря.
    
    Args:
        config (dict): Словарь с конфигурационными параметрами
    """
    # извлечение параметров из конфига
    input_path = config["input_path"]
    output_path = config["output_path"]
    model_weights = config["model_weights"] # По умолчанию yolov8m.pt
    detection_params = config["detection"]
    viz_params = config["visualization"]
    video_params = config["video"]
    
    # загрузка предобученной модели YOLOv8
    model = YOLO(model_weights)
    
    # открытие видеофайла для чтения
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видеофайл {input_path}")
    
    # извлечение параметров видео:
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # FPS (количество кадров в секунду)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ширина кадра в пикселях
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Высота кадра в пикселях
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Общее количество кадров в видео
    
    # настройка VideoWriter для сохранения обработанного видео
    fourcc = cv2.VideoWriter_fourcc(*video_params['codec']) # по умолчанию mp4v
    # создание объекта для записи видео с теми же параметрами, что и исходное
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # обработка видео по кадрам
    for _ in tqdm(range(total_frames), desc="Обработка видео"):

        ret, frame = cap.read() # ret - флаг успешного чтения, frame - сам кадр в виде numpy массива
        if not ret:
            break
        
        # детекция объектов на кадре:
        results = model.predict(
            frame, 
            conf=detection_params['conf'],
            iou=detection_params['iou'],
            classes=detection_params['classes'], # детектируем только класс 0 - люди
            verbose=detection_params['verbose']
        )
        
        # results содержит информацию обо всех обнаруженных объектах
        for result in results:  

            boxes = result.boxes # информация о bounding boxes
            
            # обработка каждой обнаруженной рамки
            for box in boxes:

                # координаты рамки(x1,y1 - левый верхний, x2,y2 - правый нижний)
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                conf = float(box.conf[0]) # confidence score
                cls_id = int(box.cls[0]) # всегда 0, т.к. только люди
                
                # отрисовка bounding box на кадре:
                cv2.rectangle(
                    frame, 
                    (x1, y1), 
                    (x2, y2), 
                    viz_params['color'],  # цвет bb - по умолчанию зеленый
                    viz_params['thickness'] # толщина линии bb - по умолчанию 2
                )
                
                label = f"{result.names[cls_id]}: {conf:.2f}" # класс и уверенность
                
                # текст с классом и уверенностью
                if viz_params['show_labels']:
                    label = f"{result.names[cls_id]}: {conf:.2f}"
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 10), # (x1, y1 - 10) - текст над bb
                        cv2.FONT_HERSHEY_SIMPLEX, # шрифт
                        viz_params['font_scale'], # масштаб шрифта
                        viz_params['color'], # цвет текста
                        viz_params['font_thickness'] # толщина
                    )
        
        # запись обработанного кадра в выходное видел
        out.write(frame)
        
    # освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
from config import CONFIG  # Импортируем конфиг

print(f"Начало обработки видео {CONFIG['input_path']}...")
print(f"Используемая модель: {CONFIG['model_weights']}")
print(f"Порог уверенности: {CONFIG['detection']['conf']}")

process_video(CONFIG)

print(f"Обработка завершена. Результат сохранён в {CONFIG['output_path']}")