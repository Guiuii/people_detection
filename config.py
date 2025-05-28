# Конфигурация для проекта детекции людей с помощью YOLOv8

CONFIG = {
    # Путь к входному видеофайлу
    "input_path": "crowd.mp4",
    
    # Путь для сохранения обработанного видео
    "output_path": "detected_people.mp4",
    
    # Выбор модели YOLOv8
    # Доступные варианты: 
    #   'yolov8n.pt' - nano (самая быстрая)
    #   'yolov8s.pt' - small
    #   'yolov8m.pt' - medium (оптимальный баланс)
    #   'yolov8l.pt' - large
    #   'yolov8x.pt' - xlarge (самая точная)
    "model_weights": "yolov8m.pt",
    
    # Параметры детекции
    "detection": {
        'conf': 0.5,       # Порог уверенности для детекции (0-1)
        'iou': 0.7,        # Порог IoU для подавления дубликатов (0-1)
        'classes': [0],    # Класс 0 - человек
        'verbose': False   # Вывод дополнительной информации
    },
    
    # Параметры визуализации
    "visualization": {
        'color': (0, 255, 0),  # Цвет bounding box (B, G, R)
        'thickness': 2,         # Толщина линий
        'font_scale': 0.5,      # Размер шрифта
        'font_thickness': 2,    # Толщина шрифта
        'show_labels': True     # Показывать метки с уверенностью
    },
    
    # Параметры обработки видео
    "video": {
        'codec': 'mp4v',        # Кодек для сохранения видео
        'show_progress': True   # Показывать прогресс-бар
    }
}