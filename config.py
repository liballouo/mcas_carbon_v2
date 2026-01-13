# 定義不同視頻的點位參數
POINTS_CONFIGS = {
    # 演講廳
    'lecture_room': {
        'topleft': [0, 0],
        'bottomleft': [25, 1100],
        'topright': [1100, -10],
        'bottomright': [1850, 600]
    },
    # 電腦教室視角A
    'computer_room_A': {
        'topleft': [0, 50],
        'bottomleft': [-10, 800],
        'topright': [1850, -10],
        'bottomright': [2000, 300]
    },
    # 電腦教室視角B
    'computer_room_B': {
        'topleft': [800, 125],
        'bottomleft': [-350, 350],
        'topright': [1700, 250],
        'bottomright': [1200, 1200]
    },
    # 大廳視角A
    'hall_A': {
        'topleft': [50, 200],
        'bottomleft': [225, 1150],
        'topright': [1050, 50],
        'bottomright': [1950, 500]
    },
    # 大廳視角B
    'hall_B': {
        'topleft': [200, 300],
        'bottomleft': [325, 1150],
        'topright': [1350, 200],
        'bottomright': [1950, 550]
    },
}

# 定義不同空間的預測閾值
PREDICTION_THRESHOLDS_CONFIG = {
    'lecture_room': {
        "tf": {"conf": 0.0001, "iou": 0.0001},
        "tr": {"conf": 0.0001, "iou": 0.1},
        "bf": {"conf": 0.2, "iou": 0.1},
        "br": {"conf": 0.2, "iou": 0.1}
    },
    'computer_room_A': {
        "tf": {"conf": 0.1, "iou": 0.1},
        "tr": {"conf": 0.1, "iou": 0.1},
        "bf": {"conf": 0.1, "iou": 0.1},
        "br": {"conf": 0.1, "iou": 0.1}
    },
    'computer_room_B': {
        "tf": {"conf": 0.1, "iou": 0.1},
        "tr": {"conf": 0.1, "iou": 0.1},
        "bf": {"conf": 0.1, "iou": 0.1},
        "br": {"conf": 0.1, "iou": 0.1}
    },
    'hall_A': {
        "tf": {"conf": 0.1, "iou": 0.1},
        "tr": {"conf": 0.1, "iou": 0.1},
        "bf": {"conf": 0.1, "iou": 0.1},
        "br": {"conf": 0.1, "iou": 0.1}
    },
    'hall_B': {
        "tf": {"conf": 0.1, "iou": 0.1},
        "tr": {"conf": 0.1, "iou": 0.1},
        "bf": {"conf": 0.1, "iou": 0.1},
        "br": {"conf": 0.1, "iou": 0.1}
    },
}