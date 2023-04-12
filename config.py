from supervision import Point

VIDEO_FILE_NAME = 'vehicle-traffic.mp4'

VEHICLE_DETECTION_MODEL = 'yolov8n.pt'
PLATE_DETECTION_MODEL = './model_weights/plate_detection_model.pt'
TEXT_RECOGNITION_MODEL = './model_weights/text_recognition_model.pt'

# Class IDs of Interest
WANTED_CLASS_ID_LIST = [
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
]

LINE_COUNTER_START = Point(50, 1400)
LINE_COUNTER_END = Point(3840-50, 1400)

# Text Recognition Configs
VOCABULARY = [
    '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
]
IDX2CHAR = {k: v for k, v in enumerate(VOCABULARY, start=0)}
CHAR2IDX = {v: k for k, v in IDX2CHAR.items()}

