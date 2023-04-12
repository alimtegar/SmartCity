from supervision import Point

VIDEO_FILE_NAME = 'vehicle-traffic.mp4'

VEHICLE_DETECTION_MODEL = 'yolov8n.pt'
PLATE_DETECTION_MODEL = './models/plate_detection_model.pt'

# Class IDs of Interest
WANTED_CLASS_ID_LIST = [
    1, # bicycle
    2, # car
    3, # motorcycle
    5, # bus
    7, # truck
]

LINE_COUNTER_START = Point(50, 1400)
LINE_COUNTER_END = Point(3840-50, 1400)