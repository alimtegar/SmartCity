import cv2
import numpy as np
from ultralytics import YOLO
from supervision import Detections, BoxAnnotator, LineZoneAnnotator
import torch
import torchvision

from config import VIDEO_FILE_NAME, VEHICLE_DETECTION_MODEL, PLATE_DETECTION_MODEL, WANTED_CLASS_ID_LIST, LINE_COUNTER_START, LINE_COUNTER_END
from utils import filter_detections, detections2boxes, match_detections_with_tracks

from tracker import byte_tracker
from line_counter import CustomLineZone

vehicle_detection_model = YOLO(VEHICLE_DETECTION_MODEL)
plate_detection_model = YOLO(PLATE_DETECTION_MODEL)
vehicle_detection_model.fuse()

box_annotator = BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
line_annotator = LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
line_counter = CustomLineZone(start=LINE_COUNTER_START, end=LINE_COUNTER_END)

# Check if camera opened successfully
cap = cv2.VideoCapture(VIDEO_FILE_NAME)

if cap.isOpened() == False:
  print('Error when opening the video file!')
  
while cap.isOpened():
  # Capture frame-by-frame
  ret, frame = cap.read()
  
  if ret == True:
    # Detect vehicles in the frame
    results = vehicle_detection_model(frame)[0]
    vehicle_detections = Detections.from_yolov8(results)
    
    # Apply NMS and filter detections from unwanted classes
    kept_box_idxs = torchvision.ops.nms(torch.from_numpy(vehicle_detections.xyxy), torch.from_numpy(vehicle_detections.confidence), iou_threshold=0.5)
    mask = np.array([
      class_id in WANTED_CLASS_ID_LIST and idx in kept_box_idxs
      for idx, class_id in enumerate(vehicle_detections.class_id)
    ], dtype=bool)
    vehicle_detections = filter_detections(detections=vehicle_detections, mask=mask)    
    
    # Crop the frame and detect vehicles' plate
    for vehicle_detection_xyxy in vehicle_detections.xyxy:
      x1, y1, x2, y2 = map(int, vehicle_detection_xyxy[:4])
      vehicle_image = frame[y1:y2, x1:x2]
      # cv2.imshow('Vehicle Image', vehicle_image)
      # cv2.moveWindow('Vehicle Image', 200, 200)
      
      plate_detections = plate_detection_model.predict(source=vehicle_image, show=True)[0]
      
      if (len(plate_detections) > 0):
        max_plate_detection_xyxy = plate_detections.boxes[torch.argmax(plate_detections.boxes.conf)].xyxy[0]
        x1, y1, x2, y2 = map(int, max_plate_detection_xyxy[:4])
        plate_image = vehicle_image[y1:y2, x1:x2]
        cv2.imshow('Plate Image', plate_image)
        cv2.moveWindow('Plate Image', 0, 300)
    
    # Track vehicles
    tracks = byte_tracker.update(
      output_results=np.array(detections2boxes(detections=vehicle_detections)),
      img_info=frame.shape,
      img_size=frame.shape,
    )
    tracker_ids = match_detections_with_tracks(detections=vehicle_detections, tracks=tracks)

    # Assign tracker ID into detections
    vehicle_detections.tracker_id = np.array(tracker_ids)

    # Filter untracked detections
    mask = np.array([tracker_id is not None for tracker_id in vehicle_detections.tracker_id], dtype=bool)
    vehicle_detections = filter_detections(detections=vehicle_detections, mask=mask)

    # Count detections
    line_counter.trigger(detections=vehicle_detections)

    # Annotate bounding boxes and line
    labels = [
        f'{vehicle_detection_model.model.names[class_id]} {confidence:0.2f}' 
        for _, confidence, class_id, tracker_id, 
        in vehicle_detections
    ]
    frame = box_annotator.annotate(scene=frame, detections=vehicle_detections, labels=labels)
    line_annotator.annotate(frame=frame, line_counter=line_counter)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Press "Q" to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
  
# Release the video capture object When everything done
cap.release()

# Close all windows
cv2.destroyAllWindows()