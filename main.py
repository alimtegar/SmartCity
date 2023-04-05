import cv2
import numpy as np
from ultralytics import YOLO
from supervision import Detections, BoxAnnotator, LineZoneAnnotator
import torch
import torchvision

from config import VIDEO_FILE_NAME, MODEL, WANTED_CLASS_ID_LIST, LINE_COUNTER_START, LINE_COUNTER_END
from utils import filter_detections, detections2boxes, match_detections_with_tracks

from tracker import byte_tracker
from line_counter import CustomLineZone

model = YOLO(MODEL)
model.fuse()

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
    results = model(frame)[0]
    detections = Detections.from_yolov8(results)
    
    # Apply NMS and filter detections from unwanted classes
    kept_box_idxs = torchvision.ops.nms(torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence), iou_threshold=0.5)
    mask = np.array([
      class_id in WANTED_CLASS_ID_LIST and idx in kept_box_idxs
      for idx, class_id in enumerate(detections.class_id)
    ], dtype=bool)
    detections = filter_detections(detections=detections, mask=mask)    
    
    # Track vehicles
    tracks = byte_tracker.update(
      output_results=np.array(detections2boxes(detections=detections)),
      img_info=frame.shape,
      img_size=frame.shape,
    )
    tracker_ids = match_detections_with_tracks(detections=detections, tracks=tracks)

    # Assign tracker ID into detections
    detections.tracker_id = np.array(tracker_ids)

    # Filter untracked detections
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    detections = filter_detections(detections=detections, mask=mask)

    # Count detections
    line_counter.trigger(detections=detections)

    # Annotate bounding boxes and line
    labels = [
        f'{model.model.names[class_id]} {confidence:0.2f}' 
        for _, confidence, class_id, tracker_id, 
        in detections
    ]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
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