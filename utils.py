import numpy as np
from typing import Union, List
from supervision import Detections
from ByteTrack.yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch

def filter_detections(detections: Detections, mask: np.ndarray) -> Detections:
  return Detections(
    xyxy=detections.xyxy[mask],
    confidence=detections.confidence[mask],
    class_id=detections.class_id[mask],
    tracker_id=detections.tracker_id[mask]
    if detections.tracker_id is not None
    else None,
  )
  
def detections2boxes(detections: Union[Detections, list]) -> np.ndarray:
  if(isinstance(detections, Detections)):
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))
  elif(isinstance(detections, list)):
    return [np.append(d[0], d[1]) for d in detections]
  else:
    return np.empty((0,))

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)

def match_detections_with_tracks(detections: Union[Detections, list], tracks: List[STrack]) -> Detections:
    detections_xyxy = np.empty((0,))

    if(isinstance(detections, Detections)):
        detections_xyxy = detections.xyxy
    elif(isinstance(detections, list)):
        detections_xyxy = np.array([d[0].tolist() for d in detections])

    if not np.any(detections_xyxy) or len(tracks) == 0:
        return np.empty((0,))

    track_boxes = tracks2boxes(tracks=tracks)
    # print('track_boxes', track_boxes)
    iou = box_iou_batch(track_boxes, detections_xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_idx, detection_idx in enumerate(track2detection):
        if iou[tracker_idx, detection_idx] != 0:
            tracker_ids[detection_idx] = tracks[tracker_idx].track_id
    
    return tracker_ids