import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, List
from torchvision import transforms
from PIL import Image
from supervision import Detections
from ByteTrack.yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch

from config import IDX2CHAR

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

# Text Recognition Utils
def decode_text(labels):
  tokens = F.softmax(labels, 2).argmax(2)
  tokens = tokens.numpy().T
  plates = []
  for token in tokens:
      chars = [IDX2CHAR[idx] for idx in token]
      plate = ''.join(chars)
      plates.append(plate)
  return plates

def remove_duplicates(text):
  if len(text) > 1:
    letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
  elif len(text) == 1:
    letters = [text[0]]
  else:
    return ''
  return ''.join(letters)

def correct_text(word):
  parts = word.split('-')
  parts = [remove_duplicates(part) for part in parts]
  corrected_text = ''.join(parts)
  return corrected_text

def recognize_text(np_image: np.ndarray, model):
  pil_image = Image.fromarray(np_image).convert('RGB')

  # Preprocess the image
  transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  image = transform(pil_image)
  image = image.unsqueeze(0)

  # Perform inference
  with torch.no_grad():
    output = model(image)
    recognized_text = decode_text(output)
    recognized_text = correct_text(recognized_text[0])

  return recognized_text
