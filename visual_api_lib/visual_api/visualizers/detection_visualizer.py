import cv2

import logging as log
from typing import List
from visual_api.models import DetectionOutput

class DetectionVisualizer:
    def __init__(self, labels, palette) -> None:
        self.labels = labels
        self.palette = palette

    def draw_detections(self, frame, detections: List[DetectionOutput], output_transform=None):
        for detection in detections:
            class_id = int(detection.id)
            color = self.palette[class_id]
            det_label = self.labels[class_id] if self.labels and len(self.labels) >= class_id else '#{}'.format(class_id)
            xmin, ymin, xmax, ymax = detection.get_coords()
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        return frame


    def print_raw_results(self, detections, frame_id):
        log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
        log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
        for detection in detections:
            xmin, ymin, xmax, ymax = detection.get_coords()
            class_id = int(detection.id)
            det_label = self.labels[class_id] if self.labels and len(self.labels) >= class_id else '#{}'.format(class_id)
            log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                      .format(det_label, detection.score, xmin, ymin, xmax, ymax))
