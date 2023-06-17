"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import cv2
import numpy as np

from ..common import NumericalValue, ListValue, StringValue, load_labels, NetworkInfo

from .image_model import ImageModel
from typing import List

class DetectionOutput:
    '''Class to describe output of detection models

    Represents as rectangle
    '''
    def __init__(self, xmin, ymin, xmax, ymax, score, id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.id = id

    def bottom_left_point(self):
        return self.xmin, self.ymin

    def top_right_point(self):
        return self.xmax, self.ymax

    def get_coords(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


class Detection(ImageModel):
    '''An abstract wrapper for object detection model

    The DetectionModel must have a single image input.
    It inherits `preprocess` from `ImageModel` wrapper. Also, it defines `_resize_detections` method,
    which should be used in `postprocess`, to clip bounding boxes and resize ones to original image shape.

    The `postprocess` method must be implemented in a specific inherited wrapper.
    '''
    __model__ = 'detection'


    def __init__(self, network_info: NetworkInfo, configuration=None):
        '''Detection Model constructor

        It extends the `ImageModel` construtor.

        Args:
            network_info (NetworkInfo): it contains information about inputs and outputs of model
            configuration (dict, optional): it contains values for parameters accepted by specific
              wrapper (`confidence_threshold`, `labels` etc.) which are set as data attributes

        Raises:
            WrapperError: if the model has more than 1 image inputs
        '''
        super().__init__(network_info, configuration)

        if not self.image_blob_name:
            self.raise_error("The Wrapper supports only one image input, but {} found".format(
                len(self.image_blob_names)))

        if self.path_to_labels:
            self.labels = load_labels(self.path_to_labels)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'confidence_threshold': NumericalValue(default_value=0.5, description="Threshold value for detection box confidence"),
            'labels': ListValue(description="List of class labels"),
            'path_to_labels': StringValue(
                description="Path to file with labels. Overrides the labels, if they sets via 'labels' parameter"
            )
        })

        return parameters

    def _resize_detections(self, detections: List[DetectionOutput], meta: dict):
        '''Resizes detection bounding boxes according to source image shape.

        It implements image resizing depending on the set `resize_type`(see `ImageModel` for details).
        Next, it applies bounding boxes clipping.

        Args:
            detections (List[DetectionOutput]): list of detections with coordinates in normalized form
            meta (dict): the input metadata obtained from `preprocess` method

        Returns:
            - list of detections with resized and clipped coordinates fit to source image

        Raises:
            WrapperError: If the model uses custom resize or `resize_type` is not set
        '''
        resized_shape = meta['resized_shape']
        original_shape = meta['original_shape']

        if self.resize_type == 'fit_to_window_letterbox':
            detections = resize_detections_letterbox(detections, original_shape[1::-1], resized_shape[1::-1])
        elif self.resize_type == 'fit_to_window':
            detections = resize_detections_with_aspect_ratio(detections, original_shape[1::-1], resized_shape[1::-1], (self.w, self.h))
        elif self.resize_type == 'standard':
            detections = resize_detections(detections, original_shape[1::-1])
        else:
            self.raise_error('Unknown resize type {}'.format(self.resize_type))
        return clip_detections(detections, original_shape)


def resize_detections(detections, original_image_size):
    for detection in detections:
        detection.xmin *= original_image_size[0]
        detection.xmax *= original_image_size[0]
        detection.ymin *= original_image_size[1]
        detection.ymax *= original_image_size[1]
    return detections

def resize_detections_with_aspect_ratio(detections, original_image_size, resized_image_size, model_input_size):
    scale_x = model_input_size[0] / resized_image_size[0] * original_image_size[0]
    scale_y = model_input_size[1] / resized_image_size[1] * original_image_size[1]
    for detection in detections:
        detection.xmin *= scale_x
        detection.xmax *= scale_x
        detection.ymin *= scale_y
        detection.ymax *= scale_y
    return detections

def resize_detections_letterbox(detections, original_image_size, resized_image_size):
    scales = [x / y for x, y in zip(resized_image_size, original_image_size)]
    scale = min(scales)
    scales = (scale / scales[0], scale / scales[1])
    offset = [0.5 * (1 - x) for x in scales]
    for detection in detections:
        detection.xmin = ((detection.xmin - offset[0]) / scales[0]) * original_image_size[0]
        detection.xmax = ((detection.xmax - offset[0]) / scales[0]) * original_image_size[0]
        detection.ymin = ((detection.ymin - offset[1]) / scales[1]) * original_image_size[1]
        detection.ymax = ((detection.ymax - offset[1]) / scales[1]) * original_image_size[1]
    return detections

def clip_detections(detections, size):
    for detection in detections:
        detection.xmin = max(int(detection.xmin), 0)
        detection.ymin = max(int(detection.ymin), 0)
        detection.xmax = min(int(detection.xmax), size[1])
        detection.ymax = min(int(detection.ymax), size[0])
    return detections

class YoloV8(Detection):
    __model__ = "yolo-v8"

    def __init__(self,  network_info: NetworkInfo, configuration=None):
        super().__init__(network_info, configuration)
        # model have 1 image input and 1 ouptut
        self._check_io_number(1, 1)
        self.output_blob_name = next(iter(self.outputs))

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                "iou_threshold": NumericalValue(default_value=0.65, description="Threshold for NMS filtering"),
            }
        )
        return parameters

    def postprocess(self, outputs, meta):
        predictions = outputs[self.output_blob_name]
        predictions = cv2.transpose(predictions[0])

        scores = np.max(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        boxes[:, 0] -= 0.5 * boxes[:, 2]
        boxes[:, 1] -= 0.5 * boxes[:, 3]
        box_indexes = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)

        boxes = boxes[box_indexes]
        scores = scores[box_indexes]
        class_ids = np.argmax(predictions[box_indexes][:, 4:], axis=1)

        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        scale_h = meta["original_shape"][0] / self.h
        scale_w = meta["original_shape"][1] / self.w
        boxes *= np.array([scale_w, scale_h, scale_w, scale_h])

        detections = [DetectionOutput(*boxes[i], scores[i], class_ids[i]) for i in range(len(box_indexes))]

        return clip_detections(detections, meta["original_shape"])
