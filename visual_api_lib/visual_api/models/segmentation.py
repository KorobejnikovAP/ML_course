"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import cv2
import numpy as np

from ..common import NumericalValue, ListValue, StringValue, load_labels, NetworkInfo, sigmoid

from .image_model import ImageModel
from .detection import Detection
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


class YoloV8Seg(Detection):
    '''An abstract wrapper for instance segmentation model

    The SegmentationModel must have a single image input.
    It inherits `preprocess` from `ImageModel` wrapper. Also, it defines `_resize_detections` method,
    which should be used in `postprocess`, to clip bounding boxes and resize ones to original image shape.

    The `postprocess` method must be implemented in a specific inherited wrapper.
    '''
    __model__ = 'yolov8-seg'


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

        self._check_io_number(1, 2)

        if self.path_to_labels:
            self.labels = load_labels(self.path_to_labels)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'confidence_threshold': NumericalValue(default_value=0.5, description="Threshold value for detection box confidence"),
            "iou_threshold": NumericalValue(default_value=0.65, description="Threshold for NMS filtering"),
            'labels': ListValue(description="List of class labels"),
            'path_to_labels': StringValue(
                description="Path to file with labels. Overrides the labels, if they sets via 'labels' parameter"
            )
        })

        return parameters

    def postprocess(self, outputs, meta):
        for value in outputs.values():
            if value.ndim == 4:
                mask = value[0]
            if value.ndim == 3:
                predictions = cv2.transpose(value[0])


        print(mask.shape, predictions.shape)

        # nms for boxes
        scores = np.max(predictions[:, 4:84], axis=1)
        boxes = predictions[:, :4]
        boxes[:, 0] -= 0.5 * predictions[:, 2]
        boxes[:, 1] -= 0.5 * predictions[:, 3]
        keep_indexes = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)
        print(keep_indexes)

        # from ultralytics
        mask_channels = mask.shape[0]
        masks = self.process_mask(mask, predictions[keep_indexes, -mask_channels:])  # HWC

        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        scale_h = meta["original_shape"][0] / self.h
        scale_w = meta["original_shape"][1] / self.w
        boxes *= np.array([scale_w, scale_h, scale_w, scale_h])

        # resize mask to input image
        masks = cv2.resize(masks, (meta["original_shape"][0], meta["original_shape"][1]))

        masks = np.where(masks > 0.5, 1, 0)

        return masks

    def process_mask(self, mask, pred_masks ):
        """
        Apply masks to bounding boxes using the output of the mask head.

        Args:
            mask (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
            pred_masks (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.

        Returns:
            (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
                are the height and width of the mask. The mask is applied to the bounding boxes.
        """

        masks = np.reshape(sigmoid(pred_masks @ np.reshape(mask, (mask.shape[0], -1))), (-1, mask.shape[1], mask.shape[2])) #.view(-1, mh, mw)  # CHW

        return masks

