"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from .base_model import Model
from .image_model import ImageModel
from .classification import Classification
from .detection import Detection, DetectionOutput
from .segmentation import YoloV8Seg


__all__ = [
    "Model",
    "ImageModel",
    "Classification",
    "Detection",
    "DetectionOutput",
    "YoloV8Seg"
]
