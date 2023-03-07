"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from .base_model import Model
from .image_model import ImageModel
from .classification import Classification


__all__ = [
    "Model",
    "ImageModel",
    "Classification"
]
