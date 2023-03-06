"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from .base_adapter import BaseLauncher
from onnxruntime import InferenceSession


class ONNXLauncher(BaseLauncher):
    def __init__(self, model_path):
        # create InferenceSession and load model
        self.session = InferenceSession(model_path)

    def get_input_layers(self):
        inpurt = self.session.get_inputs()
        return super().get_input_layers()

