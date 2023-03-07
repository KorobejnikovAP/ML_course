"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from .base_launcher import BaseLauncher, Metadata
from ..common import NetworkInfo
from onnxruntime import InferenceSession


class ONNXLauncher(BaseLauncher):
    __provider__ = "onnx"

    def __init__(self, model_path):
        # create InferenceSession and load model
        self.session = InferenceSession(model_path)
        # get info about model
        self.model_info = NetworkInfo(self.get_input_layers(), self.get_output_layers())

    def get_input_layers(self):
        input_info = {}
        for input_node in self.session.get_inputs():
            input_info[input_node.name] = Metadata(input_node.name, input_node.shape)

        return input_info

    def get_output_layers(self):
        output_info = {}
        for output_node in self.session.get_outputs():
            output_info[output_node.name] = Metadata(output_node.name, output_node.shape)

        return output_info

    def infer_sync(self, dict_data):
        outputs = self.session.run(list(self.model_info.outputs_info.keys()), dict_data)

        output_dict = {}
        for key, value in zip(self.model_info.outputs_info.keys(), outputs):
            output_dict[key] = value

        return output_dict
