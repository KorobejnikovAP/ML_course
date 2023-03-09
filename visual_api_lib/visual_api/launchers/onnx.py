"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import numpy as np

from .base_launcher import BaseLauncher, Metadata
from ..common import NetworkInfo
from onnxruntime import InferenceSession


MAPPING_TYPES = {
    "tensor(float)": np.float32,
    "tensor(double)": np.double,
    "tensor(uint8)": np.uint8
}

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
            input_info[input_node.name] = Metadata(input_node.name, input_node.shape, type=MAPPING_TYPES[input_node.type])

        return input_info

    def get_output_layers(self):
        output_info = {}
        for output_node in self.session.get_outputs():
            output_info[output_node.name] = Metadata(output_node.name, output_node.shape, type=MAPPING_TYPES[output_node.type])

        return output_info

    def infer_sync(self, dict_data):
        # convert types of tensors
        for key, input_tensor in dict_data.items():
            dict_data[key] = input_tensor.astype(self.model_info.inputs_info[key].type)

        outputs = self.session.run(list(self.model_info.outputs_info.keys()), dict_data)

        output_dict = {}
        for key, value in zip(self.model_info.outputs_info.keys(), outputs):
            output_dict[key] = value

        return output_dict
