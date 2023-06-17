"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from .base_launcher import BaseLauncher, Metadata
from ..common import NetworkInfo
import tensorflow as tf


class TFLiteLauncher(BaseLauncher):
    __provider__ = "tflite"
    def __init__(self, model_configuration):
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_configuration.get("model_path", ""))
        self.interpreter.allocate_tensors()
        # get info about model
        self.model_info = NetworkInfo(self.get_input_layers(), self.get_output_layers())

    def get_input_layers(self):
        input_info = {}
        for tensor_info in self.interpreter.get_input_details():
            print(tensor_info)
            input_info[tensor_info["name"]] = Metadata(tensor_info["name"], list(tensor_info["shape"]),
                                                       index=tensor_info["index"], type=tensor_info["dtype"])

        return input_info

    def get_output_layers(self):
        output_info = {}
        for tensor_info in self.interpreter.get_output_details():
            output_info[tensor_info["name"]] = Metadata(tensor_info["name"], list(tensor_info["shape"]),
                                                        index=tensor_info["index"], type=tensor_info["dtype"])

        return output_info

    def infer_sync(self, dict_data: dict):
        # convert types of tensors
        for key, input_tensor in dict_data.items():
            dict_data[key] = input_tensor.astype(self.model_info.inputs_info[key].type)
        for tensor_name, input_data in dict_data.items():
            self.interpreter.set_tensor(self.model_info.inputs_info[tensor_name].index, input_data)
        self.interpreter.invoke()

        output_dict = {}
        for tensor_name in self.model_info.outputs_info.keys():
            output_dict[tensor_name] = self.interpreter.get_tensor(self.model_info.outputs_info[tensor_name].index)

        return output_dict
