"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import cv2
import numpy as np
from typing import Optional
from openvino.runtime import layout_helpers

class InputTransform:
    def __init__(self, reverse_input_channels=False, mean_values=None, scale_values=None):
        self.reverse_input_channels = reverse_input_channels
        self.is_trivial = not (reverse_input_channels or mean_values or scale_values)
        self.means = np.array(mean_values, dtype=np.float32) if mean_values else np.array([0., 0., 0.])
        self.std_scales = np.array(scale_values, dtype=np.float32) if scale_values else np.array([1., 1., 1.])

    def __call__(self, inputs):
        if self.is_trivial:
            return inputs
        if self.reverse_input_channels:
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        return (inputs - self.means) / self.std_scales

def softmax(logits, axis=None, keepdims=False):
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp, axis=axis, keepdims=keepdims)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def resolution(value):
    try:
        result = [int(v) for v in value.split('x')]
        if len(result) != 2:
            raise RuntimeError('Correct format of --output_resolution parameter is "width"x"height".')
    except ValueError:
        raise RuntimeError('Correct format of --output_resolution parameter is "width"x"height".')
    return result

class Layout:
    def __init__(self, layout = '') -> None:
        self.layout = layout

    @staticmethod
    def from_shape(shape):
        '''
        Create Layout from given shape
        '''
        if len(shape) == 2:
            return 'NC'
        if len(shape) == 3:
            return 'CHW' if shape[0] in range(1, 5) else 'HWC'
        if len(shape) == 4:
            return 'NCHW' if shape[1] in range(1, 5) else 'NHWC'

        raise RuntimeError("Get layout from shape method doesn't support {}D shape".format(len(shape)))

    @staticmethod
    def from_openvino(input):
        '''
        Create Layout from openvino input
        '''
        return layout_helpers.get_layout(input).to_string().strip('[]').replace(',', '')

    @staticmethod
    def from_user_layouts(input_names: set, user_layouts: dict):
        '''
        Create Layout for input based on user info
        '''
        for input_name in input_names:
            if input_name in user_layouts:
                return user_layouts[input_name]
        return user_layouts.get('', '')

    @staticmethod
    def parse_layouts(layout_string: str) -> Optional[dict]:
        '''
        Parse layout parameter in format "input0:NCHW,input1:NC" or "NCHW" (applied to all inputs)
        '''
        if not layout_string:
            return None
        search_string = layout_string if layout_string.rfind(':') != -1 else ":" + layout_string
        colon_pos = search_string.rfind(':')
        user_layouts = {}
        while (colon_pos != -1):
            start_pos = search_string.rfind(',')
            input_name = search_string[start_pos + 1:colon_pos]
            input_layout = search_string[colon_pos + 1:]
            user_layouts[input_name] = input_layout
            search_string = search_string[:start_pos + 1]
            if search_string == "" or search_string[-1] != ',':
                break
            search_string = search_string[:-1]
            colon_pos = search_string.rfind(':')
        if search_string != "":
            raise ValueError("Can't parse input layout string: " + layout_string)
        return user_layouts

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels_map = [x.strip() for x in f]
    return labels_map
