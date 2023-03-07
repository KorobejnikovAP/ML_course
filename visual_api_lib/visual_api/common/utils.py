"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import cv2
import numpy as np

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

def resolution(value):
    try:
        result = [int(v) for v in value.split('x')]
        if len(result) != 2:
            raise RuntimeError('Correct format of --output_resolution parameter is "width"x"height".')
    except ValueError:
        raise RuntimeError('Correct format of --output_resolution parameter is "width"x"height".')
    return result
