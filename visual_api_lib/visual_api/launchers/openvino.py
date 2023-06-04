"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import numpy as np

import logging as log
from .base_launcher import BaseLauncher, Metadata
from ..common import NetworkInfo, Layout
try:
    from openvino.runtime import AsyncInferQueue, Core, PartialShape, layout_helpers, get_version, Dimension
    openvino_absent = False
except ImportError:
    openvino_absent = True


def create_core():
    if openvino_absent:
        raise ImportError('The OpenVINO package is not installed')

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    return Core()


MAPPING_TYPES = {
    "f16": np.float16,
    "f32": np.float32,
    "I16": np.int16,
    "I32": np.int32,
    "I8": np.int8,
    "U8": np.uint8
}

class OpenVINOLauncher(BaseLauncher):
    __provider__ = "openvino"

    def __init__(self, model_configuration: dict):
        self.core = create_core()
        self.model_path = model_configuration.get("model_path", "")
        self.weights_path = model_configuration.get("weights_path", "")
        # Set inference parameters
        self.device = "CPU"
        self.max_num_requests = 1
        # Load model
        self.load_model()
        # Compile model
        self.compile_model()
        # get info about model
        self.model_info = NetworkInfo(self.get_input_layers(), self.get_output_layers())

    def load_model(self):
        self.model_from_buffer = isinstance(self.model_path, bytes) and isinstance(self.weights_path, bytes)
        log.info('Reading model {}'.format('from buffer' if self.model_from_buffer else self.model_path))
        weights = self.weights_path if self.model_from_buffer else ''
        self.model = self.core.read_model(self.model_path, weights)

    def compile_model(self):
        self.compiled_model = self.core.compile_model(self.model, self.device)
        self.async_queue = AsyncInferQueue(self.compiled_model, self.max_num_requests)
        if self.max_num_requests == 0:
            # +1 to use it as a buffer of the pipeline
            self.async_queue = AsyncInferQueue(self.compiled_model, len(self.async_queue) + 1)

        log.info('The model {} is loaded to {}'.format("from buffer" if self.model_from_buffer else self.model_path, self.device))
        self.log_runtime_settings()

    def log_runtime_settings(self):
        devices = set(self.device)
        if 'AUTO' not in devices:
            for device in devices:
                try:
                    nstreams = self.compiled_model.get_property(device + '_THROUGHPUT_STREAMS')
                    log.info('\tDevice: {}'.format(device))
                    log.info('\t\tNumber of streams: {}'.format(nstreams))
                    if device == 'CPU':
                        nthreads = self.compiled_model.get_property('CPU_THREADS_NUM')
                        log.info('\t\tNumber of threads: {}'.format(nthreads if int(nthreads) else 'AUTO'))
                except RuntimeError:
                    pass
        log.info('\tNumber of model infer requests: {}'.format(len(self.async_queue)))

    def get_input_layers(self):
        inputs = {}
        for input in self.model.inputs:
            input_shape = get_input_shape(input)
            input_layout = self.get_layout_for_input(input, input_shape)
            inputs[input.get_any_name()] = Metadata(input.get_names(), input_shape, layout=input_layout, type=MAPPING_TYPES[input.get_element_type().get_type_name()])
        inputs = self._get_meta_from_ngraph(inputs)
        return inputs

    def get_layout_for_input(self, input, shape=None) -> str:
        input_layout = ''
        if not layout_helpers.get_layout(input).empty:
            input_layout = Layout.from_openvino(input)
        else:
            input_layout = Layout.from_shape(shape if shape is not None else input.shape)
        return input_layout

    def _get_meta_from_ngraph(self, layers_info):
        for node in self.model.get_ordered_ops():
            layer_name = node.get_friendly_name()
            if layer_name not in layers_info.keys():
                continue
            layers_info[layer_name].meta = node.get_attributes()
            layers_info[layer_name].type = node.get_type_name()
        return layers_info

    def operations_by_type(self, operation_type):
        layers_info = {}
        for node in self.model.get_ordered_ops():
            if node.get_type_name() == operation_type:
                layer_name = node.get_friendly_name()
                layers_info[layer_name] = Metadata(type=node.get_type_name(), meta=node.get_attributes())
        return layers_info

    def get_output_layers(self):
        output_info = {}
        for output in self.model.outputs:
            output_shape = output.partial_shape.get_min_shape() if self.model.is_dynamic() else output.shape
            output_info[output.get_any_name()] = Metadata(output.get_names(), list(output_shape), type=MAPPING_TYPES[output.get_element_type().get_type_name()])

        return output_info

    def infer_sync(self, dict_data):
        # convert types of tensors
        self.infer_request = self.async_queue[self.async_queue.get_idle_request_id()]
        self.infer_request.infer(dict_data)
        output_dict = self.get_raw_result(self.infer_request)

        return output_dict

    def get_raw_result(self, request):
        return {key: request.get_tensor(key).data for key in self.get_output_layers()}

    def reshape_model(self, new_shape):
        new_shape = {name: PartialShape(
            [Dimension(dim) if not isinstance(dim, tuple) else Dimension(dim[0], dim[1])
            for dim in shape]) for name, shape in new_shape.items()}
        self.model.reshape(new_shape)

def get_input_shape(input_tensor):
    def string_to_tuple(string, casting_type=int):
        processed = string.replace(' ', '').replace('(', '').replace(')', '').split(',')
        processed = filter(lambda x: x, processed)
        return tuple(map(casting_type, processed)) if casting_type else tuple(processed)
    if not input_tensor.partial_shape.is_dynamic:
        return list(input_tensor.shape)
    ps = str(input_tensor.partial_shape)
    if ps[0] == '[' and ps[-1] == ']':
        ps = ps[1:-1]
    preprocessed = ps.replace('{', '(').replace('}', ')').replace('?', '-1')
    preprocessed = preprocessed.replace('(', '').replace(')', '')
    if '..' in preprocessed:
        shape_list = []
        for dim in preprocessed.split(','):
            if '..' in dim:
                shape_list.append(string_to_tuple(dim.replace('..', ',')))
            else:
                shape_list.append(int(dim))
        return shape_list
    return string_to_tuple(preprocessed)
