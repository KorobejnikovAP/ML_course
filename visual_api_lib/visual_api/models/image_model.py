"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from .base_model import Model
from ..common import NetworkInfo, BooleanValue, ListValue, StringValue, RESIZE_TYPES, InputTransform, pad_image, INTERPOLATION_TYPES


class ImageModel(Model):
    '''An abstract wrapper for an image-based model
    The ImageModel has 1 or more inputs with images - 4D tensors with NHWC or NCHW layout.
    It may support additional inputs - 2D tensors.
    The ImageModel implements basic preprocessing for an image provided as model input.
    See `preprocess` description.
    The `postprocess` method must be implemented in a specific inherited wrapper.
    Attributes:
        image_blob_names (List[str]): names of all image-like inputs (4D tensors)
        image_info_blob_names (List[str]): names of all secondary inputs (2D tensors)
        image_blob_name (str): name of the first image input
        nchw_layout (bool): a flag whether the model input layer has NCHW layout
        resize_type (str): the type for image resizing (see `RESIZE_TYPE` for info)
        resize (function): resizing function corresponding to the `resize_type`
        input_transform (InputTransform): instance of the `InputTransform` for image normalization
    '''

    def __init__(self, network_info: NetworkInfo, configuration=None):
        '''Image model constructor
        It extends the `Model` constructor.
        Args:
            network_info (NetworkInfo): it contains information about inputs and outputs of model
            configuration (dict, optional): it contains values for parameters accepted by specific
              wrapper (`confidence_threshold`, `labels` etc.) which are set as data attributes
        Raises:
            WrapperError: if the wrapper configuration is incorrect
        '''
        super().__init__(network_info, configuration)
        self.image_blob_names, self.image_info_blob_names = self._get_inputs()
        self.image_blob_name = self.image_blob_names[0]

        self.nchw_layout = self.inputs[self.image_blob_name].shape[1] == 3
        if self.nchw_layout:
            self.n, self.c, self.h, self.w = self.inputs[self.image_blob_name].shape
        else:
            self.n, self.h, self.w, self.c = self.inputs[self.image_blob_name].shape
        self.resize_func = RESIZE_TYPES[self.resize_type]
        self.input_transform = InputTransform(self.reverse_input_channels, self.mean_values, self.scale_values)

    '''Wrapper for image resize function
    Args: input image
    '''
    def resize(self, image):
        if (self.w != -1 and self.h != -1):
            resized_image = self.resize_func(image, (self.w, self.h), interpolation=INTERPOLATION_TYPES[self.interpolation_type])
        else:
            resized_image = image

        return resized_image

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mean_values': ListValue(
                default_value=None,
                description='Normalization values, which will be subtracted from image channels for image-input layer during preprocessing'
            ),
            'scale_values': ListValue(
                default_value=None,
                description='Normalization values, which will divide the image channels for image-input layer'
            ),
            'reverse_input_channels': BooleanValue(default_value=False, description='Reverse the channel order'),
            'resize_type': StringValue(
                default_value='standard', choices=tuple(RESIZE_TYPES.keys()),
                description="Type of input image resizing"
            ),
            'interpolation_type': StringValue(
                default_value='LINEAR', choices=tuple(INTERPOLATION_TYPES.keys()),
                description="Type of interpolation for input resizing"
            )
        })
        return parameters

    def _get_inputs(self):
        '''Defines the model inputs for images and additional info.
        Raises:
            WrapperError: if the wrapper failed to define appropriate inputs for images
        Returns:
            - list of inputs names for images
            - list of inputs names for additional info
        '''
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 4:
                image_blob_names.append(name)
            elif len(metadata.shape) == 2:
                image_info_blob_names.append(name)
            else:
                self.raise_error('Failed to identify the input for ImageModel: only 2D and 4D input layer supported')
        if not image_blob_names:
            self.raise_error('Failed to identify the input for the image: no 4D input layer found')
        return image_blob_names, image_info_blob_names

    def preprocess(self, inputs):
        '''Data preprocess method
        It performs basic preprocessing of a single image:
            - Resizes the image to fit the model input size via the defined resize type
            - Normalizes the image: subtracts means, divides by scales, switch channels BGR-RGB
            - Changes the image layout according to the model input layout
        Also, it keeps the size of original image and resized one as `original_shape` and `resized_shape`
        in the metadata dictionary.
        Note:
            It supports only models with single image input. If the model has more image inputs or has
            additional supported inputs, the `preprocess` should be overloaded in a specific wrapper.
        Args:
            inputs (ndarray): a single image as 3D array in HWC layout
        Returns:
            - the preprocessed image in the following format:
                {
                    'input_layer_name': preprocessed_image
                }
            - the input metadata, which might be used in `postprocess` method
        '''
        image = inputs
        meta = {'original_shape': image.shape}
        resized_image = self.resize(image)
        meta.update({'resized_shape': resized_image.shape})
        if self.resize_type == 'fit_to_window':
            resized_image = pad_image(resized_image, (self.w, self.h))
            meta.update({'padded_shape': resized_image.shape})
        resized_image = self.input_transform(resized_image)
        resized_image = self._change_layout(resized_image)
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def _change_layout(self, image):
        '''Changes the input image layout to fit the layout of the model input layer.
        Args:
            inputs (ndarray): a single image as 3D array in HWC layout
        Returns:
            - the image with layout aligned with the model layout
        '''
        if self.nchw_layout:
            image = image.transpose((2, 0, 1))  # HWC->CHW
            image = image.reshape((1, self.c, self.h, self.w))
        else:
            image = image.reshape((1, self.h, self.w, self.c))
        return image
