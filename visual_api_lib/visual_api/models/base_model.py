"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import logging as log
from ..common.network_info import NetworkInfo

class WrapperError(RuntimeError):
    '''Special class for errors occurred in API wrappers'''
    def __init__(self, wrapper_name, message):
        super().__init__(f"{wrapper_name}: {message}")


class Model:
    '''An abstract model wrapper
    The abstract model wrapper is free from any executor dependencies.
    This class handle any info about used model and model's pre- and postprocessing.
    Next, it loads the provided configuration variables and sets it as wrapper attributes.
    The keys of the configuration dictionary should be presented in the `parameters` method.
    The `preprocess` and `postprocess` methods must be implemented in a specific inherited wrapper.
    Attributes:
        logger (Logger): instance of the Logger, need to print info about model
        inputs (dict): keeps the model inputs names and `Metadata` structure for each one
        outputs (dict): keeps the model outputs names and `Metadata` structure for each one
    '''

    __model__ = None # Abstract wrapper has no name

    def __init__(self, network_info: NetworkInfo, configuration=None):
        '''Model constructor
        Args:
            network_info (NetworkInfo): it contains information about inputs and outputs of model
            configuration (dict, optional): it contains values for parameters accepted by specific
              wrapper (`confidence_threshold`, `labels` etc.) which are set as data attributes
        Raises:
            WrapperError: if the wrapper configuration is incorrect
        '''
        self.logger = log.getLogger()
        self.inputs = network_info.inputs_info
        self.outputs = network_info.outputs_info
        for name, parameter in self.parameters().items():
            self.__setattr__(name, parameter.default_value)
        self._load_config(configuration)

    @classmethod
    def get_model(cls, name):
        subclasses = [subclass for subclass in cls.get_subclasses() if subclass.__model__]
        if cls.__model__:
            subclasses.append(cls)
        for subclass in subclasses:
            if name.lower() == subclass.__model__.lower():
                return subclass
        cls.raise_error('There is no model with name "{}" in list: {}'.format(
            name, ', '.join([subclass.__model__ for subclass in subclasses])))

    @classmethod
    def create_model(cls, name, network_info, configuration=None):
        Model = cls.get_model(name)
        return Model(network_info, configuration)

    @classmethod
    def get_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(subclass.get_subclasses())
        return all_subclasses

    @classmethod
    def available_wrappers(cls):
        available_classes = [cls] if cls.__model__ else []
        available_classes.extend(cls.get_subclasses())
        return [subclass.__model__ for subclass in available_classes if subclass.__model__]

    @classmethod
    def parameters(cls):
        '''Defines the description and type of configurable data parameters for the wrapper.
        See `types.py` to find available types of the data parameter. For each parameter
        the type, default value and description must be provided.
        The example of possible data parameter:
            'confidence_threshold': NumericalValue(
                default_value=0.5, description="Threshold value for detection box confidence"
            )
        The method must be implemented in each specific inherited wrapper.
        Returns:
            - the dictionary with defined wrapper data parameters
        '''
        parameters = {}
        return parameters

    def _load_config(self, config):
        '''Reads the configuration and creates data attributes
           by setting the wrapper parameters with values from configuration.
        Args:
            config (dict): the dictionary with keys to be set as data attributes
              and its values. The example of the config is the following:
              {
                  'confidence_threshold': 0.5,
                  'resize_type': 'fit_to_window',
              }
        Note:
            The config keys should be provided in `parameters` method for each wrapper,
            then the default value of the parameter will be updated. If some key presented
            in the config is not introduced in `parameters`, it will be omitted.
         Raises:
            WrapperError: if the configuration is incorrect
        '''
        if config is None: return
        parameters = self.parameters()
        for name, value in config.items():
            if name in parameters:
                errors = parameters[name].validate(value)
                if errors:
                    self.logger.error(f'Error with "{name}" parameter:')
                    for error in errors:
                        self.logger.error(f"\t{error}")
                    self.raise_error('Incorrect user configuration')
                value = parameters[name].get_value(value)
                self.__setattr__(name, value)
            else:
                self.logger.warning(f'The parameter "{name}" not found in {self.__model__} wrapper, will be omitted')

    def raise_error(self, message):
        '''Raises the WrapperError.
        Args:
            message (str): error message to be shown in the following format:
              "WrapperName: message"
        '''
        raise WrapperError(self.__model__, message)

    def preprocess(self, inputs):
        '''Interface for preprocess method.
        Args:
            inputs: raw input data, the data type is defined by wrapper
        Returns:
            - the preprocessed data which is submitted to the model for inference
                and has the following format:
                {
                    'input_layer_name_1': data_1,
                    'input_layer_name_2': data_2,
                    ...
                }
            - the input metadata, which might be used in `postprocess` method
        '''
        raise NotImplementedError

    def postprocess(self, outputs, meta):
        '''Interface for postprocess method.
        Args:
            outputs (dict): model raw output in the following format:
                {
                    'output_layer_name_1': raw_result_1,
                    'output_layer_name_2': raw_result_2,
                    ...
                }
            meta (dict): the input metadata obtained from `preprocess` method
        Returns:
            - postprocessed data in the format defined by wrapper
        '''
        raise NotImplementedError

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        '''Checks whether the number of model inputs/outputs is supported.
        Args:
            number_of_inputs (int, Tuple(int)): number of inputs supported by wrapper.
              Use -1 to omit the check
            number_of_outputs (int, Tuple(int)): number of outputs supported by wrapper.
              Use -1 to omit the check
        Raises:
            WrapperError: if the model has unsupported number of inputs/outputs
        '''
        if not isinstance(number_of_inputs, tuple):
            if len(self.inputs) != number_of_inputs and number_of_inputs != -1:
                self.raise_error("Expected {} input blob{}, but {} found: {}".format(
                    number_of_inputs, 's' if number_of_inputs !=1 else '',
                    len(self.inputs), ', '.join(self.inputs)
                ))
        else:
            if not len(self.inputs) in number_of_inputs:
                self.raise_error("Expected {} or {} input blobs, but {} found: {}".format(
                    ', '.join(str(n) for n in number_of_inputs[:-1]), int(number_of_inputs[-1]),
                    len(self.inputs), ', '.join(self.inputs)
                ))

        if not isinstance(number_of_outputs, tuple):
            if len(self.outputs) != number_of_outputs and number_of_outputs != -1:
                self.raise_error("Expected {} output blob{}, but {} found: {}".format(
                    number_of_outputs, 's' if number_of_outputs !=1 else '',
                    len(self.outputs), ', '.join(self.outputs)
                ))
        else:
            if not len(self.outputs) in number_of_outputs:
                self.raise_error("Expected {} or {} output blobs, but {} found: {}".format(
                    ', '.join(str(n) for n in number_of_outputs[:-1]), int(number_of_outputs[-1]),
                    len(self.outputs), ', '.join(self.outputs)
                ))

    def log_layers_info(self):
        '''Prints the shape, precision and layout for all model inputs/outputs.
        '''
        for name, metadata in self.inputs.items():
            self.logger.info('\tInput layer: {}, shape: {}, precision: {}, layout: {}'.format(
                name, metadata.shape, metadata.type, metadata.layout))
        for name, metadata in self.outputs.items():
            self.logger.info('\tOutput layer: {}, shape: {}, precision: {}, layout: {}'.format(
                name, metadata.shape, metadata.type, metadata.layout))
