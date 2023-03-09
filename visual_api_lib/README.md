# Visual API package

Visual API package is a set of wrapper classes for particular tasks and model architectures, simplifying data preprocess and postprocess as well as routine procedures (model loading, synchronous execution, etc...). An application feeds model class with input data, then the model returns postprocessed output data in user-friendly format.


## Package structure

The Visual API consists of 4 libraries:
* _launchers_ implements a common interface to allow model's wrappers usage with different executors. Now are available `tflite` and `onnx` executors.
* _models_ implements wrappers for DL models. These wrappers contain functions for pre- and postprocessing operations.
* _common_ implements common methods for reading inputs, resize images, etc. and common structures needed in other modules.
* _handlers_ implements handlers which manage the synchronous/asynchronous execution of models.

## Installing Visual API package

Use the following command to install Visual API from source:
```sh
python -m pip install -e visual_api_lib/
```
