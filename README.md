# Applications based on Python Visual API

This repo contains:

1. **Visual API package** is a set of wrapper classes for particular tasks and model architectures, simplifying data preprocess and postprocess as well as routine procedures (model loading, synchronous execution, etc...). An application feeds model class with input data, then the model returns postprocessed output data in user-friendly format. More information see [this](./visual_api_lib/README.md).

2. **<task_name>.py** - an application that solves one of the following computer vision tasks:
    * classification
    * object_detection
    * segmentation

## Usage

To run `demo_application` next the following steps:

1. Install visual api package:

```
  python -m venv ml_env
  source ml_env/bin/activate
  python -m pip install -e visual_api_lib/
```

2. Run demo with needed options:

```
  python demo_application.py [options]
```

> **NOTE**: For example was created `classification_demo` to solve classification task.

## Options of demo

Demos have 2 required options:

```
  -c CONFIG, --config CONFIG
                        Required. Path to model config file
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a single image, a folder of images, video
                        file or camera id.
```

Config file contains info about model weights and some needed info for preprocessing and postprocessing. For example minimum config is:

```
name: yolo-v8
model_path: yolov8m.xml
mean_values: [y, y, y]
scale_values: [x, x, x]
```

Such config developer should provide for all models in bank of models.

## Tasks and demos

|demos\launchers      | pytorch | onnx | openvino | tflite  |
|---------------------|---------|------|----------|---------|
|classification demo  |  Yes    | Yes  |  Yes     |  Yes    |
|detection_demo       |  Yes    | Yes  |  Yes     |  Yes    |
