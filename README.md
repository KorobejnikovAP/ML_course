## Demo Application using Python Visual API

This repo contains:

1. **Visual API package** is a set of wrapper classes for particular tasks and model architectures, simplifying data preprocess and postprocess as well as routine procedures (model loading, synchronous execution, etc...). An application feeds model class with input data, then the model returns postprocessed output data in user-friendly format. More information see [this](./visual_api_lib/README.md).

2. **demo_application.py** - an application that solves one of the following computer vision tasks:
    * classification
    * object_detection
    * segmentation

### Usage

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
