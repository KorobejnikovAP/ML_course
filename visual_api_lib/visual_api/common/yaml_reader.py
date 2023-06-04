"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from pathlib import Path
from typing import List, Union

import copy
import sys
import yaml


'''
Read YAML file to python dict
'''
def read_yaml(file: Union[str, Path]) -> dict:
    with open(file, "r") as content:
        return yaml.safe_load(content)

'''
Add paths to sys.path (usefull to find models code)
'''
class prepend_to_path:
    def __init__(self, paths):
        self._preprended_paths = paths
        self._original_path = None

    def __enter__(self):
        self._original_path = copy.deepcopy(sys.path)
        if self._preprended_paths is not None:
            sys.path = self._preprended_paths + sys.path

    def __exit__(self, type, value, traceback):
        if self._original_path is not None:
            sys.path = self._original_path


'''
Create model_configuration dict from YAML file
'''
def read_model_config(file: Union[str, Path]) -> dict:
    model_config = read_yaml(file)
    # update path to model
    model_config["model_path"] = str(Path(file).parent.resolve() / Path(model_config["model_path"]))
    # update path to module for PyTorch
    if "module_name" in model_config:
        model_config["module_name"] = str(Path(file).parent.resolve() / Path(model_config["module_name"]))
    return model_config

