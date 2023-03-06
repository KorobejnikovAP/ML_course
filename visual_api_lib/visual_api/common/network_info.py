"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

class NetworkInfo:
    def __init__(self, inputs_info: dict, ouptuts_info: dict) -> None:
        self._inputs = inputs_info
        self._outputs = ouptuts_info


    @property
    def inputs_info(self) -> dict:
        return self._inputs

    @property
    def outputs_info(self) -> dict:
        return self._outputs
