"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""
import tempfile
import yaml
from visual_api.common import read_yaml


def test_read_yaml():
    data = {
        "name": "Aleksey",
        "surname": "Korobeynikov",
        "age": 23
    }
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'w') as f:
        f.write(yaml.dump(data))
    assert data == read_yaml(tmp.name)
