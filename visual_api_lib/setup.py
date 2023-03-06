"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from pathlib import Path
from setuptools import setup, find_packages


SETUP_DIR = Path(__file__).resolve().parent

with open(SETUP_DIR / 'requirements.txt') as f:
    required = f.read().splitlines()

packages = find_packages(str(SETUP_DIR))
package_dir = {'api': str(SETUP_DIR / 'api')}

setup(
    name='visual_api',
    version='8.8.8',
    author='Aleksey Korobeynikov',
    license='NO_LICENSE',
    description='API: model and launchers wrappers to create AI demos',
    python_requires = ">=3.7",
    packages=packages,
    package_dir=package_dir,
    install_requires=required
)
