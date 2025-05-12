"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from ._config import BaseConfig
from .workspace import GLOBAL_CONFIG, create, register
from .yaml_config import YAMLConfig
from .yaml_utils import *
