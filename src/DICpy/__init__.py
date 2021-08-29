"""
DICpy: digital imahe correlation with python
========================================
"""

import pkg_resources

import DICpy.dic2d
import DICpy.dic3d
import DICpy.utils

from DICpy.dic2d import *
from DICpy.dic3d import *
from DICpy.utils import *

try:
    __version__ = pkg_resources.get_distribution("DICpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("DICpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None
