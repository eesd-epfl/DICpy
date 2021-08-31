"""
DICpy: digital imahe correlation with python
========================================
"""

import pkg_resources

import DICpy.DIC_2D
import DICpy.utils

from DICpy.DIC_2D import *
from DICpy.utils import *

try:
    __version__ = pkg_resources.get_distribution("DICpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("DICpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None
