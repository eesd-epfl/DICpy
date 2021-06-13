"""
DICpy: digital imahe correlation with python
========================================
"""

import pkg_resources

import DICpy.DIC
import DICpy.Utils
from DICpy.DIC import *
from DICpy.Utils import *

try:
    __version__ = pkg_resources.get_distribution("DICpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("DICpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None
