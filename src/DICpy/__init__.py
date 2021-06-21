"""
DICpy: digital imahe correlation with python
========================================
"""

import pkg_resources

import DICpy.DIC2Local
import DICpy.PreProcessing
import DICpy.Utils
from DICpy.DIC2Local import *
from DICpy.PreProcessing import *
from DICpy.Utils import *

try:
    __version__ = pkg_resources.get_distribution("DICpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("DICpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None
