# DICpy is distributed under the MIT license.
#
# Copyright (C) 2021  -- Katrin Beyer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
``Simulation`` is the module for ``DICpy`` to generate synthetic data.

This module contains the classes and methods to simulate synthetic data.

The module currently contains the following class:

* ``Images``: Class for reading the gray images and to calibrate the analysis.

"""

from Utils import _close
import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import skimage as sk
import skimage.io as sio
from matplotlib.widgets import Button

########################################################################################################################
########################################################################################################################
#                                                    Images                                                            #
########################################################################################################################
########################################################################################################################

class SyntheticData:
    """
    This class contains the methods for generate synthetic data for the DIC analysis.
    **Input:**

    **Attributes:**

    * **path_speckle** (`str`)
        Path to the speckle images.

    * **path_calibration** (`str`)
        Path to the calibration image.

    * **ref_image** (`ndarray`)
        Reference image.

    * **calibration_image** (`ndarray`)
        Calibration image.

    * **images** (`ndarray`)
        Gray images.

    * **images_normalized** (`ndarray`)
        Gray images normalized by 255.

    * **lx** (`int`)
        Number of columns of each image.

    * **ly** (`int`)
        Number of rows of each image.

    * **num_img** (`int`)
        Number of images.

    * **pixel_dim** (`float`)
        Size of each pixel in length dimension.

    **Methods:**
    """

    def __init__(self, num_images=None, size=None, extension='png', random_state=None, save_images=True):

        self.random_state = random_state
        self.num_images = num_images
        self.size = size
        self.extension = extension
        self.images = None
        self.save_images = save_images

        if self.random_state is None:
            self.random_state = np.random.randint(0, high=99999, size=1, dtype=int)[0]

    def generate_images(self, num_speckles=100, sigma=2, displacement_x=None, displacement_y=None):

        n = self.num_images

        if len(displacement_x) != n:
            raise ValueError('DICpy: size of displacement_x must be equal to num_images.')

        if len(displacement_y) != n:
            raise ValueError('DICpy: size of displacement_y must be equal to num_images.')

        images = []
        img0 = self._gen_gaussian(dx=displacement_x[0], dy=displacement_y[0], num_speckles=num_speckles, sigma=sigma)
        images.append(img0)
        if self.save_images:
            cv2.imwrite(str(0) + '.' + self.extension, img0)

        print(img0)

        for i in np.arange(1, n):

            #img = self._gen_gaussian(dx=displacement_x[i], dy=displacement_y[i], num_speckles=num_speckles, sigma=sigma)
            t = (displacement_x[i], displacement_y[i])

            # Create Afine transform
            afine_tf = sk.transform.AffineTransform(matrix=None, scale=None, rotation=None, shear=0.01, translation=t)

            # Apply transform to image data
            img = sk.transform.warp(img0, inverse_map=afine_tf,mode='constant', cval=1)
            img=np.round(img * 255).astype(np.uint8)
            print(img)
            images.append(img)

            if self.save_images:
                cv2.imwrite(str(i)+'.'+self.extension, img)

        self.images = images

    def _gen_gaussian(self, dx=None, dy=None, num_speckles=100, sigma=2):

        """
        Read the speckle images.

        **Input:**
        * **dx** (`float`)
            Displacement in the x direction.

        * **dy** (`float`)
            Displacement in the y direction.

        * **num_speckles** (`int`)
            Number of speackles.

        * **ref_id** (`int`)
            Define a file to be used as reference, the default is zero, which means that the first image in the stack will
            be used as reference.

        * **verbose** (`bool`)
            Boolean varible to print some information on screen.

        **Output/Returns:**
        """

        nx, ny = self.size
        np.random.seed(self.random_state)

        xk = np.random.randint(0, high=nx, size=num_speckles, dtype=int)
        yk = np.random.randint(0, high=ny, size=num_speckles, dtype=int)
        vald = np.zeros((ny, nx, num_speckles))
        img = np.zeros((ny, nx))

        for k in range(num_speckles):

            I0 = np.random.randint(0, high=256, size=1, dtype=int)[0]

            for i in range(ny):
                for j in range(nx):
                    vald[i, j, k] = I0 * np.exp(-((i - yk[k] - dy) ** 2 + (j - xk[k] - dx) ** 2) / (sigma ** 2))

            img = img + np.floor(vald[:, :, k])

        img = np.floor(255 * (1 - img / np.max(img)))
        img = img.astype(np.uint8)

        return img



