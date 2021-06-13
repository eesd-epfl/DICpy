# PyCrack is distributed under the MIT license.
#
# Copyright (C) 2020  -- Katrin Beyer
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


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def get_template(im_source=None, center=None, side=None):
    # Crop image (im_source) to get a square patch (im_template) with center in
    # (center_row, center_col) with width (in pixels) given as an argument.
    if not isinstance(side, int):
        raise ValueError('pyCrack: side must be an integer!')

    else:
        center_row = center[0]
        center_col = center[1]
        row_0 = center_row - side
        col_0 = center_col - side

        row_1 = center_row + side
        col_1 = center_col + side

        id_row = np.arange(row_0, row_1 + 1)
        id_col = np.arange(col_0, col_1 + 1)

        im_template = im_source[np.ix_(id_row, id_col)]

    return im_template, id_row, id_col


def get_template_left(im_source=None, point=None, sidex=None, sidey=None):
    # Crop image (im_source) to get a square patch (im_template) with center in
    # (center_row, center_col) with width (in pixels) given as an argument.
    #if not isinstance(sidex, int):
    #    raise ValueError('pyCrack: side must be an integer!')

    #if not isinstance(sidey, int):
    #    raise ValueError('pyCrack: side must be an integer!')

    #else:
    p_row = point[0]
    p_col = point[1]
    row_0 = p_row
    col_0 = p_col

    row_1 = p_row + sidey
    col_1 = p_col + sidex

    id_row = np.arange(row_0, row_1 + 1)
    id_col = np.arange(col_0, col_1 + 1)

    im_template = im_source[np.ix_(id_row, id_col)]

    return im_template

def _close(event):
    plt.close()

def pad(model_script, model_object_name, sample, dict_kwargs=None):
    """
    Execute the python model in parallel
    :param sample: One sample point where the model has to be evaluated
    :return:
    """

    exec('from ' + model_script[:-3] + ' import ' + model_object_name)
    # if kwargs is not None:
    #     par_res = eval(model_object_name + '(sample, kwargs)')
    # else:
    if dict_kwargs is None:
        par_res = eval(model_object_name + '(sample)')
    else:
        par_res = eval(model_object_name + '(sample, **dict_kwargs)')
    # par_res = parallel_output
    # if self.model_is_class:
    #     par_res = parallel_output.qoi
    # else:
    #     par_res = parallel_output

    return par_res
