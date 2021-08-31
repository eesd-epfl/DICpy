from DICpy.utils import *
import numpy as np
import copy
from DICpy.math4dic import gradient, interpolate_template
from DICpy.DIC_2D._image_registration import ImageRegistration
from scipy.interpolate import RectBivariateSpline


class Oversampling(ImageRegistration):
    """
    DIC with subpixel resolution using oversampling.

    **Input:**
    * **mesh_obj** (`object`)
        Object of the RegularGrid class.

    **Attributes:**

    * **pixel_dim** (`float`)
        Size of each pixel in length dimension.

    * **mesh_obj** (`object`)
        Object of the RegularGrid class.

    * **u** (`ndarray`)
        Displacements in the x (columns) dimension at the center of each cell.

    * **v** (`ndarray`)
        Displacements in the y (rows) dimension at the center of each cell.

    **Methods:**
    """

    def __init__(self, mesh_obj=None, over_x=1, over_y=1):

        self.pixel_dim = mesh_obj.images_obj.pixel_dim
        self.mesh_obj = mesh_obj
        self.u = None
        self.v = None
        self.over_x = over_x
        self.over_y = over_y

        super().__init__(mesh_obj=mesh_obj)

    def _registration(self, img_0, img_1, psearch, ptem, lengths, windows, gaps):

        """
        Private method for estimating the displacements using image registration techniques.

        **Input:**
        * **img_0** (`ndarray`)
            Image in time t.

        * **img_1** (`ndarray`)
            Image in time t + dt.

        * **psearch** (`tuple`)
            Upper left corner of the searching area.

        * **ptem** (`tuple`)
            Point containing the upper left corner of the template.

        * **lengths** (`tuple`)
            Lengths in x and y of the searching area (length_x, length_y).

        * **windows** (`tuple`)
            Lengths in x and y of the template (window_x, window_y).

        * **gaps** (`tuple`)
            Gaps between template and searching area (gap_x, gap_y).

        **Output/Returns:**
        * **px** (`float`)
            Displacement in x (columns).

        * **px** (`float`)
            Displacement in y (rows).
        """

        l_x = lengths[0]
        l_y = lengths[1]
        window_x = windows[0]
        window_y = windows[1]

        # Image of the template.
        img_template = get_template_left(im_source=img_0, point=ptem, sidex=window_x, sidey=window_y)

        # Image of the searching area.
        img_search = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

        # No subpixel resolution required.
        px, py = self._template_match_sk(img_search, img_template, mlx=self.over_x, mly=self.over_y)
        px = int(px)
        py = int(py)

        return px, py