from DICpy.utils import *
import numpy as np
import cv2
from skimage.feature import match_template


class ImageRegistration:
    """
    This class has the methods to perform the DIC analysis without considering the subpixel resolution.
    This is a parent class for children classes implementing methods with subpixel resolution.

    **Input:**
    * **mesh_obj** (`object`)
        Object of the RectangularMesh class.

    **Attributes:**

    * **pixel_dim** (`float`)
        Size of each pixel in length dimension.

    * **mesh_obj** (`object`)
        Object of the RectangularMesh class.

    * **u** (`ndarray`)
        Displacements in the x (columns) dimension at the center of each cell.

    * **v** (`ndarray`)
        Displacements in the y (rows) dimension at the center of each cell.

    **Methods:**
    """

    def __init__(self, mesh_obj=None):

        self.pixel_dim = mesh_obj.images_obj.pixel_dim
        self.mesh_obj = mesh_obj
        self.u = None
        self.v = None

    def run(self):

        """
        Method to perform the DIC analysis.

        **Input:**
        * **sub_pixel** (`str`)
            Method for the sub-pixel refining:
            - None: do not perform the sub-pixel refining.
            - 'Crude': crude method based on the image oversampling.
            - 'gradient': gradient based sub-pixel refining.
            - 'coarse_fine': method based on the sequential refining of the pixel domain.
            - 'lukas_kanade': method based on the Lukas-Kanade optical flow.

        * **oversampling_x** (`int`)
            Oversampling in the x dimension used when 'Crude' method is adopted, default is 1 (equal to 'crude').

        * **oversampling_y** (`int`)
            Oversampling in the y dimension used when 'Crude' method is adopted, default is 1 (equal to 'crude')..

        * **niter** (`int`)
            Number of iterations used when 'coarse_fine' method is adopted, default is 1.

        **Output/Returns:**
        """

        images = self.mesh_obj.images_obj.images
        num_img = self.mesh_obj.images_obj.num_img
        xp = self.mesh_obj.xp
        yp = self.mesh_obj.yp

        u = []
        v = []

        usum = np.zeros((self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))
        vsum = np.zeros((self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))

        # Loop over the images.
        for k in range(num_img - 1):

            img_0 = images[k]
            img_1 = images[k + 1]

            # Loop over the elements.
            c = 0
            centers = []
            # print(self.mesh_obj.ny - 1, self.mesh_obj.nx - 1)
            for i in range(self.mesh_obj.ny - 1):
                for j in range(self.mesh_obj.nx - 1):
                    # print(i, j)
                    l_x = abs(xp[i + 1, j + 1] - xp[i, j])  # searching area.
                    l_y = abs(yp[i + 1, j + 1] - yp[i, j])  # searching area.
                    xc = (xp[i + 1, j + 1] + xp[i, j]) / 2  # center searching area.
                    yc = (yp[i + 1, j + 1] + yp[i, j]) / 2  # center searching area.
                    centers.append((xc, yc))

                    gap_x = int(max(np.ceil(l_x / 3), 3))  # default gap: 1/3.
                    gap_y = int(max(np.ceil(l_y / 3), 3))  # default gap: 1/3

                    xtem_0 = xp[i, j] + gap_x  # x coordinate of the top right (template).
                    ytem_0 = yp[i, j] + gap_y  # y coordinate of the top right (template).
                    xtem_1 = xp[i + 1, j + 1] - gap_x  # x coordinate of the bottom left (template).
                    ytem_1 = yp[i + 1, j + 1] - gap_y  # x coordinate of the bottom left (template).

                    window_x = abs(xtem_1 - xtem_0) + 1  # size x (columns) template.
                    window_y = abs(ytem_1 - ytem_0) + 1  # size y (rows) template.

                    ptem = (ytem_0, xtem_0)  # top right coordinate of the template for the matching processing.
                    psearch = (yp[i, j], xp[i, j])  # top right coordinate of the searching area for matching.

                    lengths = (l_x, l_y)
                    windows = (window_x, window_y)
                    gaps = (gap_x, gap_y)
                    px, py = self._registration(img_0, img_1, psearch, ptem, lengths, windows, gaps)

                    u0 = px - gap_x
                    v0 = py - gap_y

                    usum[i, j] = usum[i, j] + (u0 * self.pixel_dim)
                    vsum[i, j] = vsum[i, j] + (v0 * self.pixel_dim)

                u.append(usum)
                v.append(vsum)

        self.u = np.array(u)
        self.v = np.array(v)

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
        px, py = self._template_match_sk(img_search, img_template, mlx=1, mly=1)
        px = int(px)
        py = int(py)

        return px, py


    @staticmethod
    def _template_match_sk(img_source, img_template, mlx=1, mly=1):

        """
        Method for template matching using correlation.

        **Input:**
        * **img_source** (`ndarray`)
            Image where the search will be performed.

        * **img_template** (`ndarray`)
            Template image.

        * **mlx** (`int`)
            Parameter used for oversampling in the x direction (columns).

        * **mly** (`int`)
            Parameter used for oversampling in the y direction (rows).

        **Output/Returns:**
        * **px** (`int`)
            Location in the x direction.

        * **py** (`int`)
            Location in the y direction.
        """

        # Image oversampling
        img_source = cv2.resize(img_source, None, fx=mlx, fy=mly, interpolation=cv2.INTER_CUBIC)
        img_template = cv2.resize(img_template, None, fx=mlx, fy=mly, interpolation=cv2.INTER_CUBIC)

        result = match_template(img_source, img_template)
        ij = np.unravel_index(np.argmax(result), result.shape)
        px, py = ij[::-1]

        px = px / mlx
        py = py / mly

        return px, py