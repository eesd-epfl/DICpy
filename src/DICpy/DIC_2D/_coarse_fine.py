from DICpy.utils import *
import numpy as np
from DICpy.math4dic import interpolate_template, norm_xcorr
from DICpy.DIC_2D._image_registration import ImageRegistration
from scipy.interpolate import RectBivariateSpline


class CoarseFine(ImageRegistration):
    """
    DIC with subpixel resolution using an coarse-fine search approach as presented in the following paper:

     "An advanced coarse-fine search approach for digital image correlation applications."
    (By: Samo Simoncic, Melita Kompolsek, Primoz Podrzaj)

    Further, this class is a child class of ``ImageRegistration``, and it inherit the method `run` and override
    `_registration`.

    **Input:**
    * **mesh_obj** (`object`)
        Object of ``RegularGrid``.

    **Attributes:**

    * **pixel_dim** (`float`)
        Constant to convert pixel into a physical quantity (e.g., mm).

    * **mesh_obj** (`object`)
        Object of the ``RegularGrid``.

    * **u** (`ndarray`)
        Displacements in the x (columns) dimension at the center of each cell.

    * **v** (`ndarray`)
        Displacements in the y (rows) dimension at the center of each cell.

    * **niter** (`int`)
    Number of iterations in the coarse-fine algorithm.

    **Methods:**
    """

    def __init__(self, mesh_obj=None, niter=1):

        self.pixel_dim = mesh_obj.images_obj.pixel_dim
        self.mesh_obj = mesh_obj
        self.u = None
        self.v = None
        self.niter = niter

        super().__init__(mesh_obj=mesh_obj)

    def _registration(self, img_0, img_1, psearch, ptem, lengths, windows, gaps):
        """
        Private method for estimating the displacements using the template matching technique with subpixel resolution.
        This method overrides the one in the parent class.

        **Input:**
        * **img_0** (`ndarray`)
            Image in time t.

        * **img_1** (`ndarray`)
            Image in time t + dt.

        * **psearch** (`tuple`)
            Upper left corner of the searching area.

        * **ptem** (`tuple`)
            Upper left corner of the template.

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

        # Template.
        img_template = get_template_left(im_source=img_0, point=ptem, sidex=window_x, sidey=window_y)

        # Searching area.
        img_search = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

        # Get positions with subpixel resolution.
        px, py = self._template_match_sk(img_search, img_template, mlx=1, mly=1)
        px = int(px)
        py = int(py)
        pval = (py, px)

        img_u = get_template_left(im_source=img_0, point=ptem, sidex=window_x, sidey=window_y)

        delta = self._coarse_fine(img_u=img_u, img_search=img_search, n=self.niter, pval=pval,
                                  window_x=window_x, window_y=window_y)

        px = px + delta[0]
        py = py + delta[1]

        return px, py

    @staticmethod
    def _coarse_fine(img_u=None, img_search=None, n=None, pval=None, window_x=None, window_y=None):
        """
        Private method implementing the coarse-fine search.

        **Input:**
        * **img_u** (`ndarray`)
            Reference image.

        * **img_search** (`ndarray`)
            Image of the searching area.

        * **n** (`int`)
            Number of iterations.

        * **pval** (`tuple`)
            Corner: integer location of the template in the deformed image.

        * **window_x** (`int`)
            Length of the template in the x dimension.

        * **window_y** (`int`)
            Length of the template in the y dimension.

        **Output/Returns:**
        * **delta** (`tuple`)
            Sub-pixel increment.
        """

        # Identify the coordinates of the template in the deformed image.
        px = pval[1]
        py = pval[0]
        x = np.arange(px, px + window_x)
        y = np.arange(py, py + window_y)
        img_0 = img_u

        # Get the interpolator.
        dim = np.shape(img_search)
        x0 = np.arange(0, dim[1])
        y0 = np.arange(0, dim[0])
        interp_img = RectBivariateSpline(y0, x0, img_search)

        # Positions for rows and columns in the sub-grid.
        r0 = -0.5
        r1 = 0.5
        c0 = -0.5
        c1 = 0.5

        # Start the iterative process.
        yc0 = 0
        xc0 = 0
        delta = (0, 0)
        for k in range(n):

            # Determine the sub-grid for each iteration.
            row = np.linspace(r0, r1, 3)
            col = np.linspace(c0, c1, 3)

            corr = []
            posi = []
            posif = []
            center = []

            # Find the normalized cross correlation for each cell in the sub-grid.
            for i in range(len(row) - 1):
                for j in range(len(col) - 1):

                    # Coordinate of the center of the cells in the sub-grid.
                    yc = (row[i] + row[i + 1]) / 2 + yc0 * 0
                    xc = (col[j] + col[j + 1]) / 2 + xc0 * 0
                    center.append((xc, yc))

                    # Get the interpolated template.
                    img_1 = interpolate_template(f=interp_img, x=x, y=y, dx=xc, dy=yc, dim=dim)

                    # Compute the normalized cross correlation between the reference and the interpolated template.
                    c = norm_xcorr(img_0, img_1)

                    corr.append(c)
                    posi.append((row[i], col[j]))
                    posif.append((row[i + 1], col[j + 1]))

            # Get the position in the sub-grid with maximum cross correlation and define a smaller sub-grid.
            c_max = corr.index(max(corr))
            p0 = posi[c_max]
            p1 = posif[c_max]
            delta = center[c_max]
            yc0 = delta[1]
            xc0 = delta[0]

            # Update the sub-grid and repeat the process within this sub-grid.
            r0 = p0[0]
            r1 = p1[0]
            c0 = p0[1]
            c1 = p1[1]

        return delta