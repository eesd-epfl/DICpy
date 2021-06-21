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
``DIC2Local`` is the module for ``DICpy`` to perform the local 2D digital image correlation (DIC).

This module contains the classes and methods.

The module currently contains the following classes:

* ``RectangularMesh``: Class for defining a rectangular mesh.

* ``Analysis``: Class to perform the DIC analysis.

* ``PostProcessing``: Class for visualization.

"""

from DICpy.Utils import *
from DICpy.Utils import _close, _correlation
import numpy as np
import scipy as sp
from scipy.interpolate import RectBivariateSpline
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
from skimage.feature import match_template
from sklearn.gaussian_process import GaussianProcessRegressor


########################################################################################################################
########################################################################################################################
#                                     Define a rectangular mesh                                                        #
########################################################################################################################
########################################################################################################################

class RectangularMesh:
    """
    This class contains the methods for creating a rectangular mesh used in the DIC analysis.
    **Input:**
    * **images_obj** (`object`)
        Object containing the speckle images, reference images, and calibration images, as well as calibration
        parameters.

    **Attributes:**

    * **images_obj** (`object`)
        Object containing the speckle images, reference images, and calibration images, as well as calibration
        parameters.

    * **stepx** (`ndarray`)
        Steps for the grid in the x direction (columns).

    * **stepy** (`ndarray`)
        Steps for the grid in the y direction (rows).

    * **centers** (`list`)
        Centers for each cell in the grid.

    * **wind** (`list`)
        Dimensions in both x and y dimensions for each cell.

    * **xp** (`ndarray`)
        Position of the upper right corner for each cell (grid in the x direction - columns).

    * **yp** (`ndarray`)
        Position of the upper right corner for each cell (grid in the y direction - rows).

    * **nx** (`int`)
        Number of nodes in the x direction (columns).

    * **ny** (`int`)
        Number of nodes in the y direction (rows).

    * **point_a** (`float`)
        First corner of the region of interest (ROI).

    * **point_b** (`float`)
        Corner opposed to point_b of the region of interest (ROI).

    **Methods:**
    """

    def __init__(self, images_obj=None):
        self.images_obj = images_obj
        self.stepx = None
        self.stepy = None
        self.centers = None
        self.wind = None
        self.xp = None
        self.yp = None
        self.nx = None
        self.ny = None
        self.elem = None
        self.point_a = None
        self.point_b = None

    def define_mesh(self, point_a=None, point_b=None, nx=2, ny=2):

        """
        Method to construct the rectangular mesh used in the DIC analysis.

        **Input:**
        * **point_a** (`float`)
            First corner of the region of interest (ROI).

        * **point_b** (`float`)
            Corner opposed to point_b of the region of interest (ROI).

        * **nx** (`int`)
            Number of nodes in the x direction (columns), default 2.

        * **ny** (`int`)
            Number of nodes in the y direction (rows), default 2.

        **Output/Returns:**
        """

        maxl = max(self.images_obj.lx, self.images_obj.ly)
        self.nx = nx
        self.ny = ny

        global rcirc, ax, fig, coords, cid
        rcirc = maxl / 160

        if point_a is None or point_b is None:

            coords = []

            fig = plt.figure()
            ax = fig.add_subplot(111)
            cid = fig.canvas.mpl_connect('button_press_event', self._mouse_click)
            plt.imshow(self.images_obj.images[0], cmap="gray")
            plt.show()

            point_a = coords[0]
            point_b = coords[1]

        else:

            fig = plt.figure()
            ax = fig.add_subplot(111)

            circle = plt.Circle(point_a, rcirc, color='red')
            ax.add_patch(circle)
            fig.canvas.draw()  # this line was missing earlier
            circle = plt.Circle(point_b, rcirc, color='red')
            ax.add_patch(circle)
            fig.canvas.draw()  # this line was missing earlier

            lx = (point_b[0] - point_a[0])
            ly = (point_b[1] - point_a[1])
            rect = patches.Rectangle(point_a, lx, ly, linewidth=1, edgecolor='None', facecolor='b', alpha=0.4)
            ax.add_patch(rect)
            fig.canvas.draw()  # this line was missing earlier

            plt.imshow(self.images_obj.images[0], cmap="gray")
            plt.show(block=False)
            plt.close()

        xa = point_a[0]
        xb = point_b[0]
        ya = point_a[1]
        yb = point_b[1]

        self.point_a = point_a
        self.point_b = point_b

        stepx = np.sort(np.linspace(xa, xb, nx))
        stepy = np.sort(np.linspace(ya, yb, ny))

        # Transform from continuous to discrete.
        stepx = np.array([round(x) for x in stepx])
        stepy = np.array([round(y) for y in stepy])

        if min(np.diff(stepx)) < 15:
            raise ValueError('DICpy: small subset, reduce nx.')

        if min(np.diff(stepy)) < 15:
            raise ValueError('DICpy: small subset, reduce ny.')

        self.stepx = stepx
        self.stepy = stepy

        xp, yp = np.meshgrid(stepx, stepy)

        centers = []
        wind = []

        for i in range(ny - 1):
            for j in range(nx - 1):
                l_x = abs(xp[i + 1, j + 1] - xp[i, j])
                l_y = abs(yp[i + 1, j + 1] - yp[i, j])
                xc = (xp[i + 1, j + 1] + xp[i, j]) / 2
                yc = (yp[i + 1, j + 1] + yp[i, j]) / 2
                centers.append((xc, yc))
                wind.append((l_x, l_y))

        self.xp = xp
        self.yp = yp
        self.centers = centers
        self.wind = wind

        # plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(nx):
            for j in range(ny):
                xx = stepx[i]
                yy = stepy[j]
                circle = plt.Circle((xx, yy), rcirc, color='red')
                ax.add_patch(circle)
                fig.canvas.draw()  # this line was missing earlier

        lx = (point_b[0] - point_a[0])
        ly = (point_b[1] - point_a[1])
        rect = patches.Rectangle(point_a, lx, ly, linewidth=1, edgecolor='None', facecolor='b', alpha=0.4)
        ax.add_patch(rect)
        fig.canvas.draw()
        plt.imshow(self.images_obj.images[0], cmap="gray")
        axclose = plt.axes([0.81, 0.05, 0.1, 0.075])
        bclose = Button(axclose, 'Next')
        # bclose.on_clicked(self._close)
        bclose.on_clicked(_close)
        plt.show()

    def _get_elements_Q4(self):

        """
        Private method to determine the elements of the mesh (under construction!).

        **Input:**

        **Output/Returns:**
        """

        xp = self.xp
        yp = self.yp
        nelem = (self.nx - 1) * (self.ny - 1)

        node_id = 0
        elem = []
        for i in range(self.ny - 1):
            for j in range(self.nx - 1):
                nodes = [(xp[i, j], yp[i, j]), (xp[i, j + 1], yp[i, j + 1]), (xp[i + 1, j], yp[i + 1, j]),
                         (xp[i + 1, j + 1], yp[i + 1, j + 1])]
                dic = {"nodes": nodes, "id": node_id}
                print(dic)
                elem.append(dic)
                node_id = node_id + 1

        self.elem = elem

    @staticmethod
    def _mouse_click(event):

        """
        Private method: mouse click to select the ROI.

        **Input:**
        * **event** (`event`)
            Click event.

        **Output/Returns:**
        """

        global x, y
        x, y = event.xdata, event.ydata

        if event.button:
            circle = plt.Circle((event.xdata, event.ydata), rcirc, color='red')
            ax.add_patch(circle)
            fig.canvas.draw()  # this line was missing earlier

        global coords
        coords.append((x, y))

        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)

            xa = coords[0][0]
            ya = coords[0][1]
            xb = coords[1][0]
            yb = coords[1][1]
            lx = (xb - xa)
            ly = (yb - ya)
            rect = patches.Rectangle((xa, ya), lx, ly, linewidth=1, edgecolor='None', facecolor='b', alpha=0.4)
            ax.add_patch(rect)
            fig.canvas.draw()  # this line was missing earlier
            plt.close()

        return coords


########################################################################################################################
########################################################################################################################
#                                               2D Analysis                                                            #
########################################################################################################################
########################################################################################################################

class Analysis:
    """
    This class contains the methods for the DIC analysis.
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

    def run(self, sub_pixel=None, oversampling_x=1, oversampling_y=1, niter=1):

        """
        Method to perform the DIC analysis.

        **Input:**
        * **sub_pixel** (`str`)
            Method for the sub-pixel refining:
            - None: do not perform the sub-pixel refining.
            - 'Crude': crude method based on the image oversampling.
            - 'gradient': gradient based sub-pixel refining.
            - 'coarse_fine': method based on the sequential refining of the pixel domain.

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

        u = np.zeros((num_img - 1, self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))
        v = np.zeros((num_img - 1, self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))

        usum = np.zeros((self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))
        vsum = np.zeros((self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))

        # Loop over the images.
        for k in range(num_img - 1):

            img_0 = images[k]
            img_1 = images[k + 1]

            # Loop over the elements.
            c = 0
            centers = []
            for i in range(self.mesh_obj.ny - 1):
                for j in range(self.mesh_obj.nx - 1):
                    l_x = abs(xp[i + 1, j + 1] - xp[i, j])
                    l_y = abs(yp[i + 1, j + 1] - yp[i, j])
                    xc = (xp[i + 1, j + 1] + xp[i, j]) / 2
                    yc = (yp[i + 1, j + 1] + yp[i, j]) / 2
                    centers.append((xc, yc))

                    gap_x = int(max(np.ceil(l_x / 3), 3))
                    gap_y = int(max(np.ceil(l_y / 3), 3))

                    xtem_0 = xp[i, j] + gap_x
                    ytem_0 = yp[i, j] + gap_y
                    xtem_1 = xp[i + 1, j + 1] - gap_x
                    ytem_1 = yp[i + 1, j + 1] - gap_y

                    window_x = abs(xtem_1 - xtem_0) + 1
                    window_y = abs(ytem_1 - ytem_0) + 1

                    ptem = (ytem_0, xtem_0)
                    psearch = (yp[i, j], xp[i, j])

                    img_template = get_template_left(im_source=img_0, point=ptem, sidex=window_x, sidey=window_y)
                    img_search = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

                    if sub_pixel is None:
                        px, py = self.template_match_sk(img_search, img_template, mlx=1, mly=1)
                        px = int(px)
                        py = int(py)
                    elif sub_pixel == 'crude':
                        px, py = self.template_match_sk(img_search, img_template, mlx=oversampling_x,
                                                        mly=oversampling_y)
                    elif sub_pixel == 'gradient':
                        px, py = self.template_match_sk(img_search, img_template, mlx=1, mly=1)
                        px = int(px)
                        py = int(py)
                        ff = get_template_left(im_source=img_0, point=psearch, sidex=l_x, sidey=l_y)
                        gg = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)
                        delta = self._grad_subpixel(f=ff, g=gg, gap_x=gap_x, gap_y=gap_y, p_corner=(py, px))

                        px = px + delta[0]
                        py = py + delta[1]
                    elif sub_pixel == 'coarse_fine':
                        px, py = self.template_match_sk(img_search, img_template, mlx=1, mly=1)
                        px = int(px)
                        py = int(py)
                        pval = (py, px)

                        img_u = get_template_left(im_source=img_0, point=ptem, sidex=window_x, sidey=window_y)

                        delta = self._coarse_fine(img_u=img_u, img_search=img_search, n=niter, pval=pval,
                                                  window_x=window_x, window_y=window_y)

                        px = px + delta[0]
                        py = py + delta[1]

                    else:
                        raise NotImplementedError('DICpy: not recognized method.')

                    u0 = px - gap_x
                    v0 = py - gap_y

                    usum[i, j] = usum[i, j] + (u0 * self.pixel_dim)
                    vsum[i, j] = vsum[i, j] + (v0 * self.pixel_dim)

                    u[k, i, j] = usum[i, j]
                    v[k, i, j] = vsum[i, j]

        self.u = u
        self.v = v

    def _coarse_fine(self, img_u=None, img_search=None, n=None, pval=None, window_x=None, window_y=None):

        """
        Private method from the paper: An advanced coarse-fine search approach for digital image correlation
        applications.
        By: Samo Simoncic, Melita Kompolsek, Primoz Podrzaj.

        **Input:**
        * **img_u** (`ndarray`)
            Reference image.

        * **img_search** (`ndarray`)
            Image of the searching area.

        * **n** (`int`)
            Number of iterations.

        * **pval** (`tuple`)
            Coorner: integer location of the template in the deformed image.

        * **window_x** (`int`)
            Length of the template in the x dimension.

        * **window_y** (`int`)
            Length of the template in the y dimension.

        **Output/Returns:**
        * **delta** (`tuple`)
            Sub-pixel increment.
        """

        px = pval[1]
        py = pval[0]
        ly, lx = np.shape(img_search)
        x = np.arange(px, px + window_x)
        y = np.arange(py, py + window_y)

        plt.close()
        plt.figure(111)
        plt.imshow(img_u)

        img_0 = img_u

        r0 = -0.5
        r1 = 0.5
        c0 = -0.5
        c1 = 0.5

        yc0 = 0
        xc0 = 0
        delta = (0, 0)
        for i in range(n):

            row = np.linspace(r0, r1, 3)
            col = np.linspace(c0, c1, 3)

            corr = []
            posi = []
            posif = []
            center = []
            for i in range(len(row) - 1):
                for j in range(len(col) - 1):
                    yc = (row[i] + row[i + 1]) / 2 + yc0 * 0
                    xc = (col[j] + col[j + 1]) / 2 + xc0 * 0
                    center.append((xc, yc))
                    img_1 = self.interpolate_template(f=img_search, x=x, y=y, dx=xc, dy=yc)
                    c = _correlation(img_0, img_1)
                    corr.append(c)
                    posi.append((row[i], col[j]))
                    posif.append((row[i + 1], col[j + 1]))

            c_max = corr.index(max(corr))
            p0 = posi[c_max]
            p1 = posif[c_max]
            delta = center[c_max]
            yc0 = delta[1]
            xc0 = delta[0]

            r0 = p0[0]
            r1 = p1[0]
            c0 = p0[1]
            c1 = p1[1]

        return delta

    @staticmethod
    def interpolate_template(f=None, x=None, y=None, dx=None, dy=None):

        """
        Method of interpolation.

        **Input:**
        * **f** (`ndarray`)
            Source image.

        * **x** (`ndarray`)
            Integer pixel position in the x direction.

        * **y** (`ndarray`)
            Integer pixel position in the y direction.

        * **dx** (`float`)
            Sub-pixel increment in the x direction.

        * **dy** (`float`)
            Sub-pixel increment in the y direction.

        **Output/Returns:**
        * **z** (`ndarray`)
            Interpolated image.
        """

        ly, lx = np.shape(f)
        # Regularly-spaced, coarse grid
        x0 = np.arange(0, lx)
        y0 = np.arange(0, ly)
        # X, Y = np.meshgrid(x, y)

        interp_spline = RectBivariateSpline(y0, x0, f)

        xt = x + dx
        yt = y + dy

        z = np.zeros((len(yt), len(xt)))
        for i in range(len(yt)):
            for j in range(len(xt)):
                z[i, j] = interp_spline(yt[i], xt[j])

        return z

    def _grad_subpixel(self, f=None, g=None, gap_x=None, gap_y=None, p_corner=None):

        """
        Private method from the paper: Application of an improved subpixel registration algorithm on digital speckle
        correlation measurement.
        By: Jun Zhang, Guanchang Jin, Shaopeng Ma, Libo Meng.

        **Input:**
        * **f** (`ndarray`)
            Reference image.

        * **g** (`ndarray`)
            Deformed image.

        * **gap_x** (`int`)
            Gap between searching area and the template in the x direction.

        * **gap_y** (`int`)
            Gap between searching area and the template in the y direction.

        * **p_corner** (`tuple`)
            Point containing the upper left corner of the searching area.

        **Output/Returns:**
        * **delta** (`tuple`)
            Sub-pixel displacement.
        """

        ly, lx = np.shape(f)
        xtem_0 = gap_x
        ytem_0 = gap_y
        xtem_1 = lx - gap_x
        ytem_1 = ly - gap_y

        # Local: in the searching area.
        ptem = (ytem_0, xtem_0)

        window_x = abs(xtem_1 - xtem_0) + 1
        window_y = abs(ytem_1 - ytem_0) + 1

        # using Sobel.
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=7)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=7)
        # gx, gy = self._gradient(g)

        gx = get_template_left(im_source=gx, point=p_corner, sidex=window_x, sidey=window_y)
        gy = get_template_left(im_source=gy, point=p_corner, sidex=window_x, sidey=window_y)
        f = get_template_left(im_source=f, point=ptem, sidex=window_x, sidey=window_y)
        g = get_template_left(im_source=g, point=p_corner, sidex=window_x, sidey=window_y)

        fg = f - g

        a11 = np.sum(gx ** 2)
        a22 = np.sum(gy ** 2)
        a12 = np.sum(gx * gy)

        c1 = np.sum(fg * gx)
        c2 = np.sum(fg * gy)

        Ainv = np.linalg.inv(np.array([[a11, a12], [a12, a22]]))
        C = np.array([c1, c2])
        delta = Ainv @ C

        return delta

    @staticmethod
    def _gradient(f):

        """
        Private method: gradient of image.

        **Input:**
        * **f** (`ndarray`)
            Source image.

        **Output/Returns:**
        * **grad_x** (`ndarray`)
            Derivative in the x direction.

        * **grad_y** (`ndarray`)
            Derivative in the y direction.
        """

        ly, lx = np.shape(f)

        grad_x = np.zeros(np.shape(f))
        grad_y = np.zeros(np.shape(f))
        for i in range(ly):
            for j in range(lx):
                dfy = 0
                dfx = 0
                if i <= 1:
                    dfy = float(f[i + 1, j]) - float(f[i, j])
                elif i >= ly - 2:
                    dfy = float(f[i, j]) - float(f[i - 1, j])
                else:
                    # dfy = (float(f[i + 1, j]) - float(f[i - 1, j]))/2
                    dfy = (-float(f[i + 2, j]) + 8 * float(f[i + 1, j]) - 8 * float(f[i - 1, j]) + float(
                        f[i - 2, j])) / 12

                if j <= 1:
                    dfx = float(f[i, j + 1]) - float(f[i, j])
                elif j >= lx - 2:
                    dfx = float(f[i, j]) - float(f[i, j - 1])
                else:
                    # dfx = (float(f[i, j + 1]) - float(f[i, j - 1])) / 2
                    dfx = (-float(f[i, j + 2]) + 8 * float(f[i, j + 1]) - 8 * float(f[i, j - 1]) + float(
                        f[i, j - 2])) / 12

                grad_x[i, j] = dfx
                grad_y[i, j] = dfy

        return grad_x, grad_y

    @staticmethod
    def template_match_sk(img_source, img_template, mlx=1, mly=1):

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


########################################################################################################################
########################################################################################################################
#                                             Visualization                                                            #
########################################################################################################################
########################################################################################################################

class PostProcessing:
    """
    This class contains the methods for visualizing the results of the DIC analysis.
    **Input:**
    * **analysis_obj** (`object`)
        Object of the Analysis class.

    **Attributes:**

    * **analysis_obj** (`float`)
        Object of the Analysis class.

    * **mesh_obj** (`object`)
        Object of the RectangularMesh class.

    * **strain11** (`ndarray`)
        Strain xx at the center of each cell.

    * **strain22** (`ndarray`)
        Strain yy at the center of each cell.

    * **strain12** (`ndarray`)
        Strain xy at the center of each cell.

    * **strain21** (`ndarray`)
        Strain yx (equal to Strain xy) at the center of each cell.

    **Methods:**
    """

    def __init__(self, analysis_obj=None):
        self.mesh_obj = analysis_obj.mesh_obj
        self.analysis_obj = analysis_obj
        self.strain11 = None
        self.strain22 = None
        self.strain12 = None
        self.strain21 = None

    def get_fields(self):

        """
        Method to estimate the strain fields.

        **Input:**

        **Output/Returns:**
        """

        # Derivative
        d_ker = np.matrix([-1., 0, 1.])
        u = self.analysis_obj.u
        v = self.analysis_obj.v
        pixel_dim = self.analysis_obj.pixel_dim

        centers = self.mesh_obj.centers
        zero_mat = np.zeros((np.shape(u)[1], np.shape(u)[2]))
        strain_matrix_11 = []
        strain_matrix_12 = []
        strain_matrix_21 = []
        strain_matrix_22 = []

        strain_matrix_11.append(zero_mat)
        strain_matrix_12.append(zero_mat)
        strain_matrix_21.append(zero_mat)
        strain_matrix_22.append(zero_mat)
        for k in range(np.shape(u)[0]):

            points = []
            dx = []
            dy = []
            c = 0

            for i in range(np.shape(u)[1]):
                for j in range(np.shape(u)[2]):
                    points.append([centers[c][0], centers[c][1]])
                    dx.append(u[k, i, j])
                    dy.append(v[k, i, j])
                    c = c + 1

            gpu = GaussianProcessRegressor(n_restarts_optimizer=10, normalize_y=True)
            gpu.fit(points, dx)

            gpv = GaussianProcessRegressor(n_restarts_optimizer=10, normalize_y=False)
            gpv.fit(points, dy)

            strain_11 = np.zeros((np.shape(u)[1], np.shape(u)[2]))
            strain_22 = np.zeros((np.shape(u)[1], np.shape(u)[2]))
            strain_12 = np.zeros((np.shape(u)[1], np.shape(u)[2]))
            strain_21 = np.zeros((np.shape(u)[1], np.shape(u)[2]))
            c = 0
            h = pixel_dim / 100

            for i in range(np.shape(u)[1]):
                for j in range(np.shape(u)[2]):
                    p0 = [[centers[c][0], centers[c][1]]]
                    p1 = [[centers[c][0] + h, centers[c][1]]]
                    pred0ux = gpu.predict(p0)[0]
                    pred1ux = gpu.predict(p1)[0]
                    pred0vx = gpv.predict(p0)[0]
                    pred1vx = gpv.predict(p1)[0]

                    p0 = [[centers[c][0], centers[c][1]]]
                    p1 = [[centers[c][0], centers[c][1] + h]]
                    pred0uy = gpu.predict(p0)[0]
                    pred1uy = gpu.predict(p1)[0]
                    pred0vy = gpv.predict(p0)[0]
                    pred1vy = gpv.predict(p1)[0]

                    d11 = (pred1ux - pred0ux) / h
                    d12 = (pred1uy - pred0uy) / h
                    d21 = (pred1vx - pred0vx) / h
                    d22 = (pred1vy - pred0vy) / h

                    strain_11[i, j] = d11 + 0.5 * (d11 ** 2 + d22 ** 2)
                    strain_22[i, j] = d22 + 0.5 * (d11 ** 2 + d22 ** 2)
                    strain_12[i, j] = 0.5 * (d12 + d21 + d11 * d12 + d21 * d22)
                    strain_21[i, j] = 0.5 * (d12 + d21 + d11 * d12 + d21 * d22)

                    c = c + 1

            strain_matrix_11.append(strain_11)
            strain_matrix_22.append(strain_22)
            strain_matrix_12.append(strain_12)
            strain_matrix_21.append(strain_21)

        self.strain11 = np.array(strain_matrix_11)
        self.strain22 = np.array(strain_matrix_22)
        self.strain12 = np.array(strain_matrix_12)
        self.strain21 = np.array(strain_matrix_21)

    def visualization(self, results="u", step=0, smooth=False):

        """
        Method to plot the results in terms of displacements and strains.

        **Input:**
        * **results** (`str`)
            Visulaize the results:
            -'u': displacement x.
            -'v': displacement y.
            -'e11': strain xx.
            -'e22': strain yy:
            -'e12': strain xy.
            -'e21': strain yx.
        * **step** (`int`)
            Define the result for a given loading step.

        * **smooth** (`bool`)
            Gaussian filtering.

        **Output/Returns:**
        """

        if step < 0:
            raise ValueError("pyCrack: step must be larger than or equal to 0.")

        if not isinstance(step, int):
            raise TypeError("pyCrack: step must be an integer.")

        point_a = self.mesh_obj.point_a
        point_b = self.mesh_obj.point_b
        images = self.mesh_obj.images_obj.images
        stepx = self.mesh_obj.stepx
        stepy = self.mesh_obj.stepy

        self.get_fields()

        u_ = self.analysis_obj.u
        v_ = self.analysis_obj.v

        if step == 0:
            u = np.zeros(np.shape(u_[0, :, :]))
            v = np.zeros(np.shape(u_[0, :, :]))
        else:
            u = u_[step - 1, :, :]
            v = v_[step - 1, :, :]

        e11 = self.strain11[step, :, :]
        e12 = self.strain11[step, :, :]
        e21 = self.strain11[step, :, :]
        e22 = self.strain22[step, :, :]

        if results == 'u':
            mask = u
        elif results == 'v':
            mask = v
        elif results == 'e11':
            mask = e11
        elif results == 'e12':
            mask = e12
        elif results == 'e21':
            mask = e21
        elif results == 'e22':
            mask = e22
        else:
            raise ValueError('DICpy: not valid option for results.')

        # mask = 255*np.ones((2000, 2000))
        # mask = self.strain11[step, :, :]
        # mask = v[0,:,:]
        img = images[step]

        x = np.arange(0, np.shape(img)[1])
        y = np.arange(0, np.shape(img)[0])
        X, Y = np.meshgrid(x, y)

        xm = np.arange(min(point_a[0], point_b[0]), max(point_a[0], point_b[0]))
        ym = np.arange(min(point_a[1], point_b[1]), max(point_a[1], point_b[1]))
        Xm, Ym = np.meshgrid(xm, ym)

        extent = np.min(x), np.max(x), np.min(y), np.max(y)
        extentm = np.min(xm), np.max(xm), np.shape(img)[0] - np.max(ym), np.shape(img)[0] - np.min(ym)

        if smooth:
            lx = stepx[1] - stepx[0]
            ly = stepy[1] - stepy[0]
            sigma = 0.005 * max(lx, ly)
            mask = sp.ndimage.gaussian_filter(mask, sigma=sigma)

        plt.close()
        fig = plt.figure(frameon=False)
        im1 = plt.imshow(img, cmap=plt.cm.gray, interpolation='bilinear', extent=extent)
        im2 = plt.imshow(mask, cmap=plt.cm.hsv, alpha=.7, interpolation='bilinear', extent=extentm)
        im3 = plt.plot([0, np.shape(img)[1]], [0, np.shape(img)[0]], '.')
        plt.xlim(0, np.shape(img)[1])
        plt.ylim(0, np.shape(img)[0])
        plt.colorbar(im2)
        plt.show()
