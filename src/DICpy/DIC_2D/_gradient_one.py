from DICpy.utils import *
import numpy as np
import copy
from DICpy.math4dic import gradient, interpolate_template
from DICpy.DIC_2D._image_registration import ImageRegistration
import scipy.spatial.distance as sd
from scipy.interpolate import RectBivariateSpline


class GradientOne(ImageRegistration):
    """
    DIC with subpixel resolution using gradients. This class implement a method using zeroth order shape functions, and
    It is a child class of `ImageRegistration`.

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

    def __init__(self, mesh_obj=None, max_iter=20, tol=1e-3):

        self.pixel_dim = mesh_obj.images_obj.pixel_dim
        self.mesh_obj = mesh_obj
        self.u = None
        self.v = None
        self.max_iter = max_iter
        self.tol = tol

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
        gap_x = gaps[0]
        gap_y = gaps[1]

        # Image of the template.
        img_template = get_template_left(im_source=img_0, point=ptem, sidex=window_x, sidey=window_y)

        # Image of the searching area.
        img_search = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

        # Subpixel resolution.
        # Integer location.
        px, py = self._template_match_sk(img_search, img_template, mlx=1, mly=1)
        px = int(px)
        py = int(py)

        # Crop the searching areas for both images (sequential images).
        ff = get_template_left(im_source=img_0, point=psearch, sidex=l_x, sidey=l_y)
        gg = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

        # subpixel estimation.
        delta = self._grad_subpixel(f=ff, g=gg, gap_x=gap_x, gap_y=gap_y, p_corner=(py, px))

        px = px + delta[0]
        py = py + delta[1]

        return px, py

    def _grad_subpixel(self, f=None, g=None, gap_x=None, gap_y=None, p_corner=None):
        """
        Private method from the paper: Application of an improved subpixel registration algorithm on digital speckle
        correlation measurement.
        By: Jun Zhang, Guanchang Jin, Shaopeng Ma, Libo Meng.

        This is a gradient method considering a shape function for affine transformation
        (including distortion in the deformed image): x* = a x + b.

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
        gx, gy = gradient(g, k=7)

        # Interpolants.
        dim = np.shape(g)
        x0 = np.arange(0, dim[1])
        y0 = np.arange(0, dim[0])
        interp_g = RectBivariateSpline(y0, x0, g)
        interp_gx = RectBivariateSpline(y0, x0, gx)
        interp_gy = RectBivariateSpline(y0, x0, gy)

        # Crop the searching areas in f and g, for the correlated points.
        gx_crop = get_template_left(im_source=gx, point=p_corner, sidex=window_x, sidey=window_y)
        gy_crop = get_template_left(im_source=gy, point=p_corner, sidex=window_x, sidey=window_y)
        f_crop = get_template_left(im_source=f, point=ptem, sidex=window_x, sidey=window_y)
        g_crop = get_template_left(im_source=g, point=p_corner, sidex=window_x, sidey=window_y)

        # write p_corner as an array.
        p_corner = np.array(p_corner, dtype=float)
        pc = copy.copy(p_corner)

        # Initiate the update of the parameters of the shape function.
        delta = np.zeros(np.shape(p_corner))
        err = 1000
        tol = self.tol
        max_iter = self.max_iter  # maximum number of iterations.
        niter = 0
        while err > tol and niter <= max_iter:

            xx = np.linspace(p_corner[1], p_corner[1] + window_x, window_x)
            yy = np.linspace(p_corner[0], p_corner[0] + window_y, window_y)
            XX, YY = np.meshgrid(xx, yy)

            fg_crop = f_crop - g_crop

            a11 = np.sum(gx_crop * gx_crop)
            a12 = np.sum(gx_crop * gy_crop)
            a13 = np.sum(gx_crop * gx_crop * XX)
            a14 = np.sum(gx_crop * gx_crop * YY)
            a15 = np.sum(gx_crop * gy_crop * XX)
            a16 = np.sum(gx_crop * gy_crop * YY)

            a22 = np.sum(gy_crop * gy_crop)
            a23 = np.sum(gx_crop * gy_crop * XX)
            a24 = np.sum(gx_crop * gy_crop * YY)
            a25 = np.sum(gy_crop * gy_crop * XX)
            a26 = np.sum(gy_crop * gy_crop * YY)

            a33 = np.sum(gx_crop * gx_crop * XX * XX)
            a34 = np.sum(gx_crop * gx_crop * XX * YY)
            a35 = np.sum(gx_crop * gy_crop * XX * XX)
            a36 = np.sum(gx_crop * gx_crop * XX * YY)

            a44 = np.sum(gx_crop * gx_crop * YY * YY)
            a45 = np.sum(gx_crop * gy_crop * XX * YY)
            a46 = np.sum(gx_crop * gy_crop * YY * YY)

            a55 = np.sum(gy_crop * gy_crop * XX * XX)
            a56 = np.sum(gy_crop * gy_crop * XX * YY)

            a66 = np.sum(gy_crop * gy_crop * YY * YY)

            #A_vec = [a11, a12, a13, a14, a15, a16, a22, a23, a24, a25, a26, a33, a34, a35, a36, a44, a45, a46, a55, a56,
            # a66]

            A_vec = [a12, a13, a14, a15, a16, a23, a24, a25, a26, a34, a35, a36, a45, a46, a56]
            A = sd.squareform(np.array(A_vec))
            A[0, 0] = a11
            A[1, 1] = a22
            A[2, 2] = a33
            A[3, 3] = a44
            A[4, 4] = a55
            A[5, 5] = a66

            A_inv = np.linalg.inv(A)

            C = np.zeros(6)
            C[0] = np.sum(fg_crop * gx_crop)
            C[1] = np.sum(fg_crop * gy_crop)
            C[1] = np.sum(fg_crop * gx_crop * XX)
            C[1] = np.sum(fg_crop * gx_crop * YY)
            C[1] = np.sum(fg_crop * gy_crop * XX)
            C[1] = np.sum(fg_crop * gy_crop * YY)

            d_delta = A_inv @ C

            err = np.linalg.norm(d_delta)

            delta[0] = delta[0] + d_delta[0]
            delta[1] = delta[1] + d_delta[1]

            p_corner[0] = pc[0] + delta[0]
            p_corner[1] = pc[1] + delta[1]

            # Interpolate.
            # todo: adjust convention.
            x = np.linspace(p_corner[1], p_corner[1] + window_x, window_x)
            y = np.linspace(p_corner[0], p_corner[0] + window_y, window_y)
            # g_crop = interpolate_template(f=g, x=x, y=y)
            # gx_crop = interpolate_template(f=gx, x=x, y=y)
            # gy_crop = interpolate_template(f=gy, x=x, y=y)
            g_crop = interpolate_template(f=interp_g, x=x, y=y, dim=dim)
            gx_crop = interpolate_template(f=interp_gx, x=x, y=y, dim=dim)
            gy_crop = interpolate_template(f=interp_gy, x=x, y=y, dim=dim)

            niter += 1

        # print(niter-1, err)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(f_crop)
        # ax2.imshow(g_crop)
        # plt.show()
        # time.sleep(1000)
        #print(' ')
        return delta

