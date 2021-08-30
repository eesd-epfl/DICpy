from DICpy.utils import *
from DICpy.utils import _correlation
import numpy as np
import cv2
from skimage.feature import match_template
import copy
from DICpy.math4dic import gradient, interpolate_template


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

            if sub_pixel == 'lucas_kanade':

                centers = []
                positions = []
                lx_list = []
                ly_list = []
                for i in range(self.mesh_obj.ny - 1):
                    for j in range(self.mesh_obj.nx - 1):
                        positions.append([i, j])
                        l_x = abs(xp[i + 1, j + 1] - xp[i, j])
                        l_y = abs(yp[i + 1, j + 1] - yp[i, j])
                        xc = (xp[i + 1, j + 1] + xp[i, j]) / 2
                        yc = (yp[i + 1, j + 1] + yp[i, j]) / 2
                        centers.append(np.array([np.float32(xc), np.float32(yc)]))
                        lx_list.append(l_x)
                        ly_list.append(l_y)

                l_x = np.max(lx_list)
                l_y = np.max(ly_list)
                centers = np.array(centers)
                positions = np.array(positions)

                lk_params = dict(winSize=(l_x, l_y), maxLevel=10,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

                final_positions, st, err = cv2.calcOpticalFlowPyrLK(img_0, img_1, centers, None, **lk_params)

                # print(final_positions - centers)
                u0 = np.zeros((self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))
                v0 = np.zeros((self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))
                for i in range(len(final_positions)):
                    ii = positions[i, 0]
                    jj = positions[i, 1]
                    u0[ii, jj] = final_positions[i, 0] - centers[i, 0]
                    v0[ii, jj] = final_positions[i, 1] - centers[i, 1]

                usum = usum + (u0 * self.pixel_dim)
                vsum = vsum + (v0 * self.pixel_dim)

                u.append(usum)
                v.append(vsum)

            else:

                # Loop over the elements.
                c = 0
                centers = []
                # print(self.mesh_obj.ny - 1, self.mesh_obj.nx - 1)
                for i in range(self.mesh_obj.ny - 1):
                    for j in range(self.mesh_obj.nx - 1):
                        # print(i,j)
                        l_x = abs(xp[i + 1, j + 1] - xp[i, j])  # searching area.
                        l_y = abs(yp[i + 1, j + 1] - yp[i, j])  # searching area.
                        xc = (xp[i + 1, j + 1] + xp[i, j]) / 2  # center searching area.
                        yc = (yp[i + 1, j + 1] + yp[i, j]) / 2  # center searching area.
                        centers.append((xc, yc))

                        gap_x = int(max(np.ceil(l_x / 3), 3))  # default gap: 1/3.
                        gap_y = int(max(np.ceil(l_y / 3), 3))  # default gap: 1/3.

                        xtem_0 = xp[i, j] + gap_x  # x coordinate of the top right (template).
                        ytem_0 = yp[i, j] + gap_y  # y coordinate of the top right (template).
                        xtem_1 = xp[i + 1, j + 1] - gap_x  # x coordinate of the bottom left (template).
                        ytem_1 = yp[i + 1, j + 1] - gap_y  # x coordinate of the bottom left (template).

                        window_x = abs(xtem_1 - xtem_0) + 1  # size x (columns) template.
                        window_y = abs(ytem_1 - ytem_0) + 1  # size y (rows) template.

                        ptem = (ytem_0, xtem_0)  # top right coordinate of the template for the matching processing.
                        psearch = (yp[i, j], xp[i, j])  # top right coordinate of the searching area for matching.

                        # Image of the template.
                        img_template = get_template_left(im_source=img_0, point=ptem, sidex=window_x, sidey=window_y)

                        # Image of the searching area.
                        img_search = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

                        if sub_pixel is None:
                            # No subpixel resolution required.
                            px, py = self.template_match_sk(img_search, img_template, mlx=1, mly=1)
                            px = int(px)
                            py = int(py)
                        elif sub_pixel == 'crude':
                            # Crude method: use oversampling. This method has an inferior computational performance.
                            px, py = self.template_match_sk(img_search, img_template, mlx=oversampling_x,
                                                            mly=oversampling_y)

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

                        elif sub_pixel == 'gradient':
                            # Simple gradient based method.

                            # Integer location.
                            px, py = self.template_match_sk(img_search, img_template, mlx=1, mly=1)
                            px = int(px)
                            py = int(py)

                            # Crop the searching areas for both images (sequential images).
                            ff = get_template_left(im_source=img_0, point=psearch, sidex=l_x, sidey=l_y)
                            gg = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

                            # subpixel estimation.
                            delta = self._grad_subpixel(f=ff, g=gg, gap_x=gap_x, gap_y=gap_y, p_corner=(py, px))
                            #px = delta[0]
                            #py = delta[1]

                            px = px + delta[0]
                            py = py + delta[1]

                        elif sub_pixel == 'affine':
                            # shape function with affine transformation.
                            px, py = self.template_match_sk(img_search, img_template, mlx=1, mly=1)
                            px = int(px)
                            py = int(py)

                            ff = get_template_left(im_source=img_0, point=psearch, sidex=l_x, sidey=l_y)
                            gg = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

                            xx = np.arange(xtem_0, xtem_1) + px
                            yy = np.arange(ytem_0, ytem_1) + py
                            XX, YY = np.meshgrid(xx, yy)
                            fg = ff - gg
                            gx, gy = derivatives(gg)

                            print(np.shape(XX), np.shape(ff), (l_y, l_x))
                            # print(np.sum(gx * fg))
                            # print(np.sum(gx * XX * fg))

                            # delta = self._grad_subpixel(f=ff, g=gg, gap_x=gap_x, gap_y=gap_y, p_corner=(py, px))

                            # px = px + delta[0]
                            # py = py + delta[1]

                        else:
                            raise NotImplementedError('DICpy: not recognized method.')

                        u0 = px - gap_x
                        v0 = py - gap_y

                        usum[i, j] = usum[i, j] + (u0 * self.pixel_dim)
                        vsum[i, j] = vsum[i, j] + (v0 * self.pixel_dim)

                        # u[k, i, j] = usum[i, j]
                        # v[k, i, j] = vsum[i, j]

                u.append(usum)
                v.append(vsum)

        self.u = np.array(u)
        self.v = np.array(v)

    @staticmethod
    def _coarse_fine(img_u=None, img_search=None, n=None, pval=None, window_x=None, window_y=None):

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
        x = np.arange(px, px + window_x)
        y = np.arange(py, py + window_y)

        img_0 = img_u

        r0 = -0.5
        r1 = 0.5
        c0 = -0.5
        c1 = 0.5

        yc0 = 0
        xc0 = 0
        delta = (0, 0)
        for k in range(n):

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
                    img_1 = interpolate_template(f=img_search, x=x, y=y, dx=xc, dy=yc)
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
    def _grad_subpixel(f=None, g=None, gap_x=None, gap_y=None, p_corner=None):

        """
        Private method from the paper: Application of an improved subpixel registration algorithm on digital speckle
        correlation measurement.
        By: Jun Zhang, Guanchang Jin, Shaopeng Ma, Libo Meng.

        This is a gradient method considering a shape function for pure translation only: x* = x + p.

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
        tol = 1e-3
        max_iter = 20  # maximum number of iterations.
        niter = 0
        while err > tol and niter <= max_iter:

            fg_crop = f_crop - g_crop

            a11 = np.sum(gx_crop ** 2)
            a22 = np.sum(gy_crop ** 2)
            a12 = np.sum(gx_crop * gy_crop)

            c1 = np.sum(fg_crop * gx_crop)
            c2 = np.sum(fg_crop * gy_crop)

            Ainv = np.linalg.inv(np.array([[a11, a12], [a12, a22]]))
            C = np.array([c1, c2])
            d_delta = Ainv @ C

            err = np.linalg.norm(d_delta)

            delta[0] = delta[0] + d_delta[0]
            delta[1] = delta[1] + d_delta[1]

            p_corner[0] = pc[0] + delta[0]
            p_corner[1] = pc[1] + delta[1]

            # Interpolate.
            # todo: adjust convention.
            x = np.linspace(p_corner[1], p_corner[1] + window_x, window_x)
            y = np.linspace(p_corner[0], p_corner[0] + window_y, window_y)
            g_crop = interpolate_template(f=g, x=x, y=y)
            gx_crop = interpolate_template(f=gx, x=x, y=y)
            gy_crop = interpolate_template(f=gy, x=x, y=y)

            niter += 1

        #print(niter-1, err)
        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #ax1.imshow(f_crop)
        #ax2.imshow(g_crop)
        #plt.show()
        #time.sleep(1000)
        #print(' ')
        return delta

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