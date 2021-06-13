# DICpy is distributed under the MIT license.
#
# Copyright (C) 2020  -- EESD
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
``DIC`` is the module for ``pyopt`` to perform ...

This module contains the classes and methods to perform ...

The module currently contains the following classes:

* ``ClassA``: Class for...

* ``ClassB``: Class for...

"""

from DICpy.Utils import *
from DICpy.Utils import _close
import numpy as np
import scipy as sp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import skimage.io as sio
from matplotlib.widgets import Button
import cv2
from skimage.feature import match_template
from sklearn.gaussian_process import GaussianProcessRegressor


class Images:

    def __init__(self, path=None, extension=None, as_gray=True):

        self.path = path
        self.extension = extension
        self.as_gray = as_gray
        self.images = None
        self.images_normalized = None
        self.lx = None
        self.ly = None
        self.num_img = None

        if path is not None:
            if extension is None:
                raise TypeError('PyCrack: extension cannot be NoneType when using path.')

    def read_images(self,verbose=True):

        fnames = [f for f in os.listdir(self.path) if f.endswith('.' + self.extension)]
        fnames = np.sort(fnames)

        images = []
        images_normalized = []
        for filenames in fnames:

            if verbose:
                print(filenames)

            im = sio.imread(os.path.join(self.path, filenames), as_gray=self.as_gray)

            images.append(255 * im)
            images_normalized.append(im)
            # images.append(255*np.invert(np.array(im)))
            # images_normalized.append(np.invert(np.array(im)))

        self.images = images
        self.images_normalized = images_normalized
        (lx, ly) = np.shape(im)
        self.lx = lx
        self.ly = ly
        self.num_img = len(images)


class RectangularMesh:

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
                centers.append((xc,yc))
                wind.append((l_x,l_y))

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

    def get_elements_Q4(self):

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

    #    @staticmethod
    #    def _close(event):
    #        plt.close()

    @staticmethod
    def _mouse_click(event):

        global x, y
        x, y = event.xdata, event.ydata

        if event.button:
            circle = plt.Circle((event.xdata, event.ydata), rcirc, color='red')
            ax.add_patch(circle)
            fig.canvas.draw()  # this line was missing earlier

        global coords
        coords.append((x, y))
        print(x, y)
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


class Analysis:

    def __init__(self, pixel_dim=1, mesh_obj=None):

        self.pixel_dim = pixel_dim
        self.mesh_obj = mesh_obj
        self.u = None
        self.v = None

    def correlation_local(self, gap=1):

        if not isinstance(gap, int):
            raise TypeError("pyCrack: gap must be an integer.")

        if gap < 1:
            raise ValueError("pyCrack: gap must be an integer larger than 1.")

        # window = 2*half_window + 1
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

                    xtem_0 = xp[i, j] + gap
                    ytem_0 = yp[i, j] + gap
                    xtem_1 = xp[i + 1, j + 1] - gap
                    ytem_1 = yp[i + 1, j + 1] - gap

                    window_x = abs(xtem_1 - xtem_0)
                    window_y = abs(ytem_1 - ytem_0)

                    ptem = (ytem_0, xtem_0)
                    psearch = (yp[i, j], xp[i, j])

                    img_template = get_template_left(im_source=img_0, point=ptem, sidex=window_x, sidey=window_y)
                    img_search = get_template_left(im_source=img_1, point=psearch, sidex=l_x, sidey=l_y)

                    px, py = self.template_match_sk(img_search, img_template, mlx=1, mly=1)

                    #px = px + window_x / 2
                    #u0 = xp[i, j] + px - 0.5
                    #v0 = yp[i, j] + py - 0.5
                    u0 = px - gap
                    v0 = py - gap

                    usum[i, j] = usum[i, j] + (u0 * self.pixel_dim)
                    vsum[i, j] = vsum[i, j] + (v0 * self.pixel_dim)
                    #usum[i, j] = usum[i, j] + (u0 * self.pixel_dim - xc * self.pixel_dim)
                    #vsum[i, j] = vsum[i, j] + (v0 * self.pixel_dim - yc * self.pixel_dim)

                    u[k, i, j] = usum[i, j]
                    v[k, i, j] = vsum[i, j]

        self.u = u
        self.v = v

        #plt.close()
        #plt.subplot(211)
        #plt.imshow(img_template, cmap="gray")
        #plt.subplot(212)
        #plt.imshow(img_search, cmap="gray")
        #plt.show()

        return u, v

    from skimage.feature import match_template

    @staticmethod
    def template_match_sk(img_source, img_template, mlx=1, mly=1):

        # Image oversampling
        img_source = cv2.resize(img_source, None, fx=mlx, fy=mly, interpolation=cv2.INTER_CUBIC)
        img_template = cv2.resize(img_template, None, fx=mlx, fy=mly, interpolation=cv2.INTER_CUBIC)

        result = match_template(img_source, img_template)
        ij = np.unravel_index(np.argmax(result), result.shape)
        px, py = ij[::-1]

        px = int(px / mlx)
        py = int(py / mly)

        return px, py

    @staticmethod
    def template_match(img_template, img_search, method='cv2.TM_CCOEFF_NORMED', mlx=1, mly=1):

        # print(img_slave)
        method = 'cv2.TM_CCOEFF_NORMED'

        # img_master = 255 * img_master
        img_search = img_search.astype(np.uint8)
        # img_slave = 255 * img_slave
        img_template = img_template.astype(np.uint8)
        # Apply image oversampling
        img_search = cv2.resize(img_search, None, fx=mlx, fy=mly, interpolation=cv2.INTER_CUBIC)
        img_slave = cv2.resize(img_template, None, fx=mlx, fy=mly, interpolation=cv2.INTER_CUBIC)

        res = cv2.matchTemplate(img_template, img_search, eval(method))

        w, h = img_search.shape[::-1]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Control if the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum value
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Retrieve center coordinates
        px = (top_left[0] + bottom_right[0]) / (2.0 * mlx)
        py = (top_left[1] + bottom_right[1]) / (2.0 * mly)

        return px, py, max_val


class PostProcessing:

    def __init__(self, analysis_obj=None):
        self.mesh_obj = analysis_obj.mesh_obj
        self.analysis_obj = analysis_obj
        self.strain11 = None
        self.strain22 = None
        self.strain12 = None
        self.strain21 = None

    def get_fields(self):

        # Derivative
        #num_img = self.mesh_obj.images_obj.num_img
        d_ker = np.matrix([-1., 0, 1.])
        u = self.analysis_obj.u
        v = self.analysis_obj.v
        pixel_dim = self.analysis_obj.pixel_dim

        centers = self.mesh_obj.centers
        zero_mat = np.zeros((np.shape(u)[1],np.shape(u)[2]))
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
                    dx.append(u[k,i,j])
                    dy.append(v[k, i, j])
                    c = c + 1

            gpu = GaussianProcessRegressor(n_restarts_optimizer=10, normalize_y=True)
            gpu.fit(points, dx)

            gpv = GaussianProcessRegressor(n_restarts_optimizer=10, normalize_y=False)
            gpv.fit(points, dy)

            strain_11 = np.zeros((np.shape(u)[1],np.shape(u)[2]))
            strain_22 = np.zeros((np.shape(u)[1], np.shape(u)[2]))
            strain_12 = np.zeros((np.shape(u)[1], np.shape(u)[2]))
            strain_21 = np.zeros((np.shape(u)[1], np.shape(u)[2]))
            c = 0
            h = pixel_dim / 100

            for i in range(np.shape(u)[1]):
                for j in range(np.shape(u)[2]):

                    p0 = [[centers[c][0], centers[c][1]]]
                    p1 = [[centers[c][0]+h, centers[c][1]]]
                    pred0ux = gpu.predict(p0)[0]
                    pred1ux = gpu.predict(p1)[0]
                    pred0vx = gpv.predict(p0)[0]
                    pred1vx = gpv.predict(p1)[0]

                    p0 = [[centers[c][0], centers[c][1]]]
                    p1 = [[centers[c][0], centers[c][1]+h]]
                    pred0uy = gpu.predict(p0)[0]
                    pred1uy = gpu.predict(p1)[0]
                    pred0vy = gpv.predict(p0)[0]
                    pred1vy = gpv.predict(p1)[0]

                    d11 = (pred1ux - pred0ux)/h
                    d12 = (pred1uy - pred0uy)/h
                    d21 = (pred1vx - pred0vx)/h
                    d22 = (pred1vy - pred0vy)/h

                    # self.strain_xx = strain_xx + .5 * (np.power(strain_xx, 2) + np.power(strain_yy, 2))
                    # self.strain_yy = strain_yy + .5 * (np.power(strain_xx, 2) + np.power(strain_yy, 2))
                    # self.strain_xy = .5 * (strain_xy + strain_yx + strain_xx * strain_xy + strain_yx * strain_yy)
                    strain_11[i,j] = d11 + 0.5 * (d11**2 + d22**2)
                    strain_22[i,j] = d22 + 0.5 * (d11**2 + d22**2)
                    strain_12[i,j] = 0.5 * (d12 + d21 + d11 * d12 + d21 * d22)
                    strain_21[i,j] = 0.5 * (d12 + d21 + d11 * d12 + d21 * d22)

                    c = c + 1

            strain_matrix_11.append(strain_11)
            strain_matrix_22.append(strain_22)
            strain_matrix_12.append(strain_12)
            strain_matrix_21.append(strain_21)

        self.strain11 = np.array(strain_matrix_11)
        self.strain22 = np.array(strain_matrix_22)
        self.strain12 = np.array(strain_matrix_12)
        self.strain21 = np.array(strain_matrix_21)

    def visualization(self, step=0, smooth=False):

        if step < 0:
            raise ValueError("pyCrack: step must be larger than or equal to 0.")

        if not isinstance(step,int):
            raise TypeError("pyCrack: step must be an integer.")

        point_a = self.mesh_obj.point_a
        point_b = self.mesh_obj.point_b
        images = self.mesh_obj.images_obj.images
        stepx = self.mesh_obj.stepx
        stepy = self.mesh_obj.stepy

        self.get_fields()

        # mask = 255*np.ones((2000, 2000))
        mask = self.strain11[step, :, :]
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
            sigma = 0.005*max(lx,ly)
            mask = sp.ndimage.gaussian_filter(mask, sigma=sigma)

        plt.close()
        fig = plt.figure(frameon=False)
        im1 = plt.imshow(img, cmap=plt.cm.gray, interpolation='bilinear', extent=extent)
        im2 = plt.imshow(mask, cmap=plt.cm.hot, alpha=.7, interpolation='bilinear', extent=extentm)
        im3 = plt.plot([0, np.shape(img)[1]], [0, np.shape(img)[0]], '.')
        plt.xlim(0, np.shape(img)[1])
        plt.ylim(0, np.shape(img)[0])

        plt.show()
