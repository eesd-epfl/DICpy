import numpy as np
import scipy as sp
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt


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
        # d_ker = np.matrix([-1., 0, 1.])
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
            raise ValueError("DICpy: `step` must be larger than or equal to 0.")

        if step > len(self.analysis_obj.u):
            raise ValueError("DICpy: `step` cannot be larger than the number of steps in the analysis.")

        if not isinstance(step, int):
            raise TypeError("DICpy: step must be an integer.")

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
        e12 = self.strain12[step, :, :]
        e21 = self.strain21[step, :, :]
        e22 = self.strain22[step, :, :]

        if results == 'u':
            mask = u
        elif results == 'v':
            mask = v
        elif results == 'abs':
            mask = np.sqrt(v ** 2 + u ** 2)
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
