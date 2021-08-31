import numpy as np
import cv2
from DICpy.DIC_2D._image_registration import ImageRegistration


class LucasKanade(ImageRegistration):

    """
    DIC with subpixel resolution using the Lucas-Kanade algorithm implemented in OpenCV-Python, and It is a child class
    of `ImageRegistration`.

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

    def __init__(self, mesh_obj=None):

        self.pixel_dim = mesh_obj.images_obj.pixel_dim
        self.mesh_obj = mesh_obj
        self.u = None
        self.v = None

        super().__init__(mesh_obj=mesh_obj)

    def run(self):

        """
        Method to run the DIC analysis.

        """

        # Get images and the number of images in the object images.
        images = self.mesh_obj.images_obj.images
        num_img = self.mesh_obj.images_obj.num_img

        # Get the nodes of the grid in the mesh object.
        xp = self.mesh_obj.xp
        yp = self.mesh_obj.yp

        # initialize the lists receiving the horizontal (u) and vertical (v) displacements.
        u = []
        v = []

        # Initialize the accumulator of displacements.
        usum = np.zeros((self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))
        vsum = np.zeros((self.mesh_obj.ny - 1, self.mesh_obj.nx - 1))

        # Loop over the images.
        for k in range(num_img - 1):

            img_0 = images[k]  # image in the time t.
            img_1 = images[k + 1]  # image in the time t + dt.

            centers = []  # list for the centers of the elements in the grid.
            positions = []  # list for the position for the pixel positions of the upper left nodes of each element.
            lx_list = []  # length in the direction x (columns) of each element.
            ly_list = []  # length in the direction y (columns) of each element.

            # Loop over the upper left nodes of each element.
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

            # todo: make the lenths the same.
            l_x = np.max(lx_list)
            l_y = np.max(ly_list)
            centers = np.array(centers)
            positions = np.array(positions)

            # Parameters for the Lucas-Kanade method using opencv-python.
            lk_params = dict(winSize=(l_x, l_y), maxLevel=10,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Get the final positions for the optical flow using Lucas-Kanade.
            final_positions, st, err = cv2.calcOpticalFlowPyrLK(img_0, img_1, centers, None, **lk_params)

            # The next steps are the update of the displacement for the template.
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

        # Instantiate the displacements.
        self.u = np.array(u)
        self.v = np.array(v)