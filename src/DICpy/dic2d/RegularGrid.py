from DICpy.utils import _close
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class RegularGrid:
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

    def define_mesh(self, point_a=None, point_b=None, nx=2, ny=2, show_grid=False):

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
            
        * **show_grid** (`int`)
            This functionality is only available when `point_a` and `point_b` are provided.
            When `noplot` is True, the grid is not plotted for express calculations.

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

            if show_grid:
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
        if show_grid:
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



