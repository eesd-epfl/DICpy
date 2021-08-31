from DICpy.utils import _close
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import skimage.io as sio
from matplotlib.widgets import Button
import cv2


class Images:
    """
    This class contains the methods for reading and processing images used in digital image correlation (DIC).
    **Input:**
    **Attributes:**

    * **path_speckle** (`str`)
        Path to the speckle images.

    * **path_calibration** (`str`)
        Path to the calibration image.

    * **ref_image** (`ndarray`)
        Reference image.

    * **calibration_image** (`ndarray`)
        Calibration image.

    * **images** (`ndarray`)
        Gray images.

    * **images_normalized** (`ndarray`)
        Gray images normalized by 255.

    * **lx** (`int`)
        Number of columns of each image.

    * **ly** (`int`)
        Number of rows of each image.

    * **num_img** (`int`)
        Number of images.

    * **pixel_dim** (`float`)
        Size of each pixel in length dimension.

    **Methods:**
    """

    def __init__(self):

        self.path_speckle = None
        self.path_calibration = None
        self.ref_image = None
        self.calibration_image = None
        self.images = None
        self.images_normalized = None
        self.lx = None
        self.ly = None
        self.num_img = None
        self.pixel_dim = 1

    def read_speckle_images(self, path=None, extension=None, file_names=None, ref_id=0, verbose=False):

        """
        Read the speckle images.

        **Input:**
        * **path** (`str`)
            Path for the images used in the DIC analysis.

        * **extension** (`str`)
            The extension of the files located in `path`.

        * **file_names** (`str`)
            Define the files to be read in the sequence of your preference.

        * **ref_id** (`int`)
            Define a file to be used as reference, the default is zero, which means that the first image in the stack will
            be used as reference.

        * **verbose** (`bool`)
            Boolean varible to print some information on screen.

        **Output/Returns:**
        """

        if not isinstance(ref_id,int):
            raise TypeError('DICpy: ref_id must be an integer.')

        if ref_id < 0:
            raise ValueError('DICpy: ref_id is an integer larger than or equal to 0.')

        if path is None:
            path = os.getcwd()

        if extension is None:
            raise TypeError('DICpy: extension cannot be NoneType when using path.')

        self.path_speckle = path

        if file_names is None:
            file_names = [f for f in os.listdir(path) if f.endswith('.' + extension)]
            file_names = np.sort(file_names)

        if verbose:
            print('DICpy: reading speckle images.')

        images = []
        images_normalized = []
        for f in file_names:

            if verbose:
                print(f)

            #im = sio.imread(os.path.join(path, f), as_gray=True)
            im = cv2.imread(os.path.join(path, f), 0)

            images.append(im)
            #images.append(np.asarray(255 * im, dtype=np.uint8))
            images_normalized.append(im/255)

        self.images = images
        self.images_normalized = images_normalized
        (lx, ly) = np.shape(im)
        self.lx = lx
        self.ly = ly
        self.num_img = len(images)
        self.ref_image = images[ref_id]

    @staticmethod
    def _read_calibration_images(path=None, file_name=None, verbose=True):

        """
        Private method for reading the calibration image.

        **Input:**
        * **path** (`str`)
            Path for the calibration image used in the DIC analysis.

        * **file_names** (`str`)
            Name of the calibration image.

        * **verbose** (`bool`)
            Boolean varible to print some information on screen.

        **Output/Returns:**
        * **calibration_image** (`ndarray`)
            Calibration image.
        """

        if path is None:
            path = os.getcwd()

        if file_name is None:
            raise TypeError('DICpy: file_name cannot be NoneType.')

        if verbose:
            print('DICpy: reading the calibration image.')

        im = sio.imread(os.path.join(path, file_name), as_gray=True)
        calibration_image = 255 * im

        return calibration_image

    def calibration(self, ref_length=None, pixel_dim=None, path=None, file_name=None,
                    point_a=None, point_b=None, verbose=True):

        """
        Method for the analysis calibration.

        **Input:**
        * **ref_length** (`str`)
            Length used as reference.

        * **pixel_dim** (`float`)
            Size of each pixel in length dimension.

        * **path** (`str`)
            Path for the calibration image used in the DIC analysis.

        * **file_name** (`str`)
            Name of the calibration image.

        * **point_a** (`float`)
            First corner of the region of interest (ROI).

        * **point_b** (`float`)
            Corner opposed to point_b of the region of interest (ROI).

        * **verbose** (`bool`)
            Boolean variable to print some information on screen.

        **Output/Returns:**
        """

        if pixel_dim is None:

            if ref_length is None:
                raise TypeError('DICpy: ref_length cannot be NoneType.')

            self.path_calibration = path
            cal_img = self._read_calibration_images(path=path, file_name=file_name, verbose=verbose)
            self.calibration_image = cal_img
            (lx, ly) = np.shape(cal_img)
            maxl = max(lx, ly)

            global rcirc, ax, fig, coords, cid
            rcirc = maxl / 160

            if point_a is None or point_b is None:
                coords = []

                fig = plt.figure()
                ax = fig.add_subplot(111)
                cid = fig.canvas.mpl_connect('button_press_event', self._mouse_click)
                plt.imshow(cal_img, cmap="gray")
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
                #rect = patches.Rectangle(point_a, lx, ly, linewidth=1, edgecolor='None', facecolor='b', alpha=0.4)
                #ax.add_patch(rect)
                fig.canvas.draw()  # this line was missing earlier

                plt.imshow(cal_img, cmap="gray")
                plt.show(block=False)
                plt.close()


            point_a = (round(point_a[0]),round(point_a[1]))
            point_b = (round(point_b[0]),round(point_b[1]))

            #plt.close()
            fig = plt.figure()
            ax = fig.add_subplot(111)

            lx = (point_b[0] - point_a[0])
            ly = (point_b[1] - point_a[1])

            circle = plt.Circle(point_a, rcirc, color='red')
            ax.add_patch(circle)
            fig.canvas.draw()  # this line was missing earlier
            circle = plt.Circle(point_b, rcirc, color='red')
            ax.add_patch(circle)
            fig.canvas.draw()  # this line was missing earlier

            #rect = patches.Rectangle(point_a, lx, ly, linewidth=1, edgecolor='None', facecolor='b', alpha=0.4)
            #ax.add_patch(rect)
            ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], linewidth=2)
            fig.canvas.draw()
            plt.imshow(cal_img, cmap="gray")
            axclose = plt.axes([0.81, 0.05, 0.1, 0.075])
            bclose = Button(axclose, 'Next')
            bclose.on_clicked(_close)
            plt.show()

            self.pixel_dim = ref_length/np.sqrt(lx**2 + ly**2)

            if verbose:
                print("Points: ",point_a,point_b)
                print("mm/pixel: ",self.pixel_dim)

        else:

            if ref_length is not None:
                raise Warning('DICpy: ref_length not used when pixel_dim is provided.')

            self.pixel_dim = pixel_dim

            if verbose:
                print("mm/pixel: ",self.pixel_dim)

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



