import numpy as np
from scipy import signal
from scipy.interpolate import RectBivariateSpline
import cv2


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


def gradient(img, k):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)

    return gx, gy


def derivatives(img1, img2=None):
    # Derivatives
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])

    img1 = img1 / 255.  # normalize pixels

    mode = 'same'
    bdry = 'symm'
    gx = signal.convolve2d(img1, kernel_x, boundary=bdry, mode=mode)
    gy = signal.convolve2d(img1, kernel_y, boundary=bdry, mode=mode)

    if img2 is not None:
        img2 = img2 / 255.  # normalize pixels
        gt = signal.convolve2d(img2, kernel_t, boundary=bdry, mode=mode) + \
             signal.convolve2d(img2, -kernel_t, boundary=bdry, mode=mode)

        return gx, gy, gt

    else:

        return gx, gy
