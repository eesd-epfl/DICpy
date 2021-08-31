import numpy as np
from scipy import signal
from scipy.interpolate import RectBivariateSpline
import cv2


def norm_xcorr(f, g):
    """
    Normalized cross correlation.

    **Input:**
    * **f** (`ndarray`)
        Image.

    * **g** (`ndarray`)
        Image.

    **Output/Returns:**
    * **c** (`float`)
        Correlation.
    """
    mean_f = np.mean(f)
    mean_g = np.mean(g)

    sum_fg = np.sum((f - mean_f) * (g - mean_g))
    sum_f2 = np.sum((f - mean_f) ** 2)
    sum_g2 = np.sum((g - mean_g) ** 2)

    c = sum_fg / np.sqrt(sum_f2 * sum_g2)

    return c


def interpolate_template2(f=None, x=None, y=None, dx=0, dy=0):
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


def interpolate_template(f=None, x=None, y=None, dx=0, dy=0, dim=None):
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

    if not isinstance(f, RectBivariateSpline):
        dim = np.shape(f)
        ly = dim[0]
        lx = dim[1]

        x0 = np.arange(0, lx)
        y0 = np.arange(0, ly)

        interp_spline = RectBivariateSpline(y0, x0, f)

    else:
        ly = dim[0]
        lx = dim[1]
        interp_spline = f

    xt = x + dx
    yt = y + dy

    z = np.zeros((len(yt), len(xt)))
    for i in range(len(yt)):
        for j in range(len(xt)):
            z[i, j] = interp_spline(yt[i], xt[j])

    return z

def gradient(img, k):
    """
    Estimage the gradient of images using Sobel filters from OpenCV-Python.

    **Input:**
    * **img** (`ndarray`)
        Image.

    * **k** (`ndarray`)
        Order of approximation.

    **Output/Returns:**
    * **gx** (`ndarray`)
        Derivative in x (columns).

    * **gy** (`ndarray`)
        Derivative in y (rows).
    """
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)

    return gx, gy


def derivatives(img1, img2=None):
    """
    First order derivatives in x, y, and time (for two images).

    **Input:**
    * **img1** (`ndarray`)
        Image in time t.

    * **img2** (`ndarray`)
        Image in time t + dt.

    **Output/Returns:**
    * **gx** (`ndarray`)
        Derivative in x (columns) for img1.

    * **gy** (`ndarray`)
        Derivative in y (rows) for img1.

    * **gt** (`ndarray`)
        Derivative in time for img1 (optiional).
    """
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
