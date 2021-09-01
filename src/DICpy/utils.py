import numpy as np
import matplotlib.pyplot as plt


def _mouse_click(event):
    """
    Private: function to capture the coordinates of a mouse click over an image.

    **Input:**
    * **event** (`object`)
        Event of mouse click.

    **Output/Returns:**
    * **coords** (`list`)
        Position of the mouse click in the image.
    """

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


def get_template(im_source=None, center=None, side=None):
    """
    Get a square template from image considering the a central coordinate.

    **Input:**
    * **im_source** (`ndarray`)
        Image source is the image from where the template is retrieved.

    * **center** (`list/tuple/ndarray`)
        Coordinates of the central pixel of the template.

    * **side** (`int`)
        How many pixels in each side of the central pixel will be used to construct the template.

    * **Output/Returns:**
    * **im_template** (`ndarray`)
        Template retrieved from `im_source`.

    * **id_row** (`ndarray`)
        Coordinates (rows) for locating the template in `im_source`.

    * **id_col** (`ndarray`)
        Coordinates (columns) for locating the template in `im_source`.
    """

    # Crop image (im_source) to get a square patch (im_template) with center in
    # (center_row, center_col) with width (in pixels) given as an argument.
    if not isinstance(side, int):
        raise ValueError('pyCrack: side must be an integer!')

    else:
        center_row = center[0]
        center_col = center[1]
        row_0 = center_row - side
        col_0 = center_col - side

        row_1 = center_row + side
        col_1 = center_col + side

        id_row = np.arange(row_0, row_1 + 1)
        id_col = np.arange(col_0, col_1 + 1)

        im_template = im_source[np.ix_(id_row, id_col)]

    return im_template, id_row, id_col


def get_template_left(im_source=None, point=None, sidex=None, sidey=None):
    """
    Get a rectangular template from image considering the upper left corner as a reference.

    **Input:**
    * **im_source** (`ndarray`)
        Image source is the image from where the template is retrieved.

    * **point** (`list/tuple/ndarray`)
        Coordinates of the upper left coorner pixel (reference pixel) of the template.

    * **sidex** (`int`)
        Template size in x direction (columns).

    * **sidey** (`int`)
        Template size in y direction (rows).

    * **Output/Returns:**
    * **im_template** (`ndarray`)
        Template retrieved from `im_source`.
    """

    p_row = point[0]
    p_col = point[1]
    row_0 = p_row
    col_0 = p_col

    row_1 = p_row + sidey - 1
    col_1 = p_col + sidex - 1

    id_row = np.arange(row_0, row_1 + 1)
    id_col = np.arange(col_0, col_1 + 1)

    im_template = im_source[np.ix_(id_row, id_col)]

    return im_template


def _close(event):
    """
    Private: function to close events such as mouse click.

    **Input:**
    * **event** (`object`)
        Event of mouse click.
    """

    plt.close()
