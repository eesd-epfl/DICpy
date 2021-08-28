import numpy as np
import matplotlib.pyplot as plt


def _correlation(f, g):
    m_f = np.mean(f)
    m_g = np.mean(g)

    c = np.sqrt(np.sum((f - m_f) ** 2) * np.sum((g - m_g) ** 2))
    corr = np.sum((f - m_f) * (g - m_g)) / c

    return corr


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


def get_template(im_source=None, center=None, side=None):
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
    # Crop image (im_source) to get a square patch (im_template) with center in
    # (center_row, center_col) with width (in pixels) given as an argument.
    # if not isinstance(sidex, int):
    #    raise ValueError('pyCrack: side must be an integer!')

    # if not isinstance(sidey, int):
    #    raise ValueError('pyCrack: side must be an integer!')

    # else:
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
    plt.close()


def pad(model_script, model_object_name, sample, dict_kwargs=None):
    """
    Execute the python model in parallel
    :param sample: One sample point where the model has to be evaluated
    :return:
    """

    exec('from ' + model_script[:-3] + ' import ' + model_object_name)
    # if kwargs is not None:
    #     par_res = eval(model_object_name + '(sample, kwargs)')
    # else:
    if dict_kwargs is None:
        par_res = eval(model_object_name + '(sample)')
    else:
        par_res = eval(model_object_name + '(sample, **dict_kwargs)')
    # par_res = parallel_output
    # if self.model_is_class:
    #     par_res = parallel_output.qoi
    # else:
    #     par_res = parallel_output

    return par_res
