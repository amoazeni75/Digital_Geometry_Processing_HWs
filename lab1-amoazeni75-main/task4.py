# %%
import einops
import numpy as np
import matplotlib.pyplot as plt
from utils import numpy_equal
import matplotlib as mpl
from task2 import batch_projection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import plot_arrow


# %%

def compute_line_sdf(point, normal, res=128):
    """
    Args:
        point: the point coordinate, shape: (2)
        normal: the normal direction, shape: (2)
        res: the grid resolution

    Returns:
        grid_coords: grid coordinates, shape: (res, res, 2).
        grid_sdf: SDF values, shape: (res, res)
    """
    # TODOs:
    # - get the grid coordinates using np.meshgrid and np.stack; shape (res, res, 2)
    # - normalize the normal vector
    # - compute the grid_sdf using vectorized operations; shape (res, res)
    # 1: get the grid coordinates using np.meshgrid and np.stack; shape (res, res, 2)
    x = np.stack(np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res)), axis=-1)

    # 2: normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # 3: compute the grid_sdf using vectorized operations; shape (res, res)
    grid_sdf = np.dot(x - point, normal)

    return x, grid_sdf


def compute_circle_sdf(point, normal, res=128):
    """
    Args:
        point: the point coordinate, shape: (2)
        normal: the radius direction whose magnitude is the radius, shape: (2)
        res: the grid resolution

    Returns:
        grid_coords: grid coordinates, shape: (res, res, 2).
        grid_sdf: SDF values, shape: (res, res)
    """
    # TODOs:
    # - get the grid coordinates using np.meshgrid and np.stack; shape (res, res, 2)
    # - notice that you should not normalize the "normal" vector here.
    # - compute the grid_sdf using vectorized operations; shape (res, res)

    # 1: get the grid coordinates using np.meshgrid and np.stack; shape (res, res, 2)
    x = np.stack(np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res)), axis=-1)

    # 2: compute the grid_sdf using vectorized operations; shape (res, res)
    grid_sdf = np.linalg.norm(x - point, axis=-1) - np.linalg.norm(normal)

    return x, grid_sdf


# %%
def example_line_data():
    npzfile = np.load("task4_line.npz")
    point, normal, grid_coords, grid_sdf = npzfile["point"], npzfile["normal"], npzfile["grid_coords"], npzfile[
        "grid_sdf"]
    return point, normal, grid_coords, grid_sdf


def example_circle_data():
    npzfile = np.load("task4_circle.npz")
    point, normal, grid_coords, grid_sdf = npzfile["point"], npzfile["normal"], npzfile["grid_coords"], npzfile[
        "grid_sdf"]
    return point, normal, grid_coords, grid_sdf


def test_example():
    point, normal, gc_gt, gs_gt = example_line_data()
    grid_coords, grid_sdf = compute_line_sdf(point, normal)
    assert (grid_coords.shape == gc_gt.shape) and (grid_sdf.shape == gs_gt.shape)
    assert numpy_equal(grid_coords, gc_gt)
    assert numpy_equal(grid_sdf, gs_gt)

    point, normal, gc_gt, gs_gt = example_circle_data()
    grid_coords, grid_sdf = compute_circle_sdf(point, normal)
    assert (grid_coords.shape == gc_gt.shape) and (grid_sdf.shape == gs_gt.shape)
    assert numpy_equal(grid_coords, gc_gt)
    assert numpy_equal(grid_sdf, gs_gt)


def main():
    point, normal, gt1, gt2 = example_line_data()
    coords, sdf = compute_line_sdf(point, normal)
    plot_sdf(coords, sdf, point, normal)
    plt.show()

    point, normal, _, _ = example_circle_data()
    coords, sdf = compute_circle_sdf(point, normal)
    plot_sdf(coords, sdf, point, normal)
    plt.show()


def plot_sdf(grid_coords, grid_sdf, point, normal, cmap=plt.cm.seismic, colorbar=True,
             norm=None, ticks=None, plotrange=np.array([[0., 1.], [0., 1.]]), normal_scale=1, im_origin="lower",
             valuerange=None, resolution=(400., 400.)):
    X, Y, Z = grid_coords[..., 0], grid_coords[..., 1], grid_sdf
    dpi = resolution[0] / 4.
    fig, ax = plt.subplots(figsize=(resolution[0] / dpi, resolution[1] / dpi), dpi=dpi, tight_layout=True)

    thresh = np.maximum(np.abs(Z.min()), Z.max())
    if ticks is None:
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
    field = ax.imshow(Z, origin=im_origin,
                      extent=[*plotrange.reshape(-1)],
                      cmap=cmap, norm=mpl.colors.TwoSlopeNorm(vmin=-thresh, vcenter=0., vmax=thresh))

    contour = ax.contour(X, Y, Z, origin=im_origin, colors=('gray',), linestyles=('--',), linewidths=(1,))
    contour = ax.contour(X, Y, Z, levels=[0.], origin=im_origin, colors=('black',), linestyles=('-',), linewidths=(1,))
    # plot_arrow(end=point+normal/6., begin=point, zorder=100)
    # Compared to annotate, the quiver function is more reliable. For example, the arrow does not disappear when the end point is out of range.
    ax.quiver(*point, *(normal),
              scale=normal_scale,
              scale_units='xy',
              angles='xy', linestyle=('--',), linewidth=(12,),
              color=['k'], zorder=100)

    if colorbar == True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(field, cax=cax,
                            format=mpl.ticker.FuncFormatter(lambda x, pos: '%.1f' % x))

    ax.set_aspect('equal')
    return fig, ax


# %%
if __name__ == "__main__":
    main()
# %%
