# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import plot_arrow
from utils import plot_edge
from utils import setup_figure
from utils import numpy_equal


def example_data():
    va = np.array([1.6, .4])
    vb = np.array([.6, 1.5])
    gt = np.array([0.35862069, 0.89655172])
    return va, vb, gt


def plot_result(va, vb, proj):
    assert isinstance(proj, np.ndarray)
    plot_arrow(va, color="blue")
    plot_arrow(vb, color="green")
    plot_arrow(proj, color="red")
    plot_edge(va, proj, "--r")

    # according to https://stackoverflow.com/a/57360868 , we have to manually legend the annotation objects
    custom_lines = [mpl.lines.Line2D([0], [0], color="blue", lw=2),
                    mpl.lines.Line2D([0], [0], color="green", lw=2),
                    mpl.lines.Line2D([0], [0], color="red", lw=2)]
    plt.gca().legend(custom_lines, ['va', 'vb', 'proj'])

    plt.show()


def vector_projection(va, vb):
    """
    Args:
        va: The first vector, shape: (K)
        vb: The second vector, shape: (K)

    Returns:
        The projection vector of va to vb, shape: (K)
    """
    # TODOs:
    # - normalize the vector vb to compute vb_norm; vb_norm.shape == (3,)
    # - compute proj_length = <va, vb_norm>; proj_length.shape == (1,)
    # - return <vb_norm, proj_length>; shape == (3,)
    # HINTS: only rely on np.sqrt, np.dot, elementwise mult/div (*,/)

    # 1
    vb_norm = vb / np.sqrt(np.dot(vb, vb))

    # 2
    proj_length = np.dot(va, vb_norm)

    # 3
    return vb_norm * proj_length

    # return np.array([0,1,0])


def main():
    va, vb, gt = example_data()
    setup_figure(scale=2, symmetric=False)
    res = vector_projection(va, vb)
    plot_result(va, vb, res)


def test_simplest():
    va = np.array([1, 1, 1])
    vb = np.array([0, 0, 1])
    gt = np.array([0, 0, 1])
    res = vector_projection(va, vb)
    assert numpy_equal(res, gt)


def test_example():
    va, vb, gt = example_data()
    res = vector_projection(va, vb)
    assert numpy_equal(res, gt)


def test_golden():
    npzfile = np.load("task1.npz")
    vas = npzfile["vas"]
    vbs = npzfile["vbs"]
    res = npzfile["res"]
    assert numpy_equal(vector_projection(vas, vbs), res)


if __name__ == "__main__":
    main()
