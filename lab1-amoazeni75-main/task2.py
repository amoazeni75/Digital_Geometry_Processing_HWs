import numpy as np
import matplotlib.pyplot as plt
from utils import plot_arrow
from utils import plot_edge
from utils import setup_figure
from utils import numpy_equal


def example_data():
    va = np.array([1, 1])
    VB = np.array(
        [[0.945164, 0.326595], [0.863288, 0.504712], [0.748235, 0.663433], [0.604429, 0.796659], [0.437394, 0.89927],
         [0.253551, 0.967322]])
    VA_proj = np.array(
        [[1.202022, 0.415351], [1.180977, 0.690446], [1.05626, 0.936548], [0.846858, 1.11619], [0.584649, 1.202022],
         [0.309554, 1.180977]])
    return va, VB, VA_proj


def plot_result(va, VB, VA_proj):
    assert (type(VA_proj) is np.ndarray)

    setup_figure(scale=1.5, symmetric=False)
    plot_arrow(va, color="blue")
    for vb in VB:
        plot_arrow(end=vb, color="green", zorder=10)
    for va_proj in VA_proj:
        plot_arrow(end=va_proj, color="red")

    VA = np.broadcast_to(va, VB.shape)
    for va, va_proj in zip(VA, VA_proj):
        plot_edge(va, va_proj, '--r', linewidth=.5)


def batch_projection(va, VB):
    """
    Args:
        va: vector to project, shape: (K)
        VB: batch of vectors, shape: (N,K)

    Returns:
        The projections of va to VB, shape: (N,K).
    """
    # TODOs:
    # - broadcast va to have the same shape as VB; shape (N,K)
    # - compute VB_norms, the row-wise norm of VB; shape (N,1)
    # - compute VB_normalized, where is row has unit norm; shape (N,K)
    # - compute the row-wise inner product <VA, VB_normalized>; shape (N,1)
    # - scale VB_normalized with the scalars above; shape (N,3)

    # 1: broadcast va to have the same shape as VB; shape (N,K)
    Va_broadcast = np.broadcast_to(va, VB.shape)

    # 2: compute VB_norms, the row-wise norm of VB; shape (N,1)
    VB_norms = np.linalg.norm(VB, axis=1, keepdims=True)

    # 3: compute VB_normalized, where is row has unit norm; shape (N,K)
    VB_normalized = VB / VB_norms

    # 4: compute the row-wise inner product <VA, VB_normalized>; shape (N,1)
    inner_product = np.sum(Va_broadcast * VB_normalized, axis=1, keepdims=True)

    # 5: scale VB_normalized with the scalars above; shape (N,3)
    VA_proj = VB_normalized * inner_product
    return VA_proj

    # return np.zeros_like(VB)


def test_example():
    va, VB, VA_proj_gt = example_data()
    VA_proj = batch_projection(va, VB)
    assert numpy_equal(VA_proj, VA_proj_gt)


def test_golden():
    npzfile = np.load("task2.npz")
    va = npzfile["va"]
    VB = npzfile["VB"]
    VA_proj_gt = npzfile["VA_proj"]
    VA_proj = batch_projection(va, VB)
    assert numpy_equal(VA_proj, VA_proj_gt)


def main():
    va, vb, gt = example_data()
    res = batch_projection(va, vb)
    plot_result(va, vb, res)
    plt.show()


if __name__ == "__main__":
    main()
