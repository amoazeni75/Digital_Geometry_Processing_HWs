# %%[markdown] #########################################
# # Execute once (no edits needed)
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from utils import numpy_equal


def plot_result(query_grid, sdf_grid):
    plt.imshow(sdf_grid, origin="lower", cmap="seismic", norm=mpl.colors.TwoSlopeNorm(vcenter=0.))
    plt.contour(query_grid[..., 0], query_grid[..., 1], sdf_grid, levels=[0], colors=['black'])


def example_data():
    loaded = np.load("data/hoppe.npz")
    points, normals = loaded["points"], loaded["normals"]
    query_grid, gt_sdf_grid = loaded["query_grid"], loaded["sdf_grid"]
    return points, normals, query_grid, gt_sdf_grid


def test_example():
    points, normals, query_grid, gt_sdf_grid = example_data()
    queries = query_grid.reshape(-1, query_grid.shape[-1])
    sdfs = reconstruct(points, normals, queries)
    sdf_grid = sdfs.reshape(*gt_sdf_grid.shape)
    assert numpy_equal(sdf_grid, gt_sdf_grid)


# %%[markdown] #########################################
# # Functions definition (edit this part)

def reconstruct(points, normals, queries):
    """
    Args:
      points: The coordinates of a 2D point cloud, shape: (N, 2)
      normals: The normals of the point cloud, shape: (N, 2)
      queries: The query positions, shape: (M, 2)
    Returns:
      The SDF value of the queries, shape: (M)
    """
    # TODOs: compute the hoppe implicit function
    # - Use scipy cKDTree to find the nearest neighbor of each query point
    # - Compute the dot product of the normal and the vector from the nearest neighbor to the query point
    # HINTS: For the batch dot product, you can use np.einsum

    # dummy sdf (for display)
    tree = cKDTree(points)
    dist, idx = tree.query(queries)

    # - Compute the dot product of the normal and the vector from the nearest neighbor to the query point
    sdf = np.einsum('ij,ij->i', normals[idx], queries - points[idx])

    return sdf


def main():
    points, normals, query_grid, _ = example_data()
    queries = query_grid.reshape(-1, query_grid.shape[-1])
    sdfs = reconstruct(points, normals, queries)
    sdf_grid = sdfs.reshape(*query_grid.shape[0:2])
    plot_result(query_grid, sdf_grid)


if __name__ == "__main__":
    main()
    # test_example()
# %%
