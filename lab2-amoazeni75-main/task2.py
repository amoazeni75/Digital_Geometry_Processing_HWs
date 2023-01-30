# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

# %%[markdown] #########################################
# # Task 2: Normal Estimation
# You need to first complete the `compute_principal_directions` function.
# Then, implement the `compute_normals` function.
# # Execute once (no edits needed)

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def plot_result(points, estimated_normals):
    P, N = points, estimated_normals
    plt.figure()
    plt.axis('equal')
    plt.plot(P[:, 0], P[:, 1], '.k', markersize=5)
    plt.quiver(P[:, 0], P[:, 1], .2 * N[:, 0], .2 * N[:, 1], color='b', angles='xy', scale_units='xy', scale=.05)
    # plt.axis('off')
    plt.show()


def plot_principal_directions(points, pd_vec):
    P, V = points, pd_vec
    mean = P.mean(axis=0)
    plt.figure()
    plt.axis('equal')
    plt.plot(P[:, 0], P[:, 1], '.k', markersize=5, zorder=-1)
    plt.quiver(mean[None, 0], mean[None, 1], *V[:, 0], color='r', angles='xy', scale_units='xy', scale=.5)
    plt.quiver(mean[None, 0], mean[None, 1], *V[:, 1], color='g', angles='xy', scale_units='xy', scale=.5)
    plt.legend(['data', 'principal direction 1', 'principal direction 2'])
    plt.show()


def example_principal_directions(mean=np.array([.5, .5]), cov=np.array([[1.9, 0], [.0, .1]])):
    rd_points = np.random.multivariate_normal(mean, cov, 500)
    rd_points = rd_points @ Rotation.from_euler("z", 30, degrees=True).as_matrix()[0:-1, 0:-1].T
    print(rd_points.shape)
    return rd_points


def example_data_circle():
    theta = np.linspace(0, 2 * np.pi, 50)
    x, y = np.cos(theta), np.sin(theta)
    points = 20 * np.column_stack((x, y))
    normals = np.column_stack((x, y))
    return points, normals


def example_data_dragon():
    loaded = np.load('task2.npz')
    point = loaded['points']
    normals = loaded['estimated_normals']  # < unoriented (estimated by PCA)
    # normals = loaded['normals'] #< oriented normals (unused)
    return point, normals


def compare_normals_unsigned(gt_normal, est_normal):
    dotp = (est_normal * gt_normal).sum(axis=-1)
    return np.abs(1 - np.abs(dotp))


def test_example():
    points, normals_gt = example_data_dragon()
    normals = estimate_normals(points)
    errors = compare_normals_unsigned(normals, normals_gt)
    mean_error = errors.mean(axis=-1)
    assert mean_error < np.cos(np.deg2rad(5))



# %%[markdown] #########################################
# # Functions definition (edit this part)

def compute_principal_directions(points):
    """
    Args:
      points: 2D point cloud, shape: (N, 2)
    Returns:
      pd_vec: the (unit norm) principal directions, shape: (2,)
    """
    # TODOs: compute the principal directions of the (2D) point cloud
    #
    # HINTS:
    # - Move the point cloud to the origin (mean)
    # - Compute the 2x2 covariance matrix, (mupltiply the transpose of the point cloud with itself)
    # - Then you can use np.linalg.eig or np.linalg.eigh to compute the eigen vectors and eigen values
    # - The eigen vector are the principal directions and the eigen values are the length of the principal directions

    # Move the point cloud to the origin (mean)
    points = points - points.mean(axis=0)

    # Compute the 2x2 covariance matrix, (mupltiply the transpose of the point cloud with itself)
    cov = points.T @ points

    # Compute the eigen vectors and eigen values
    eigen_values, eigen_vectors = np.linalg.eig(cov)

    # The eigen vector are the principal directions and the eigen values are the length of the principal directions
    pd_vec = eigen_vectors

    # sort the principal directions by the eigen values from smallest to the largest
    pd_vec = pd_vec[:, np.argsort(eigen_values)]
    return pd_vec


def estimate_normals(points):
    """
    Args:
      points: 2D point cloud, shape: (N, 2)
    Returns:
      normals: the estimated normals, shape: (N, 2)
      warning: use 5 nearest neighbors (or test will fail)
    """
    # TODOs: estimate the normals of the (2D) point cloud
    #
    # HINTS:
    # - To start, compute the normal of the first point (i.e. points[0,:])
    # - Start from the circle data, then move to the dragon to verify
    # - Use cKDTree (search the usage in scipy) to find the nearest neighbours
    # - Use compute_principal_directions to compute the principal directions
    # - Make sure you pick the correct principal direction! (the one with the lowest eigen value)
    # - Use for loop to compute the normal of each point
    # - (Optional) implement the above without for loop (hint: use np.einsum) to speed up the computation

    nearest_neighbors = cKDTree(points).query(points[0], k=5)[1]
    pd_vec = compute_principal_directions(points[nearest_neighbors])
    normals = pd_vec[:, 0]
    for i in range(1, len(points)):
        nearest_neighbors = cKDTree(points).query(points[i], k=5)[1]
        pd_vec = compute_principal_directions(points[nearest_neighbors])
        normals = np.vstack((normals, pd_vec[:, 0]))

    return normals


def main():
    print("First, execute your implementation on principal directions estimation")
    points = example_principal_directions()
    pd_vec = compute_principal_directions(points)
    plot_principal_directions(points, pd_vec)

    print("Then, execute your implementation on normal estimation")
    points, _ = example_data_circle()
    # points, _ = example_data_dragon()
    normals = estimate_normals(points)
    plot_result(points, normals)


if __name__ == '__main__':
    main()

# %%

# %%
