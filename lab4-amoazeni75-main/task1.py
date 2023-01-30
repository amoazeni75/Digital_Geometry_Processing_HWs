# %%[markdown] #########################################
# Simplification Using Quadric Error Metrics (QEM) in 2D
# more details at https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf

# # Execute once (no edits needed)
# Notice that this is a interactive jupyter notebook,
# afer you installing the jupyter plugin in your vscode,
# you can run the code by pressing the run cell button on the left of the code block, or by pressing shift+enter
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def example_data():
    return scipy.io.loadmat("data/quadric.mat")["X"][:, :-1]  # (2, ?)


def test_example():
    V = example_data()
    quad = get_quadric(V)
    Q = 1 / 4 * (quad[0] + quad[1] + quad[2] + quad[3])
    # F = compute_field(Q)
    x_star = find_representative(Q)
    assert np.abs(x_star.sum() - 198.971) < 0.1


def compute_field(Q):
    # Visualize the quadric scalar field
    F = np.zeros((256, 256))
    XX, YY = np.meshgrid(np.arange(256), np.arange(256))
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            x = np.array([XX[i, j], YY[i, j], 1])
            F[i, j] = x @ Q @ x
    return F


# %%[markdown] #########################################
# # Functions definition (edit this part)

def get_quadric(V):
    """ Compute the quadric matrix for all the faces.
    Args:
      V: The (sorted) vertices of the watertight contour, shape: (2, N)
    Returns:
      quad: (N, 3, 3) The quadric matrix for each edge
    """

    # TODO: Compute the quadric matrix for all the faces
    # Hint: use np.roll to shift the vertices and get edge vectors
    #     - The normals are the rotated edge vectors (make sure the direction is correct!)
    #     - The quadric matrix is the outer product of the augmented normal vector

    # Use np.roll to shift the vertices and get edge vectors
    edges = np.roll(V, -1, axis=1) - V

    # The normals are the rotated edge vectors (make sure the direction is correct!)
    normals = np.array([-edges[1], edges[0]])
    normals = normals / np.linalg.norm(normals, axis=0)

    # calculate the dot product of the normals and the vertices
    dot = -1 * np.sum(normals * V, axis=0, keepdims=True)

    # add the dot product to the normals to get the augmented normal vector
    normals = np.concatenate([normals, dot], axis=0).T

    # The quadric matrix is the outer product of the augmented normal vector, it should be of shape (N, 3, 3)
    quad = normals[:, :, None] * normals[:, None, :]

    return quad


def find_representative(Q):
    """ Compute the location x where x^t Q x is minimized.
    Args:
      Q: The quadric matrix, shape: (3,3)
    Returns:
      x_star: The representative point, shape: [2]
    """
    # Solve the linear system to find representative
    # TODOs: solve the linear system to find representative
    # WARNING: that vertex matrix V is of size 2xN, not Nx2!
    # Hint: use np.linalg.inv
    return np.linalg.inv(Q[:2, :2]) @ (-1 * Q[:2, 2])


def main():
    V = example_data()
    quad = get_quadric(V)

    # Pick one of the following
    Q0 = quad[0]
    Q1 = .5 * (quad[0] + quad[1])
    Q2 = 1 / 3 * (quad[0] + quad[1] + quad[2])
    Q3 = 1 / 4 * (quad[0] + quad[1] + quad[2] + quad[3])

    for ith, Q in enumerate([Q0, Q1, Q2, Q3]):
        F = compute_field(Q)
        plt.imshow(F, cmap='magma', norm=mpl.colors.LogNorm())
        plt.plot(V[0, :], V[1, :], '.-w')
        x_star = find_representative(Q)

        if ith != 0:
            plt.plot(x_star[0], x_star[1], '*r', zorder=100)
            plt.plot([V[0, 0], x_star[0], V[0, ith + 1]], [V[1, 0], x_star[1], V[1, ith + 1]], '.--w')
        plt.show()


if __name__ == "__main__":
    main()
# %%
