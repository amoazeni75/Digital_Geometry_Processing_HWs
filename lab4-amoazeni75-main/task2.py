# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring

# %%[markdown] #########################################
# Simplification Using Quadric Error Metrics (QEM) in 2D
# more details at https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf

# # Execute once (no edits needed)
# Notice that this is a interactive jupyter notebook,
# afer you installing the jupyter plugin in your vscode,
# you can run the code by pressing the run cell button on the left of the code block, or by pressing shift+enter
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from task1 import get_quadric


def example_data():
    # notice that X is 2xN, not Nx2!
    return scipy.io.loadmat("data/decimation.mat")["P"]


def test_example():
    V = example_data()  # (2, N)
    V_decimated = decimate(V, target=50)
    assert V_decimated.shape == (2, 50)

    tree = cKDTree(V_decimated.T)
    dist, _ = tree.query(V.T)
    # the distance from original points to simplified points
    avg_dist = np.mean(dist)
    assert avg_dist < 4.9


# %%[markdown] #########################################
# # Functions definition (edit this part)

def decimate_init(V):
    """ Remove a vertex from the representation
    Args:
      V: vertices of the mesh, shape: (2, N)
    Returns:
      Q: corresponding quadrics (N, 3, 3)
    """
    ############ TODO: YOUR CODE HERE ############
    # Hint: Compute the quadric Qf for all faces(edges) and Q for each vertex
    # Q: (N, 3, 3), Q[i] is the quadric for the ith vertex
    # Hint: Q[i] = Qf[i] + Qf[i-1], where Qf[-1] = Qf[N-1]
    ############ END OF YOUR CODE #################
    QF = get_quadric(V)
    Q = np.zeros((V.shape[1], 3, 3))
    for i in range(V.shape[1]):
        Q[i] = QF[i] + QF[i - 1]

    return Q


def decimate_step(V, Q):
    """ Remove a vertex from the representation
    Args:
      V: vertices of the mesh, shape: (2, N)
      Q: corresponding quadrics (N, 3, 3)
    Returns:
      V: vertices of the mesh, shape: (2, N-1)
      Q: corresponding quadrics (N-1, 3, 3)
    """
    ############ TODO: YOUR CODE HERE ############
    # 1) find the optimal vertex to collapse (linear time search is "ok" in 2D)
    # 2) update the quadric to reflect the collapse
    # 3) delete the vertex (and its quadric) from the list
    # hint: no need to write vectorized numpy code right away (bonus)
    min_cost = np.inf
    min_index = 0
    for i in range(V.shape[1]):
        # add 1 to the vertex to make it a homogeneous coordinate
        v = np.append(V[:, i], 1)
        cost = np.sum(np.matmul(np.matmul(v.T, Q[i]), v))
        if cost <= min_cost:
            min_cost = cost
            min_index = i
    if min_index == V.shape[1] - 1:
        Q[0] += Q[min_index]
    else:
        Q[min_index + 1] += Q[min_index]
    V = np.delete(V, (min_index), axis=1)
    Q = np.delete(Q, (min_index), axis=0)

    return V, Q
    ############ END OF YOUR CODE #################


def visualize(V_input, V_dec, last_iteration: bool):
    plt.plot([*V_input[0, :], V_input[0, 0]], [*V_input[1, :], V_input[1, 0]], '.-r')
    plt.plot([*V_dec[0, :], V_dec[0, 0]], [*V_dec[1, :], V_dec[1, 0]], '.-b', zorder=100)
    if last_iteration:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(0.01)
    # plt.cla()


def decimate(V_input, target=50, plot_per_steps=-1):
    """ Mesh decimation using quadric error metrics
    Args:
      V_input: The vertices of the watertight one-component 2D mesh, shape: (2, N)
      target: The final number of vertices after decimation
    Returns:
      V: The vertices of the simplified mesh, shape: (2, target)
    """
    V = V_input.copy()  # < keep a copy for rendering
    Q = decimate_init(V)
    num_iterations = V_input.shape[1] - target

    plt.ion()
    for i in range(num_iterations):
        V, Q = decimate_step(V, Q)
        if (plot_per_steps != -1 and i % plot_per_steps == 0) or i + 1 == num_iterations:
            plt.title(f"iteration: {i}/{num_iterations}")
            visualize(V_input, V, i + 1 == num_iterations)

            # plt.savefig('output/{0:05d}.png'.format(i))
    return V


def main():
    V_input = example_data()  # (2, N)
    decimate(V_input, target=50, plot_per_steps=50)


if __name__ == "__main__":
    main()
# %%
