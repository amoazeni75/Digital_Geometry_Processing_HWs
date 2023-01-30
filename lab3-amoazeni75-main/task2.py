# %%[markdown] #########################################
# # Execute once (no edits needed)
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from utils import numpy_equal


def example_data():
    phi = np.load('data/marching_squares.npy')
    SKIP = 6
    return phi, SKIP


def test_example():
    phi, SKIP = example_data()
    edges = marching_squares(phi, SKIP)
    edge_l2_sum = (edges[:, 1, :] - edges[:, 0, :]) ** 2
    edge_l2_sum = edge_l2_sum.sum(axis=-1).mean(axis=0)
    assert np.abs(edge_l2_sum - 26.567) < 0.01


# %%[markdown] #########################################
# # Functions definition (edit this part)


def trace(phi, e1, e2):
    """
    Args:
    phi: The SDF grid, shape: (N,N)
    e1: The first edge point, shape: (2)
    e2: The second edge point, shape: (2)
    Returns:
    p: The intersection point, shape: (2)
    check: Whether there is an intersection, shape: (1) bool
    """
    p = np.array([np.nan, np.nan])
    # TODOs: check whether there is an intersection point of the edge and the zero level set
    # - if there is no intersection, return (np.array([np.nan, np.nan]), False)
    # - if there is an intersection p, return (p, True)
    # Hint: compute p according to the sdf values at e1 and e2

    phi_1 = phi[e1[0], e1[1]]
    phi_2 = phi[e2[0], e2[1]]
    if phi_1 * phi_2 < 0:
        check = True
        p = e1 + (e2 - e1) * (phi_1 / (phi_1 - phi_2))
    else:
        check = False
    return p, check


def marching_squares(phi, SKIP=6):
    plt.figure()
    plt.axis('off')
    edges = []
    for x in range(SKIP, phi.shape[0] - SKIP, SKIP):
        for y in range(SKIP, phi.shape[1] - SKIP, SKIP):
            grd = np.zeros((4, 2), dtype=int)
            grd[0] = np.array([x, y])
            grd[1] = np.array([x + SKIP, y])
            grd[2] = np.array([x + SKIP, y + SKIP])
            grd[3] = np.array([x, y + SKIP])
            p0, b0 = trace(phi, grd[0], grd[1])
            p1, b1 = trace(phi, grd[1], grd[2])
            p2, b2 = trace(phi, grd[2], grd[3])
            p3, b3 = trace(phi, grd[3], grd[0])
            P = np.array([p0, p1, p2, p3])
            B = np.array([b0, b1, b2, b3])
            E = P[B == True, :]
            if E.shape[
                0] == 2:  # no handling of degenerate cases, to see more, please check http://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html
                plt.plot(E[:, 0], E[:, 1], '-k', zorder=100)
                edges.append(E)
                grd_check = np.logical_or(B, np.roll(B, 1))  # one edge intersects, two corners are involved
                plt.plot(grd[grd_check, 0], grd[grd_check, 1], 'g.', markersize=5)
                plt.plot(E[:, 0], E[:, 1], 'r.', markersize=7)

    thresh = np.maximum(np.abs(phi.min()), phi.max())
    plt.imshow(phi.T, cmap="seismic", norm=mpl.colors.TwoSlopeNorm(vmin=-thresh, vcenter=0., vmax=thresh),
               origin="lower")
    return np.array(edges)


def main():
    phi, SKIP = example_data()
    edges = marching_squares(phi, SKIP)
    test_example(edges)


if __name__ == "__main__":
    main()
# %%
