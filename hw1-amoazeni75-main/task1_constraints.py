import numpy as np
from scipy.spatial import KDTree


def sample_constraints(vertices, normals, eps):
    """Sample points near vertices along normals and -normals directions.
        It should work for 2D points.
        It might be implemented in the way that it works both for 2D or 3D points (without using if...)

    Args:
        vertices (np.array, [N, 2 or 3]): 2D/3D coordinates of N points in space
        normals (np.array, [N, 2 or 3]): Normal direction for each vertex (may need to be normalized)
        eps (float): how near should sample points

    Returns:
        new_vert (np.array, [N, 2 or 3]): New sampled points
        new_values (np.array, [N, 1]): Distance value for each of the sampled point
    """
    # For each vertex:
    #  – Sample a new vertex along eps * normal direction
    #  – Check if the new vertex is the closest one to the given vertex
    #  – If not, set eps = eps/2 and start again
    # Repeat the same steps but for -eps
    # Important: there should be __NO__ cycle (for/while) over number of vertices.
    original_eps = eps
    new_vert_positive, new_values_positive = np.zeros((vertices.shape[0], vertices.shape[1])), np.zeros(vertices.shape[0])
    new_vert_negative, new_values_negative = np.zeros((vertices.shape[0], vertices.shape[1])), np.zeros(vertices.shape[0])

    # 1: normalize the input normals
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # 2: sample points along normals. p_new_i = p_i + eps * n_i
    # 2-1: check for each point if it is the closest one to the given vertex, if not, set eps = eps/2 and start again
    new_vert_positive[:] = vertices + eps * normals
    tree = KDTree(vertices)
    id_for_update = np.arange(vertices.shape[0])
    new_values_positive[id_for_update] = eps
    while True:
        dist, nearest_id = tree.query(new_vert_positive[:])
        id_for_update = np.arange(vertices.shape[0])
        if np.all(nearest_id == id_for_update):
            break
        else:  # update the new_vert and eps
            eps /= 2
            id_for_update = id_for_update[nearest_id != np.arange(vertices.shape[0])]
            new_vert_positive[id_for_update] = vertices[id_for_update] + eps * normals[id_for_update]
            new_values_positive[id_for_update] = eps

    # 3: sample points along -normals. p_new_i = p_i - eps * n_i
    # 3-1: check for each point if it is the closest one to the given vertex, if not, set eps = eps/2 and start again
    eps = original_eps
    new_vert_negative[:] = vertices - eps * normals
    tree = KDTree(vertices)
    id_for_update = np.arange(vertices.shape[0])
    new_values_negative[id_for_update] = -1 * eps
    while True:
        dist, nearest_id = tree.query(new_vert_negative[:])
        id_for_update = np.arange(vertices.shape[0])
        if np.all(nearest_id == id_for_update):
            break
        else:  # update the new_vert and eps
            eps /= 2
            id_for_update = id_for_update[nearest_id != np.arange(vertices.shape[0])]
            new_vert_negative[id_for_update] = vertices[id_for_update] - eps * normals[id_for_update]
            new_values_negative[id_for_update] = -1 * eps

    # concatenate the positive and negative samples
    vert = np.concatenate((new_vert_positive, new_vert_negative), axis=0)
    values = np.concatenate((new_values_positive, new_values_negative), axis=0)
    print(vert.shape, values.shape)
    return vert, values
