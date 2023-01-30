import numpy as np
from scipy.spatial import KDTree
from task1_constraints import sample_constraints
from skimage import measure
from utils import read_off, to_np, bounding_box_diag


def generate_grid(point_cloud, res, num_dims=3):
    """Generate grid over the point cloud with given resolution

    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution

    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    if num_dims == 3:
        coords = np.mgrid[:res, :res, :res]
        coords = coords.reshape(3, -1)
        coords_matrix = np.eye(4)
        length = b_max - b_min
        length += length / res
        coords_matrix[0, 0] = length[0] / res
        coords_matrix[1, 1] = length[1] / res
        coords_matrix[2, 2] = length[2] / res
        coords_matrix[0:3, 3] = b_min
        coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
        coords = coords.T
    elif num_dims == 2:
        coords = np.mgrid[:res, :res]
        coords = coords.reshape(2, -1)
        coords_matrix = np.eye(3)
        length = b_max - b_min
        length += length / res
        coords_matrix[0, 0] = length[0] / res
        coords_matrix[1, 1] = length[1] / res
        coords_matrix[0:2, 2] = b_min
        coords = np.matmul(coords_matrix[:2, :2], coords) + coords_matrix[:2, 2:3]
        coords = coords.T
    else:
        assert 0
    return coords, coords_matrix


def transform_to_polynomial_basis(pts, degree):
    """ Represent 2D/3D points (pts) in polynomial basis of given degree
    e.g. degree 2: (x, y) -> (1, x, y, x^2, y^2, xy)
    
    Args:
        pts (np.array [N, 2 or 3]): 2D/3D coordinates of N points in space
        degree (int): degree of Polynomial

    Returns:
        ext_pts (np.array [N, ?]): points (pts) in Polynomial basis of given degree. 
        The second dimension depends on the polynomial degree and initial dimention of points (2D or 3D)  
    """
    if degree == 0:
        return np.ones([len(pts), 1])

    if pts.shape[1] == 2:
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        ext_pts = np.concatenate([np.ones([len(pts), 1]), x, y], axis=1)
        for i in range(2, degree + 1):
            for j in range(i + 1):
                term = (x ** (i - j)) * (y ** j)
                ext_pts = np.concatenate([ext_pts, term], axis=1)
    elif pts.shape[1] == 3:
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        z = pts[:, 2:3]
        ext_pts = np.concatenate([np.ones([len(pts), 1]), x, y, z], axis=1)
        for i in range(2, degree + 1):
            for j in range(i + 1):
                for k in range(i - j + 1):
                    term = (x ** (i - j - k)) * (y ** j) * (z ** k)
                    ext_pts = np.concatenate([ext_pts, term], axis=1)
    return ext_pts


def global_predictor(grid_pts, constr_pts, constr_vals, degree=2):
    """Evaluate implicit function in space

    Args:
        grid_pts (np.array [N, 3]): 3D coordinates of N points in space. Grid points
        constr_pts (np.array [M, 3]): 3D coordinates of M points in space. Constraints points
        constr_vals (np.array, [M, 1]): constraint values defined on constr_pts
        degree (int): degree of Polynomial

    Returns:
        pred_vals (np.array [N, 1]): implicit function values for each of the grid points
    """

    # 1: Convert grid and constraint points into required polynomial basis (degree).
    grid_pts_poly = transform_to_polynomial_basis(grid_pts, degree)
    constraint_pts_poly = transform_to_polynomial_basis(constr_pts, degree)

    # 2: Solve linear least squares problem.
    coefs = np.linalg.inv(constraint_pts_poly.T @ constraint_pts_poly) @ constraint_pts_poly.T @ constr_vals

    # 3: Evaluate implicit function values in the grid points.
    pred_vals = grid_pts_poly @ coefs
    return pred_vals


def local_predictor(grid_pts, constr_pts, constr_vals, local_radius, degree, reg_coef=0):
    """Evaluate implicit function in space

    Args:
        grid_pts (np.array [N, 2 or 3]): 2D/3D coordinates of N points in space. Grid points
        constr_pts (np.array [M, 2 or 3]): 2D/3D coordinates of M points in space. Constraints points
        constr_vals (np.array, [M, 1]): constraint values defined on constr_pts
        local_radius (float): parameter to set the weight function / neighbourhood radius
        degree (int): degree of Polynomial
        reg_coef (optional) (float): regularization parameter

    Returns:
        pred_vals (np.array [N, 1]): implicit function values for each of the grid points
    """
    tree = KDTree(constr_pts)
    pred_vals = np.zeros(len(grid_pts))
    for cur_grid_idx, g_pt in enumerate(grid_pts):
        closest_pts_ids = tree.query_ball_point(g_pt, local_radius)
        closest_pts = constr_pts[closest_pts_ids]
        closest_vals = constr_vals[closest_pts_ids].reshape(-1, 1)

        # the nubmer of neighbours should be at least (degree + 1) ** 2
        if grid_pts.shape[1] == 2:
            need_neighbours = ((degree + 1) * (degree + 2)) // 2
        else:
            need_neighbours = ((degree + 1) * (degree + 2) * (degree + 3)) // 6

        if len(closest_pts) < need_neighbours:
            pred_vals[cur_grid_idx] = 1000
        else:
            pred_vals[cur_grid_idx] = eval_grid_point(g_pt, closest_pts, closest_vals, local_radius, degree, reg_coef)
    return pred_vals


def eval_grid_point(eval_pt, local_constr_pts, local_constr_vals, local_radius, degree, reg_coef):
    """Evaluate implicit function at the eval point

    Args:
        eval_pt (np.array [2 or 3,]): 2D/3D coordinate of a point in space. A grid point
        local_constr_pts (np.array [K, 2 or 3]): 2D/3D coordinates of M points in space (constrain points)
        local_constr_vals (np.array, [K, 1]): constraint values defined on local_constr_pts
        local_radius (float): parameter to set the weight function
        degree (int): degree of Polynomial
        reg_coef (optional) (float): regularization parameter

    Returns:
        pred_val (float): implicit function value at the point
    """
    weights = wendland(np.linalg.norm(eval_pt - local_constr_pts, axis=1), local_radius)

    # 1: Convert grid point and constraint points into required polynomial basis (degree).
    local_constr_pts_poly = transform_to_polynomial_basis(local_constr_pts, degree)
    eval_pt_poly = transform_to_polynomial_basis(eval_pt.reshape(1, -1), degree)

    # 2: Solve weighted linear least squares problem with the regularization parameter.

    coefs = np.linalg.inv(local_constr_pts_poly.T @ np.diag(weights) @ local_constr_pts_poly + reg_coef * np.eye(local_constr_pts_poly.shape[1])) @ local_constr_pts_poly.T @ np.diag(weights) @ local_constr_vals

    # 3: Evaluate implicit function values in the grid points.
    pred_val = eval_pt_poly @ coefs

    return pred_val


def wendland(r, h):
    """Wendland weight function: (1 - r/h)^4 * (4 * r/h + 1); if r>=h -> weight=0

    Args:
        r (np.array [N] or float): distance parameter
        h (float): weight parameter

    Returns:
        weights (np.array [N, 1] or float): weight function values
    """
    assert h >= 0
    if isinstance(r, float) or isinstance(r, int):
        assert r >= 0
    else:
        assert (r >= 0).all()

    x = r / h
    weights = (1 - x) ** 4 * (4 * x + 1)
    if isinstance(r, float) or isinstance(r, int):
        if r >= h:
            weights = 0
    else:
        weights[r >= h] = 0
    return weights


def full_reconstruction(data, resolution, predictor_type, degree=2, reg_coef=0., num_dims=2, eps_mul=0.1, radius_mul=0.1):
    vertices, normals = data['verts'], data['normals']
    bbox_diag = bounding_box_diag(vertices)
    eps = bbox_diag * eps_mul
    new_verts, new_vals = sample_constraints(vertices, normals, eps)

    constr_pts = np.concatenate([vertices, new_verts])
    constr_vals = np.concatenate([np.zeros(len(vertices)), new_vals])

    grid_pts, coords_matrix = generate_grid(constr_pts, resolution, num_dims=num_dims)

    if predictor_type == 'global':
        pred_vals = global_predictor(grid_pts, constr_pts, constr_vals, degree=degree)
    elif predictor_type == 'local':
        local_radius = bbox_diag * radius_mul
        pred_vals = local_predictor(grid_pts, constr_pts, constr_vals, local_radius, degree=degree, reg_coef=reg_coef)
    else:
        assert 0, 'unknown model type'

    if num_dims == 2:
        contours = measure.find_contours(pred_vals.reshape(resolution, resolution), 0)
        if len(contours) == 1:
            verts = contours[0]
        elif len(contours) == 0:
            return np.array([]), np.array([])
        else:
            verts = np.concatenate(contours, axis=0)
        verts = (coords_matrix[:2, :2] @ verts.T + coords_matrix[:2, 2:3]).T
        faces = np.array([])

    elif num_dims == 3:
        if np.unique(pred_vals).shape[0] == 1:
            verts, faces = np.array([]), np.array([])
        else:
            verts, faces, _, _ = measure.marching_cubes(pred_vals.reshape([resolution, resolution, resolution]), level=0)
            verts = (coords_matrix[:3, :3] @ verts.T + coords_matrix[:3, 3:4]).T
    else:
        assert 0, 'unknown num dims'

    return verts, faces
