# %%[markdown] #########################################
# # Execute once (no edits needed)
import numpy as np
import matplotlib.pyplot as plt
from utils import numpy_equal


def basis(t, degree):
    """
    Args:
      t: N positions at which to evaluate the polynomial
      degree: polynomial degree

    Returns:
      numpy array with shape (N, degree+1) whose rows
      contain entries like: [t^0, t^1, t^2, t^3, ... t^degree]

    Example:
      >> basis([0, .25, .5, 1.0], degree=4)
      array([[1.        , 0.        , 0.        , 0.        , 0.        ],
             [1.        , 0.25      , 0.0625    , 0.015625  , 0.00390625],
             [1.        , 0.5       , 0.25      , 0.125     , 0.0625    ],
             [1.        , 1.        , 1.        , 1.        , 1.        ]])
    """
    ts = np.tile(t, (degree + 1, 1))  # (degree+1,N)
    ts[0, :] = 1
    ts = np.transpose(ts)  # (N, degree+1)
    return np.cumprod(ts, 1)  # (N, degree+1)


def setup_figure(xlims):
    plt.figure()
    plt.axis('equal')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.xlim(xlims)
    plt.ylim(0, 0.8)


def example_data(down=1):
    loaded = np.load("task1.npz")
    t, f, f_reg = loaded["t"], loaded["f"], loaded["f_reg"]
    t, f, f_reg = t[::down], f[::down], f_reg[::down]  # < downsample
    xlims = -.4, 1.0
    return t, f, f_reg, xlims


def test_example():
    t, f, _, _ = example_data()
    w = compute_weights(t, f, degree=4)
    # TODO: remove f_reg from npz and instead store weights
    assert numpy_equal(w, [[0.39], [0.03], [-0.56], [0.71], [0.05]], th=0.01)


# %%[markdown] #########################################
# # Functions definition (edit this part)

def sample_polynomial(t, w):
    """
    Args:
      t: the values on horizontal axis, shape: (N,)
      w: the polynomial coefficients, shape: (degree,)
    Return:
      y = f(t): of shape (N,)
    """
    # TASK:
    # - compute the polynomials at the positions t
    # HINTS:
    # - use `@` numpy operator for matrix multiplication
    # - exploit the provided `basis` function
    # compute the polynomials at the positions t
    A = basis(t, w.shape[0] - 1)
    y = A @ w
    return y


def compute_weights(t, f, degree):
    """
    Args:
      t: time, the values on horizontal axis, shape: (N, )
      f: function values on vertical axis: (N, )
    Returns:
      w: the polynomial weights, shape: (degree+1, )
    """
    # TASK:
    # - compute the polynomial weights
    #
    # HINTS:
    # - exploit the provided `basis` function
    # - use `@` numpy operator for matrix multiplication
    # - look at np.linalg.inv and np.linalg.pinv

    # compute the polynomial weights
    A = basis(t, degree)
    w = np.linalg.pinv(A) @ f
    return w.reshape(-1, 1)


def main():
    # --- infer parameters
    t, f, f_rec_gt, xlims = example_data(down=1)
    print(t.shape)

    # --- compute polynomial weights
    degree = 4
    w = compute_weights(t, f, degree=degree)

    # --- reconstruct by sampling regularly
    t_rec = np.linspace(xlims[0], xlims[1], num=1000)
    f_rec = sample_polynomial(t_rec, w)

    # TASK (analysis):
    # - downsample the example data with down=20
    # - what happens as you vary the degree? [5, 15, 25]?
    # - what happns when #DOFs ~ #constraints?
    # - how could you choose an appropriate degreee?

    # --- plot results
    setup_figure(xlims)
    plt.plot(t, f, '.r', label='data')
    plt.plot(t, f_rec_gt, '-r', label='ground truth')
    plt.plot(t_rec, f_rec, '-b', label='reconstruction')
    plt.title(f'poly-degree={degree}, #datapoints={t.shape[0]}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

# %%
