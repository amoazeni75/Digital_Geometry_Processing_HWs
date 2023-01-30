import einops
import numpy as np
import matplotlib.pyplot as plt
from utils import numpy_equal


def example_data():
    npzfile = np.load("task3.npz")
    A = npzfile["A"]
    B = npzfile["B"]
    return A, B


def plot_slices(A):
    fig = plt.figure(dpi=240)
    N = A.shape[0]
    for n in range(N):
        fig.add_subplot(1, N, n + 1)
        plt.imshow(A[n, :, :, :])
        plt.title(f"A[{n},:,:,:]")
        plt.axis('off')


def rearrange(A):
    """
    Args:
        A: image tensor of shape (?, ?, ?, ?)

    Returns:
        The reshaped image tensor of shape (?, ?, ?).
    """
    # TODOs:
    # - use one line einops.rearrange command to rearrange the tensor.
    print(A.shape)
    #   (6, 96, 96, 3)  -> (192, 288, 3)
    B = einops.rearrange(A, '(h1 h2) w h c -> (h1 w) (h2 h) c', h1=2, h2=3)
    return B


def test_example():
    A, B_gt = example_data()
    B = rearrange(A)
    assert B_gt.shape == B.shape
    assert numpy_equal(B_gt, B)


def main():
    A, _ = example_data()
    B = rearrange(A)
    plt.imshow(B)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
