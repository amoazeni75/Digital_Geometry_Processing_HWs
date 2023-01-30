import numpy as np
import matplotlib.pyplot as plt


def numpy_equal(a, b, th=0.00001):
    return (np.abs(a - b) < th).all()


def setup_figure(scale=1.0, symmetric=True):
    # ensures axes have the same 1:1 scale
    plt.axis('scaled')
    # margins min_x max_x min_y max_y
    plt.axis((-scale * symmetric, +scale, -scale * symmetric, +scale))


def plot_arrow(end, begin=(0, 0), color="black", lw=2, zorder=0):
    prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", shrinkA=0, shrinkB=0, color=color, lw=lw)
    return plt.annotate("", xy=end, xytext=begin, arrowprops=prop, zorder=zorder)


def plot_edge(begin, end, *args, **kwargs):
    return plt.plot([begin[0], end[0]], [begin[1], end[1]], *args, **kwargs)


def plot_points(points, color="black", **kwargs):
    return plt.plot(points[:, 0], points[:, 1], ".", color=color, **kwargs)


def plot_vectors(vectors, **kwargs):
    for vector in vectors:
        plot_arrow(end=vector, **kwargs)


import time
import signal


# The following code is inspired from https://stackoverflow.com/a/2281850/1165180
# It is used to break a function after a given time, which is very useful for grading.
class TimeoutException(Exception):  # Custom exception class
    pass


def break_after(seconds=2):
    def timeout_handler(signum, frame):  # Custom signal handler
        raise TimeoutException

    def function(function):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                res = function(*args, **kwargs)
                signal.alarm(0)  # Clear alarm
                return res
            except TimeoutException:
                print(u'Oops, timeout: %s sec reached.' % seconds, function.__name__, args, kwargs)
            return

        return wrapper

    return function
# @break_after(3)
# def test(a, b, c):
#     return time.sleep(2.8999999999)
# test(1,2,3)
