import numpy as np
import matplotlib.pyplot as plt

def numpy_equal(a, b, thresh=0.00001):
  return (np.abs(a-b)<thresh).all()

def setup_figure(scale=1.0, symmetric=True):
  # ensures axes have the same 1:1 scale
  plt.axis('scaled')
  # margins min_x max_x min_y max_y
  plt.axis((-scale*symmetric, +scale, -scale*symmetric, +scale))

def plot_arrow(end, begin=(0, 0), color="black", lw=2, zorder=0):
  prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", shrinkA=0, shrinkB=0, color=color, lw=lw)
  return plt.annotate("", xy=end, xytext=begin, arrowprops=prop, zorder=zorder)

def plot_edge(begin, end, *args, **kwargs):
  return plt.plot([begin[0], end[0]], [begin[1], end[1]], *args, **kwargs)

def plot_points(points, color="black", **kwargs):
  return plt.plot(points[:,0], points[:,1], ".", color=color, **kwargs)

def plot_vectors(vectors, **kwargs):
  for vector in vectors:
      plot_arrow(end=vector, **kwargs)