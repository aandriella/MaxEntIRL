from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def heatmap3d(state_reward,  title=''):

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  states = list(zip(*state_reward))[0]
  x, y, z = list(zip(*states))[0], list(zip(*states))[1], list(zip(*states))[2]
  c = list(zip(*state_reward))[1]

  img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
  fig.colorbar(img)
  plt.show()