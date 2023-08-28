import numpy as np
import numpy.linalg as lg

coor_x = np.load("coor_x.npy")
coor_y = np.load("coor_y.npy")
baseline = np.load("filtered_baseline.npy")

# data = np.load("f=0.77_s=0.82_dmg4.npy")
data = np.load("f=0.95_s=0.95_dmg0.npy")

x_coef = lg.inv(coor_x.T @ coor_x) @ coor_x.T @ (data[:, 0] - baseline[:, 0]).reshape(-1, 1)
y_coef = lg.inv(coor_y.T @ coor_y) @ coor_y.T @ (data[:, 1] - baseline[:, 1]).reshape(-1, 1)

predicted_x = coor_x @ x_coef
predicted_y = coor_y @ y_coef

errors = np.square(predicted_x.reshape(-1) + baseline[:, 0] - data[:, 0]) +\
      np.square(predicted_y.reshape(-1) + baseline[:, 1] - data[:, 1])
print(np.mean(errors))



