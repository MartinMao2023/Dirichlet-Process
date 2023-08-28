import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt


baseline = np.load("filtered_baseline.npy")
coor_x = np.load("coor_x.npy")
coor_w = np.load("coor_w.npy")
data = np.load("f=0.85_s=0.9_dmg1.npy")


# X:
original_x_errors = (baseline[:, 0] - data[:, 0]) 

plt.hist(original_x_errors, bins=20, alpha=0.5, label="baseline prior")

x_coef = lg.inv(coor_x.T @ coor_x) @ coor_x.T @ (data[:, 0] - baseline[:, 0]).reshape(-1, 1)
predicted_x = coor_x @ x_coef

residual_x_errors = (original_x_errors + predicted_x.reshape(-1))

plt.xlabel("x deviation", fontsize=14)
plt.hist(residual_x_errors, bins=20, alpha=0.5, label="SVD prior")
plt.legend()
plt.show()


# W:
original_w_errors = (baseline[:, 3] - data[:, 3]) 

plt.hist(np.where(np.abs(original_w_errors)>2, 0, original_w_errors), bins=20, alpha=0.5, label="baseline prior")

w_coef = lg.inv(coor_w.T @ coor_w) @ coor_w.T @ (data[:, 3] - baseline[:, 3]).reshape(-1, 1)
predicted_w = coor_w @ w_coef

residual_w_errors = (original_w_errors + predicted_w.reshape(-1))

plt.xlabel("omega deviation", fontsize=14)
plt.hist(np.where(np.abs(residual_w_errors)>2, 0, residual_w_errors), bins=20,  alpha=0.5, label="SVD prior")
plt.legend()
plt.show()




