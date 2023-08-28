import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

np.random.seed(111)
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0.4*x, 0.3, 1000)


fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(6)
plt.scatter(x, y, s=10)
plt.xlim((-4, 4))
plt.ylim((-3, 3))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.quiver(0, 0, 0.9285, 0.3714, angles="xy", scale_units="xy", scale=0.75, color="#d62728", label="First component")
plt.quiver(0, 0, -0.3714, 0.9285, angles="xy", scale_units="xy", scale=1, color="#ff7f0e", label="Second component")
plt.legend(fontsize=12)
# plt.show()
plt.savefig("before_pca.pdf", dpi=500)


fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(6)
xy = np.vstack((x, y))
cov = np.cov(xy)

S, U = lg.eig(cov)
U = np.float64(U)
# print(S)
xy_ = xy.T @ U / np.sqrt(S)

x, y = xy_.T

plt.scatter(x, y, s=10)
plt.xlim((-4, 4))
plt.ylim((-4, 4))
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.show()
plt.savefig("after_pca.pdf", dpi=500)

