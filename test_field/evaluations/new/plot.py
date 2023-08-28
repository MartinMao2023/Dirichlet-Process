import numpy as np
import matplotlib.pyplot as plt


elbow_data = np.load("elbow_data.npy")

x = np.arange(1, 31)
xy, vwphi = elbow_data

plt.plot(x, xy, label="xy bases")
plt.plot(x, vwphi, label="trajectory bases")
plt.xlabel("dimension of bases")
plt.ylabel("loss")
plt.legend()
# plt.show() 
plt.savefig("elbow.png", dpi=500)


