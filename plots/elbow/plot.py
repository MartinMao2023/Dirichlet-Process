import numpy as np
import matplotlib.pyplot as plt


elbow_data = np.load("elbow_data.npy")

x = np.arange(1, 31)
xy, vwphi = elbow_data

plt.plot(x, xy, label="xy bases")
plt.plot(x, vwphi, label="trajectory bases")
plt.xlabel("dimension of bases", fontsize=14)
plt.ylabel("MSE loss", fontsize=14)
plt.legend(fontsize=12)
plt.show() 
# plt.savefig("elbow.png", dpi=500)


filtered_elbow_data = np.load("elbow_data_filtered.npy")
xy, vwphi = filtered_elbow_data
plt.plot(x, xy, label="xy bases")
plt.plot(x, vwphi, label="trajectory bases")
plt.xlabel("dimension of bases", fontsize=14)
plt.ylabel("MSE loss", fontsize=14)
plt.legend(fontsize=12)
plt.show() 


