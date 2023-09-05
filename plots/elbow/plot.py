import numpy as np
import matplotlib.pyplot as plt


elbow_data = np.load("elbow_data.npy")

x = np.arange(1, 31)
xy, vwphi = elbow_data

plt.plot(x, xy, label="displacement-based")
plt.plot(x, vwphi, label="arc-based")
plt.xlabel("number of basis", fontsize=14)
plt.ylabel("MSE loss", fontsize=14)
plt.legend(fontsize=12)
# plt.show() 
plt.savefig("elbow.pdf", dpi=500)


plt.figure()
filtered_elbow_data = np.load("elbow_data_filtered.npy")
xy, vwphi = filtered_elbow_data
plt.plot(x, xy, label="displacement-based")
plt.plot(x, vwphi, label="arc-based")
plt.xlabel("number of basis", fontsize=14)
plt.ylabel("MSE loss", fontsize=14)
plt.legend(fontsize=12)
# plt.show() 
plt.savefig("filtered_elbow.pdf", dpi=500)


