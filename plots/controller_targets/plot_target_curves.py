import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt






guassian_kernel = np.array([0.0008880585, 0.0015861066, 0.00272177, 0.00448744, 0.007108437, 0.010818767, 
                            0.015820118, 0.022226436, 0.03000255, 0.03891121, 0.048486352, 0.058048703, 
                            0.0667719, 0.073794365, 0.078357555, 0.07994048, 0.078357555, 0.073794365, 
                            0.0667719, 0.058048703, 0.048486352, 0.03891121, 0.03000255, 0.022226436, 
                            0.015820118, 0.010818767, 0.007108437, 0.00448744, 0.00272177, 0.0015861066, 
                            0.0008880585], dtype=np.float32)



t = np.linspace(0, 1000, 100)
squared_wave = np.ones(100)
squared_wave[:10] = -1
squared_wave[-30:] = -1
squared_wave *= 0.8

targets = ndimage.convolve1d(squared_wave, guassian_kernel, axis=0, mode="wrap")

plt.plot(t, squared_wave, label="squared wave", linestyle="--", linewidth=2)
plt.plot(t, targets, label="smoothed targets", linewidth=2, alpha=0.9)
plt.xlabel("time (ms)", fontsize=16)
plt.ylabel("target", fontsize=16)
plt.legend(fontsize=12, loc=(0.2, 0.5))
plt.savefig("targets.pdf", dpi=500)
# plt.show()




