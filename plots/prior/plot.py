import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 500)
prior = np.exp(-1*(x - 1.5)**2) + np.exp(-0.25*(x - 4)**2)*2


line1 = np.sin(x)*0.25
line2 = np.cos(x+2)*0.25


plt.plot(x, prior, label="prior mean")
plt.plot(x, prior+line1, linestyle="--", label="experience1")
plt.plot(x, prior+line2, linestyle="--", label="experience2")
plt.legend()
plt.show()

