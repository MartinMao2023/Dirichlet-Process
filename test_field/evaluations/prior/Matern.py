import numpy as np
import matplotlib.pyplot as plt
import math



class Matern:
    def __init__(self, length_scales):
        self.l = length_scales


    def __call__(self, x, length_scales=None):
        if length_scales is None:
            l = self.l
        else:
            l = length_scales

        # coef = math.sqrt(5) 
        coef = math.sqrt(3)
        # coef = 1

        d = np.sqrt(
            np.sum(
            np.square((x[:, None, :] - x[None, :, :]) / l[None, None, :]), 
            axis=-1)) * coef
        # cov = (1 + d + d**2 / 3) * np.exp(-d)
        cov = (1 + d) * np.exp(-d)
        # cov = np.exp(-d)
        return cov






# data_x = np.random.uniform(-3, 3, (24, 3))

# kernel = Matern(np.ones(3))
# cov = kernel(data_x)
# # print(cov)

# plt.imshow(cov)
# plt.colorbar()
# plt.show()


