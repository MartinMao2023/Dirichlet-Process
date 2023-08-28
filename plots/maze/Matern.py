import numpy as np
import matplotlib.pyplot as plt
import math



class Matern:
    def __init__(self, length_scales=None, v=1.5):
        self.l = length_scales
        if v == 0.5:
            self.matern = self.matern_half
        elif v == 1.5:
            self.matern = self.matern_one_point_five
        elif v == 2.5:
            self.matern = self.matern_two_point_five
        else:
            raise Exception(f"not supported smoothness: v={v}")
    

    @staticmethod
    def matern_half(d):
        return np.exp(-d)
    
    @staticmethod
    def matern_one_point_five(d):
        d *= math.sqrt(3)
        return (1 + d) * np.exp(-d)
    
    @staticmethod
    def matern_two_point_five(d):
        d *= math.sqrt(5) 
        return (1 + d + d**2 / 3) * np.exp(-d)


    def __call__(self, x, y=None, length_scales=None):
        if length_scales is not None:
            l = length_scales
        elif self.l is not None:
            l = self.l
        else:
            raise Exception("No length scale given")

        if y is None:
            d = np.sqrt(
                np.sum(
                np.square((x[:, None, :] - x[None, :, :]) / l[None, None, :]), 
                axis=-1)) 
        else:
            d = np.sqrt(
                np.sum(
                np.square((x[:, None, :] - y[None, :, :]) / l[None, None, :]), 
                axis=-1)) 

        cov = self.matern(d)
        return cov






# data_x = np.random.uniform(-3, 3, (24, 3))

# kernel = Matern(np.ones(3))
# cov = kernel(data_x)
# # print(cov)

# plt.imshow(cov)
# plt.colorbar()
# plt.show()


