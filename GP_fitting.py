import numpy as np
import numpy.linalg as lg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math

data_x = np.random.uniform(-5, 5, (24, 2))
data_y = data_x @ np.array([[2.5], [1]]) + np.prod(np.sin(data_x), axis=1, keepdims=True) + 1.2



def mean_fitting(data_x, data_y):
    cat_x = np.hstack((data_x, np.ones((len(data_x), 1))))
    X = cat_x.T @ cat_x
    weights = lg.inv(X) @ cat_x.T @ data_y
    return weights


# print(mean_fitting(data_x, data_y))

def negative_log_likelihood(params, data_x, diff_y):
    length_scale = params[:-2]
    prior_var, noise_var = params[-2:]
    d = np.sqrt(np.sum(np.square((data_x[:, None, :] - data_x[None, :, :]) / length_scale[None, None, :]), axis=-1))

    # print(d.shape)
    # plt.imshow(d)
    # plt.colorbar()
    # plt.show()
    cov = prior_var*(1 + math.sqrt(5)*0.5*d + 5 / 12 * np.square(d)) * np.exp(-0.5*math.sqrt(5)*d)
    cov += np.eye(len(diff_y)) * noise_var
    # plt.imshow(cov)
    # plt.colorbar()
    # plt.show()
    result =  0.5 * diff_y.T @ lg.inv(cov) @ diff_y + 0.5 * np.log(lg.det(cov))
    return float(result)



diff_y = data_y-1.2 - data_x @ np.array([[2.5], [1]])
print(negative_log_likelihood(np.array([1,1, 1, 0.01]), data_x, diff_y))

a = minimize(negative_log_likelihood, 
             np.array([1,1, np.var(diff_y), 0.01]), 
             args=(data_x, diff_y), 
             bounds=np.array([
                 [1e-5, 1e4],
                 [1e-5, 1e4],
                 [1e-6, 1e4],
                 [1e-8, 1]
             ])) 

print(a.x)

print(negative_log_likelihood(a.x, data_x, diff_y))
print(negative_log_likelihood(np.array([0.5, 0.5, np.var(diff_y), 1e-8]), data_x, diff_y))

# print(np.var(diff_y))
