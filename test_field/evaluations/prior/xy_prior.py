import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Matern import Matern


baseline = np.load("../filtered_baseline.npy")
# print(baseline.shape)
baseline_x = baseline[:, 0]
baseline_y = baseline[:, 1]

repertoire_size = len(baseline)

coor_x = np.load("../coor_x.npy")
coor_y = np.load("../coor_y.npy")

kernel = Matern(np.ones(5))


def negative_log_likelihood(params, data_x, diff_y):
    length_scale = params[:-1]
    noise_ratio = params[-1]
    if noise_ratio < 1:
        cov = 0.5*kernel(data_x, length_scale) + 0.5*np.eye(64)*noise_ratio
        log_scale = np.log(2)
    else:
        cov = 0.5*kernel(data_x, length_scale) / noise_ratio + 0.5*np.eye(64)
        log_scale = np.log(2*noise_ratio)
    result = 0.5*np.log(lg.det(cov)+1e-100) + 32*log_scale + 32*np.log(diff_y.T @ lg.inv(cov) @ diff_y) - 32*log_scale
    return float(result)



GP_x = np.zeros((128, 13), dtype=np.float32)
GP_y = np.zeros((128, 13), dtype=np.float32)



def fit_GP(coors, target, threshold=0.001):
    coef = lg.inv(coors.T @ coors) @ coors.T @ target
    prediction = coors @ coef
    diff = target - prediction

    # print(coef)

    results = []
    for initial_noise_ratio in (1, 0.5, 0.01):
        res = minimize(negative_log_likelihood, 
                    np.array([1 for i in range(5)] + [initial_noise_ratio]), 
                    args=(coors, diff), 
                    bounds=np.array([
                        [1e-5, 1e4],
                        [1e-5, 1e4], 
                        [1e-5, 1e4],
                        [1e-5, 1e4],
                        [1e-5, 1e4],
                        [1e-6, 1e6]]))
        results.append(res)

    best_res = min(results, key=lambda x: x.fun)
    length_scale = best_res.x[:5]
    noise_ratio = best_res.x[-1]
    cov = kernel(coors, length_scale) + np.eye(64) * noise_ratio

    inverse_cov = lg.inv(cov)
    coef = lg.inv(coors.T @ inverse_cov @ coors) @ coors.T @ inverse_cov @ target

    # print(coef)

    prediction = coors @ coef
    diff = target - prediction
    res = minimize(negative_log_likelihood, 
                    best_res.x, 
                    args=(coors, diff), 
                    bounds=np.array([
                        [1e-5, 1e4],
                        [1e-5, 1e4], 
                        [1e-5, 1e4],
                        [1e-5, 1e4],
                        [1e-5, 1e4],
                        [1e-6, 1e6]]))
    
    length_scale = res.x[:5]
    noise_ratio = res.x[-1]
    cov = kernel(coors, length_scale) + np.eye(64) * noise_ratio
    prior_var =  diff.T @ lg.inv(cov) @ diff / 64
    # cov *= prior_var
    # print(noise_ratio)
    # print(prior_var)
    # print(length_scale)
    # print(np.mean(np.abs(diff)))

    result = np.zeros(13, dtype=np.float32)
    result[:5] = coef[:, 0]
    result[5:10] = length_scale
    result[10] = float(prior_var)
    result[11] = noise_ratio

    cov -= np.diag(np.diagonal(cov))
    if np.mean(cov.reshape(-1))*prior_var < threshold:
        result[-1] = 0
        # print("yes", float(np.mean(cov.reshape(-1))*prior_var))
    else:
        result[-1] = 1

    return result



for index in range(128):
    data = np.load(f"{index}.npy")
    indexes = np.int32(data[:, 0])
    x_target = (data[:, 1] - baseline_x[indexes]).reshape(-1, 1)
    x = coor_x[indexes]
    y_target = (data[:, 2] - baseline_y[indexes]).reshape(-1, 1)
    y = coor_y[indexes]

    GP_x[index] = fit_GP(x, x_target)
    GP_y[index] = fit_GP(y, y_target)


np.save("GP_x.npy", GP_x)
np.save("GP_y.npy", GP_y)

print("x:", round(np.sum(GP_x[:, -1]) / len(GP_x)*100, 2))
print("y:", round(np.sum(GP_y[:, -1]) / len(GP_y)*100, 2))



