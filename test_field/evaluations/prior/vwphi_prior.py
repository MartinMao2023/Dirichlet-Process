import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Matern import Matern


baseline = np.load("../filtered_baseline.npy")
# print(baseline.shape)
baseline_v = baseline[:, 2]
baseline_w = baseline[:, 3]
baseline_phi = baseline[:, 4]


repertoire_size = len(baseline)

coor_v = np.load("../coor_v.npy")
coor_w = np.load("../coor_w.npy")
coor_phi = np.load("../coor_phi.npy")


kernel = Matern(np.ones(5))


def negative_log_likelihood(params, data_x, diff_y, noises):
    length_scale = params[:-1]
    noise_ratio = params[-1]

    log_noise_scale = np.mean(np.log(noises*noise_ratio))

    if log_noise_scale < 0:
        cov = 0.5*kernel(data_x, length_scale) + 0.5*np.diag(noises)*noise_ratio
        log_scale = np.log(2)
    else:
        cov = 0.5*kernel(data_x, length_scale) / (noise_ratio) + 0.5*np.diag(noises)
        cov *= np.exp(-log_noise_scale)
        log_scale = np.log(2*noise_ratio) + log_noise_scale
    result = 0.5*np.log(lg.det(cov)+1e-100) + 32*log_scale + 32*np.log(diff_y.T @ lg.inv(cov) @ diff_y) - 32*log_scale
    return float(result)



GP_v = np.zeros((128, 13), dtype=np.float32)
GP_w = np.zeros((128, 13), dtype=np.float32)
GP_phi = np.zeros((128, 13), dtype=np.float32)



def fit_GP(coors, noises, target, threshold):
    coef = lg.inv(coors.T @ coors) @ coors.T @ target
    prediction = coors @ coef
    diff = target - prediction

    # print(coef)
    # print(np.mean(np.square(diff)))

    results = []
    for initial_noise_ratio in (1, 0.5, 0.01):
        res = minimize(negative_log_likelihood, 
                    np.array([1 for i in range(5)] + [initial_noise_ratio]), 
                    args=(coors, diff, noises), 
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
    cov = kernel(coors, length_scale) + np.diag(noises)*noise_ratio
    # prior_var =  diff_x.T @ lg.inv(cov) @ diff_x / 64
    # cov *= prior_var

    # print(length_scale)
    # print(np.mean(np.abs(diff)))
    # print(noise_ratio)

    inverse_cov = lg.inv(cov)
    coef = lg.inv(coors.T @ inverse_cov @ coors) @ coors.T @ inverse_cov @ target

    # print(coef)

    prediction = coors @ coef
    diff = target - prediction
    res = minimize(negative_log_likelihood, 
                    best_res.x, 
                    args=(coors, diff, noises), 
                    bounds=np.array([
                        [1e-5, 1e4],
                        [1e-5, 1e4], 
                        [1e-5, 1e4],
                        [1e-5, 1e4],
                        [1e-5, 1e4],
                        [1e-6, 1e6]]))
    
    length_scale = res.x[:5]
    noise_ratio = res.x[-1]
    cov = kernel(coors, length_scale) + np.diag(noises) * noise_ratio
    prior_var =  diff.T @ lg.inv(cov) @ diff / 64
    # cov *= prior_var

    # prediction = coors @ coef
    # diff = target - prediction
    # print(np.mean(np.square(diff)))

    # print(noise_ratio)
    # print(prior_var)
    # print(length_scale)
    # print(np.mean(np.abs(diff)))

    # print(np.mean(noises), noise_ratio)
    # plt.imshow(np.where(cov > 2, 2, cov))
    # plt.colorbar()
    # plt.show()
    

    # print(np.diag(np.diagonal(cov)).shape)
    # plt.imshow(cov*prior_var)
    # plt.colorbar()
    # plt.show()



    result = np.zeros(13, dtype=np.float32)
    result[:5] = coef[:, 0]
    result[5:10] = length_scale
    result[10] = float(prior_var)
    result[11] = noise_ratio

    cov -= np.diag(np.diagonal(cov))
    if np.mean(cov.reshape(-1))*prior_var < threshold:
        result[-1] = 0
        # print("yes")
    else:
        result[-1] = 1

    return result



for index in range(128):
    data = np.load(f"{index}.npy")
    indexes = np.int32(data[:, 0])
    v_target = (data[:, 3] - baseline_v[indexes]).reshape(-1, 1)
    v_ = coor_v[indexes]
    v = data[:, 3]
    w_target = (data[:, 4] - baseline_w[indexes]).reshape(-1, 1)
    w_ = coor_w[indexes]
    w = data[:, 4]
    w = np.where(w == 0, 1e-6, w)
    phi_target = np.mod(data[:, 5] - baseline_phi[indexes] + 3*np.pi, 2*np.pi).reshape(-1, 1) - np.pi
    phi_ = coor_phi[indexes]
    phi = data[:, 5]

    # plt.hist(phi_target)
    # plt.show()
    
    sin_theta = np.sin(w*4 + phi)
    cos_theta = np.cos(w*4 + phi)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    x_ = v/w*(sin_theta - sin_phi)
    y_ = v/w*(cos_phi - cos_theta)

    noise_v = v**2 / (x_**2 + y_**2 + 1e-6) + 1e-6
    noise_w = (w**2 + 0.01) / (np.square(4*v*cos_theta - x_) + np.square(4*v*sin_theta - y_) + 1e-8)
    noise_phi = 1 / (x_**2 + y_**2 + 16e-6*v**2)

    GP_v[index] = fit_GP(v_, noise_v, v_target, threshold=0.001)
    GP_w[index] = fit_GP(w_, noise_w, w_target, threshold=0.004)
    GP_phi[index] = fit_GP(phi_, noise_phi, phi_target, threshold=0.005)



np.save("GP_v.npy", GP_v)
np.save("GP_w.npy", GP_w)
np.save("GP_phi.npy", GP_phi)

print("v:", round(np.sum(GP_v[:, -1]) / 1.28, 2))
print("w:", round(np.sum(GP_w[:, -1]) / 1.28, 2))
print("phi:", round(np.sum(GP_phi[:, -1]) / 1.28, 2))


