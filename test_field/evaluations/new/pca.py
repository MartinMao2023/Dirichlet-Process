import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt


def pca_xy():
    base_x = np.load("base_x.npy")
    base_y = np.load("base_y.npy")

    cov_x = np.cov(base_x.T)
    cov_y = np.cov(base_y.T)
    S_x, U_x = lg.eig(cov_x)
    U_x = np.float64(U_x)
    S_y, U_y = lg.eig(cov_y)
    U_y = np.float64(U_y)

    transformed_x = base_x @ U_x / np.sqrt(S_x)
    transformed_y = base_y @ U_y / np.sqrt(S_y)

    return transformed_x, transformed_y
    

def pca_vwphi():
    base_v = np.load("base_v.npy")
    base_w = np.load("base_w.npy")
    base_phi = np.load("base_phi.npy")

    cov_v = np.cov(base_v.T)
    cov_w = np.cov(base_w.T)
    cov_phi = np.cov(base_phi.T)
    S_v, U_v = lg.eig(cov_v)
    U_v = np.float64(U_v)
    S_w, U_w = lg.eig(cov_w)
    U_w = np.float64(U_w)
    S_phi, U_phi = lg.eig(cov_phi)
    U_phi = np.float64(U_phi)

    transformed_v = base_v @ U_v / np.sqrt(S_v)
    transformed_w = base_w @ U_w / np.sqrt(S_w)
    transformed_phi = base_phi @ U_phi / np.sqrt(S_phi)

    return transformed_v, transformed_w, transformed_phi



# import os
# print(os.getcwd())


transformed_x, transformed_y = pca_xy()
print(transformed_x.shape)


np.save("coor_x.npy", transformed_x)
np.save("coor_y.npy", transformed_y)

transformed_v, transformed_w, transformed_phi = pca_vwphi()
print(transformed_v.shape)
np.save("coor_v.npy", transformed_v)
np.save("coor_w.npy", transformed_w)
np.save("coor_phi.npy", transformed_phi)



