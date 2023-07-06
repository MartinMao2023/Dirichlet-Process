import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from numpy import random
import math
from datetime import datetime



x = np.linspace(0, 3.1416, 200, dtype=np.float32)
sinx = np.sin(x)
# approx_sinx = x/2 - np.power(x/2, 3) / 6 
# # + np.power(x, 9) / 362880
approx_sinx = x - 0.19895929 * np.power(x, 2.43713223)
# approx_sinx = x - 0.1584253 * np.power(x, 2.85328487)

# plt.plot(x, sinx)
# plt.plot(x, approx_sinx)
# plt.plot(x*57.3, (approx_sinx + 1e-10) / (sinx + 1e-10))
# plt.show()

# plt.plot(x, np.sin(x))
# plt.plot(x, x - x**3 / 6 + x**5/120 - x**7/5040)
# # plt.plot(x, np.sin(x/2)**3)
# # plt.plot(x, )
# plt.show()


def calculate_arc(x, y, t):
    """ This function is very hard to understand.
    For now just accept it.
    """

    t1 = t.copy()
    t2 = np.power(t1, 2.43713223)
    row = np.sqrt(x**2 + y**2)

    t1_square = np.sum(t1**2)
    t2_square = np.sum(t2**2)
    t1_t2 = np.sum(t1 * t2)
    row_t1 = np.sum(row * t1)
    row_t2 = np.sum(row * t2)
    ratio = (row_t2*t1_square - row_t1*t1_t2) / (row_t1*t2_square - row_t2*t1_t2)

    rough_w = 0 if ratio > 0 else math.pow(-13.61*ratio, 0.6958302) + 1e-10

    weight = np.where(t*rough_w > np.pi, 0, 1)
    t2 = np.power(t1, 2.8532847)

    t1_square = np.sum(weight * t1**2)
    t2_square = np.sum(weight * t2**2)
    t1_t2 = np.sum(weight * t1 * t2)
    row_t1 = np.sum(weight * row * t1)
    row_t2 = np.sum(weight * row * t2)

    det = t1_square*t2_square - t1_t2**2
    v = (row_t1*t2_square - row_t2*t1_t2) / det
    ratio = (row_t2*t1_square - row_t1*t1_t2) / (det * v)

    w = 1e-5 if ratio > 0 else math.pow(-22.8071*ratio, 0.53958) 

    cos_beta = x / (row + 1e-10)
    sin_beta = y / (row + 1e-10)
    cost = np.cos(0.5*t1*w)
    sint = np.sin(0.5*t1*w)

    sin_weight = row**2 / (cos_beta**2 * sin_beta**2 + 1 + sin_beta**4 - 2*sin_beta**2 + 0.01)
    cos_weight = row**2 / (cos_beta**2 * sin_beta**2 + 1 + cos_beta**4 - 2*cos_beta**2 + 0.01)

    cos_det = np.sum(cos_weight*sint**2)*np.sum(cos_weight*cost**2) - np.sum(cos_weight*sint*cost)**2
    sin_det = np.sum(sin_weight*sint**2)*np.sum(sin_weight*cost**2) - np.sum(sin_weight*sint*cost)**2

    sinphi = (np.sum(sin_weight*sin_beta*cost)*np.sum(sin_weight*sint**2) - np.sum(sin_weight*sin_beta*sint)*np.sum(sin_weight*sint*cost)) / sin_det
    sec_sinphi = (np.sum(cos_weight*cos_beta*sint)*np.sum(cos_weight*cost**2) - np.sum(cos_weight*cos_beta*cost)*np.sum(cos_weight*sint*cost)) / cos_det

    sec_cosphi = (np.sum(sin_weight*sin_beta*sint)*np.sum(sin_weight*cost**2) - np.sum(sin_weight*sin_beta*cost)*np.sum(sin_weight*sint*cost)) / sin_det
    cosphi = (np.sum(cos_weight*cos_beta*cost)*np.sum(cos_weight*sint**2) - np.sum(cos_weight*cos_beta*sint)*np.sum(cos_weight*sint*cost)) / cos_det

    phi = math.atan2(sinphi, cosphi)

    if abs(cosphi*sec_cosphi) >= abs(sec_sinphi*sinphi):
        w = w if cosphi*sec_cosphi > 0 else -w
    else:
        w = w if sinphi*sec_sinphi < 0 else -w
        
    return v, w, phi





R = 100
angle = math.pi * 0.5
arc = R * angle
print("arc:", round(arc, 2))
print("real v:", round(arc / 10, 4))
print("real w:", round(-angle / 10, 4))
total_time = 10 
phi = math.pi * 2/ 3

ox = R * math.sin(phi)
oy = -R * math.cos(phi)


sample_amount = 64
t = np.arange(sample_amount) / sample_amount * total_time
# print(np.max(t))

alpha = -np.arange(sample_amount) * (angle / sample_amount) + phi + math.pi / 2
fluctuation_cycle = 10
fluctuation_amplitude = 0.2
sideways_amplitude = 0.1
add_fluctuation = True

if add_fluctuation:
    alpha += angle*fluctuation_amplitude*np.sin(2*math.pi*t/total_time*fluctuation_cycle)
    plt.plot(t, alpha)
    plt.show()

    # px = np.cos(alpha)*R + ox 
    # py = np.sin(alpha)*R + oy


    px = np.cos(alpha)*(R + arc*sideways_amplitude*np.sin(2*math.pi*t/total_time*fluctuation_cycle)) + ox 
    py = np.sin(alpha)*(R + arc*sideways_amplitude*np.sin(2*math.pi*t/total_time*fluctuation_cycle)) + oy
else:
    px = np.cos(alpha)*R + ox 
    py = np.sin(alpha)*R + oy


px += np.random.normal(0, 1, sample_amount)
py += np.random.normal(0, 1, sample_amount)

if add_fluctuation:
    plt.plot(t, np.sqrt(px**2 + py**2))
    plt.show()

v, w, estimated_phi = calculate_arc(px, py, t)


print("v:", round(v, 4))
print("w:", round(w, 4))
print("phi:", round(estimated_phi*57.3, 4))
# print(round(v, 4), round(w, 4))
# print("R:", v / w)





