import numpy as np
from matplotlib import pyplot as plt
from numpy import random
import math
from datetime import datetime
import numpy.linalg as lg
from scipy.optimize import minimize




def loss(RAB, x, y, weights):
    R, A, B = RAB
    x_A_2 = np.square(x - A)
    y_B_2 = np.square(y - B)
    R_2 = R**2
    loss = R_2 + x_A_2 + y_B_2 - 2*np.sqrt(R_2*(x_A_2 + y_B_2))
    return np.mean(loss*weights)



def jac(RAB, x, y, weights):
    R, A, B = RAB
    x_A_2 = np.square(x - A)
    y_B_2 = np.square(y - B)
    denominator = np.sqrt(x_A_2 + y_B_2) + 1e-6
    L_R = np.mean((2*R - 2*denominator)*weights)
    L_A = np.mean(2*(A - x)*(1 - R / denominator)*weights)
    L_B = np.mean(2*(B - y)*(1 - R / denominator)*weights)
    return np.array([L_R, L_A, L_B])



def calculate_arc(x, y, t):
    data_size = len(t)
    row_square = (x**2 + y**2).reshape(-1, 1)
    X_matrix = np.zeros((data_size, 2), dtype=np.float32)
    X_matrix[:, 0] = x
    X_matrix[:, 1] = y
    noise = np.array([[1e-4, 0], [0, 1e-4]], dtype=np.float32)
    W = lg.inv(X_matrix.T @ X_matrix + noise) @ X_matrix.T @ row_square

    ############################
    ##    initial estimate    ##
    ############################
    A, B = W[:, 0]/2

    #########################
    ##    minimize loss    ##
    #########################

    weights = 1/(0.4*t / np.max(t) + 0.6)


    res = minimize(fun=loss, 
                   x0=np.array([math.sqrt(A**2 + B**2), A, B]), 
                   args=(x, y, weights), 
                   method="L-BFGS-B", 
                   bounds=((1e-4, 1e3), (-1e3, 1e3), (-1e3, 1e3)),
                   jac=jac)
    
    R, A, B = res.x

    ###########################
    ##    calculate omega    ##
    ###########################

    xy = np.vstack((x-A, y-B), dtype=np.float32).T
    modulus = np.sqrt(np.sum(xy**2, axis=1))
    xy_ = xy.copy()
    modulus_ = modulus.copy()
    xy_[1:] = xy[:-1]
    modulus_[1:] = modulus[:-1]

    sin_angles = np.clip(np.cross(xy_, xy) / (modulus*modulus_), -1, 1)
    delta_angles = np.arcsin(sin_angles)

    t_ = t.copy()
    t_[1:] = t[:-1]
    delta_t = t - t_

    angle_weights = 1.01 - sin_angles**2


    w =  np.sum(delta_t * delta_angles * angle_weights) / np.sum(delta_t**2 * angle_weights) 


    ###############################
    ##    calculate v and phi    ##
    ###############################

    v = np.abs(R*w)

    phi = math.atan2(-A, B) if w > 0 else math.atan2(A, -B)

    # print("w:", round(w*57.3, 4))
    # print("v:", round(v, 4))
    # print("phi", round(phi*57.3, 4))
    return v, w, phi



if __name__ == "__main__":
    # x = np.linspace(0, 3.1416, 128)
    # x = 4 - np.cos(x)*4
    # # y = np.sqrt(16 - x**2) - 4
    # # y = 4 - np.sqrt(16 - x**2)
    # y = np.sqrt(16 - (x - 4)**2) 
    # # y = 10*x



    # # # plt.plot(x, y)
    # # # plt.show()

    # v, w, phi = calculate_arc(x, y, np.linspace(0, 1, 128))
    # print("w:", round(w*57.296, 4))
    # print("v:", round(v, 4))
    # print("phi", round(phi*57.296, 4))


    angle = math.pi * 1.5    # <--- total angle. In this case it is one and half a circle
    arc = 300    # <--- arc length
    R = arc / angle 
    # print("real R:", round(R, 4))

    total_time = 10    # <--- time for this trajectory
    phi = math.pi * 5 / 12    # <--- initial angle, 75 degree


    sample_amount = 64    # <--- number of samples taken from this trajectory


    t = np.arange(sample_amount) / sample_amount * total_time
    alpha = -np.arange(sample_amount) * (angle / sample_amount) + phi + math.pi / 2

    ox = R * math.sin(phi)
    oy = -R * math.cos(phi)


    fluctuation_cycle = 5    # <--- fluctuate 5 cycles during the trajectory
    fluctuation_amplitude = 0.2    # <--- forward fluctation amplitude is 20% of total angle 
    sideways_amplitude = 0.1    # <--- sideways fluctuation is 10% of arc length


    ## | <---  we add the fluctuations  ---> | ##
    alpha += angle*fluctuation_amplitude*np.sin(2*math.pi*t/total_time*fluctuation_cycle)
    px = np.cos(alpha)*(R + arc*sideways_amplitude*np.sin(2*math.pi*t/total_time*fluctuation_cycle)) + ox 
    py = np.sin(alpha)*(R + arc*sideways_amplitude*np.sin(2*math.pi*t/total_time*fluctuation_cycle)) + oy


    # and also the noise
    px += np.random.normal(0, 5, sample_amount)
    py += np.random.normal(0, 5, sample_amount)

    v, w, phi = calculate_arc(px, py, t)
    # print("w:", round(w*57.296, 4))
    print("w:", round(w, 4))
    print("v:", round(v, 4))
    print("phi", round(phi*57.296, 4))






