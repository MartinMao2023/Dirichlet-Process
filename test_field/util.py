import os
import re
import math
import numpy as np
import numpy.linalg as lg
import scipy.ndimage as ndimage
from scipy.optimize import minimize



def crossover(parents):
    permutation = np.random.choice(8, size=(16, 32))*4 + np.tile(np.arange(4), (16, 8))
    origin_index = np.tile(np.arange(32), (16, 1))

    final_index = np.where(np.random.uniform(0, 1, size=(16, 32)) < 0.2286, 
                        permutation, origin_index).reshape(-1)

    large_noise = np.where(np.random.uniform(0, 1, (16, 8)) < 0.6, 1.0, 0)[:, :, None] 
    large_noise = large_noise * np.where(np.random.uniform(0, 1, (16, 8, 24)) < 0.1, 1.0, 0)
    large_noise = large_noise * np.random.uniform(0, 1, (16, 8, 24)) 

    off_springs = np.mod(parents.reshape(-1, 6)[final_index].reshape(16, 8, 24) + large_noise, 1)

    return np.clip(off_springs + np.random.normal(0, 0.1, (16, 8, 24)), 0, 0.9999)



density_pattern = re.compile(r"\$\$density\$\$")
friction_pattern = re.compile(r"\$\$friction\$\$")

xml_path = r"/home/runjun/python_project/pybullet/lib/python3.8/site-packages/pybullet_data/mjcf"


def modify(density, friction):
    density = "{:.2f}".format(density)
    friction = "{:.2f}".format(friction)

    with open(os.path.join(xml_path, "ant(mod).xml"), "r") as file:
        content = file.read()
    
    content = density_pattern.sub(density, content)
    content = friction_pattern.sub(friction, content)

    with open(os.path.join(xml_path, "custom_ant.xml"), "w") as file:
        file.write(content)



def decode_genotype(genotype):
    """genotype: shape (24,) array, 3 parameters for each of the 8 motors.
    each motor genotype in the form of (duty_cycle, phase, scale).

    duty_cycle : [0, 1)
    phase: [0, 1)
    scale: [0, 1]
    """
    
    targets = -np.ones((100, 8), dtype=np.float32)
    phase = np.int32(genotype[1::3] * 100)
    for index, duty_cycle in enumerate(genotype[0::3]):
        start = phase[index]
        end = int(duty_cycle*101) + start
        targets[start: end, index] = 1
        if end > 100:
            targets[: end-100, index] = 1

    guassian_kernel = np.array([0.0008880585, 0.0015861066, 0.00272177, 0.00448744, 0.007108437, 0.010818767, 
                                0.015820118, 0.022226436, 0.03000255, 0.03891121, 0.048486352, 0.058048703, 
                                0.0667719, 0.073794365, 0.078357555, 0.07994048, 0.078357555, 0.073794365, 
                                0.0667719, 0.058048703, 0.048486352, 0.03891121, 0.03000255, 0.022226436, 
                                0.015820118, 0.010818767, 0.007108437, 0.00448744, 0.00272177, 0.0015861066, 
                                0.0008880585], dtype=np.float32)

    targets = ndimage.convolve1d(targets, guassian_kernel, axis=0, mode="wrap")
    
    scales = np.float32(genotype[2::3])
    return targets, scales



def loss(RAB, x, y, weights):
    R, A, B = RAB
    x_A_2 = np.square(x - A)
    y_B_2 = np.square(y - B)
    R_2 = R**2
    loss = R_2 + x_A_2 + y_B_2 - 2*np.sqrt(R_2*(x_A_2 + y_B_2))
    return float(np.mean(loss*weights))



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
    A, B = W[:, 0]/2

    weights = 1/(0.4*t / np.max(t) + 0.6)
    res = minimize(fun=loss, 
                   x0=np.array([math.sqrt(A**2 + B**2), A, B]), 
                   args=(x, y, weights), 
                   method="L-BFGS-B", 
                   bounds=((1e-4, 1e3), (-1e3, 1e3), (-1e3, 1e3)),
                   jac=jac)
    R, A, B = res.x

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
    v = np.abs(R*w)
    phi = math.atan2(-A, B) if w > 0 else math.atan2(A, -B)

    return v, w, phi



class Repertoire:
    def __init__(self, 
                 bins=(32, 32), 
                 bounds=((-4.8, 4.8), (-4.8, 4.8)), 
                 path=None,
                 inherit=True):
        
        self.occupied_cells = []
        self.expansion = {i:(0, 0, 0) for i in range(12)}
        (self.x_min, self.x_max), (self.y_min, self.y_max) = bounds
        self.x_bins, self.y_bins = bins
        self.x_sep = (self.x_max - self.x_min) / self.x_bins
        self.y_sep = (self.y_max - self.y_min) / self.y_bins
        self.genotypes = np.zeros((self.y_bins, self.x_bins, 24), dtype=np.float32)
        self.outcomes = np.zeros((self.y_bins, self.x_bins, 4), dtype=np.float32)

        if path:
            data = np.load(path)
            if inherit:
                assert data.shape == (self.y_bins, self.x_bins, 28)
                self.genotypes = data[:, :, :24]
                self.outcomes = data[:, :, 24:]
                for i in range(self.y_bins):
                    for j in range(self.x_bins):
                        x, y, _, num = self.outcomes[i, j]
                        if num == 0:
                            continue
                        self.occupied_cells.append((i, j))
                        direction = math.atan2(y, x) if y >= 0 else math.atan2(y, x) + 2*math.pi
                        direction = int(direction / 52.36)
                        distance = math.sqrt(x**2 + y**2)
                        _, __, dis = self.expansion[direction]
                        if dis < distance:
                            self.expansion[direction] = (i, j, distance)
            else:
                genotypes = data[:, :, :24].reshape(-1, 24)
                outcomes = data[:, :, 24:-1].reshape(-1, 3)
                self.update(genotypes, outcomes)

            
    def update(self, genotypes, outcomes):
        for genotype, outcome in zip(genotypes, outcomes):
            x, y, fitness = outcome
            if fitness == 0:
                continue
            second_index = int((x - self.x_min) / self.x_sep)
            first_index = int((self.y_max - y) / self.y_sep)
            if not (0 <= first_index < self.y_bins and 0 <= second_index < self.x_bins):
                continue
            _, __, old_fitness, num = self.outcomes[first_index, second_index]
            if num == 0:
                self.outcomes[first_index, second_index, :] = x, y, fitness, 1
                self.genotypes[first_index, second_index] = genotype
                self.occupied_cells.append((first_index, second_index))
            elif old_fitness < fitness:
                self.outcomes[first_index, second_index, :] = x, y, fitness, num + 1
                self.genotypes[first_index, second_index] = genotype
            else:
                continue

            direction = math.atan2(y, x) if y >= 0 else math.atan2(y, x) + 2*math.pi
            direction = int(direction / 52.36)
            distance = math.sqrt(x**2 + y**2)
            _, __, dis = self.expansion[direction]
            if dis < distance:
                self.expansion[direction] = (first_index, second_index, distance)


    def sample_parents(self, sample_number=8, mode="uniform"):
        if len(self.occupied_cells) < sample_number:
            return np.random.uniform(0, 1, (sample_number, 24))
        
        samples = np.zeros((sample_number, 24), dtype=np.float32)
        
        if mode == "uniform":
            sample_indexes = np.random.choice(len(self.occupied_cells), size=sample_number)
            sample_indexes = [self.occupied_cells[i] for i in sample_indexes]

            for index, (i, j) in enumerate(sample_indexes):
                samples[index, :] = self.genotypes[i, j]
        elif mode == "expand":
            candidates = [(i, j) for i, j, d in self.expansion.values() if d > 0]
            sample_indexes = np.random.choice(len(candidates), size=sample_number)
            sample_indexes = [candidates[i] for i in sample_indexes]

            for index, (i, j) in enumerate(sample_indexes):
                samples[index, :] = self.genotypes[i, j]
        else:
            raise Exception(f"unknown mode \"{mode}\"")

        return samples


    def save(self, path="repertoire.npy"):
        data = np.concatenate((self.genotypes, self.outcomes), axis=-1)
        np.save(path, data)



# x = np.linspace(0, 1, 100, dtype=np.float64)
# y = np.linspace(0, 2, 100, dtype=np.float64)
# t = np.linspace(0, 1, 100, dtype=np.float64)

# print(calculate_arc(x, y, t))


