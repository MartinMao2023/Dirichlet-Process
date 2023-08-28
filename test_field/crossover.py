import numpy as np

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
