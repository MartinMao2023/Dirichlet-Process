import numpy as np

def crossover(parents):
    permutation = np.random.choice(8, size=(16, 32))*4 + np.tile(np.arange(4), (16, 8))
    origin_index = np.tile(np.arange(32), (16, 1))

    final_index = np.where(np.random.uniform(0, 1, size=(16, 32)) < 0.2286, 
                        permutation, origin_index).reshape(-1)

    large_noise = np.where(np.random.uniform(0, 1, (16, 8)) < 0.5, 1, 0)
    noise = large_noise[:, :, None] * np.random.uniform(-0.2, 0.2, (16, 8, 24)) + np.random.normal(0, 0.1, (16, 8, 24))

    off_springs = parents.reshape(-1, 6)[final_index].reshape(16, 8, 24) + noise

    return np.clip(off_springs, 0, 1-1e-5)
