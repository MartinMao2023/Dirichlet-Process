import numpy as np
import math


class Repertoire:
    def __init__(self, 
                 bins=(25, 25), 
                 bounds=((-5, 5), (-5, 5)), 
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




# a = np.ones((20, 20))
# b = np.zeros((20, 20))
# plt.imshow(a)
# plt.colorbar()
# plt.savefig("a.png", dpi=300)


# import matplotlib.pyplot as plt
# repertoire = Repertoire(bins=(32, 32), 
#                         bounds=((-4.8, 4.8), (-4.8, 4.8)),
#                         path="repertoire.npy", 
#                         inherit=False)
# repertoire.outcomes[:, :, 2] *= -1
# print(np.max(repertoire.outcomes[:, :, 2]))
# repertoire.save()
# print(len(repertoire.occupied_cells))

# plt.imshow(repertoire.outcomes[:, :, 2])
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(b)
# plt.colorbar()
# plt.savefig("b.png", dpi=300)

