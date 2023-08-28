import numpy as np
import matplotlib.pyplot as plt

genotypes = np.load("filtered_repertoire.npy")
baseline = np.load("filtered_baseline.npy")

indexes = {}

for index, (i, j) in enumerate(baseline[:, :2]):
    if 0 not in indexes:
        if 3.75 < i < 4.25 and -0.25 < j < 0.25:
            indexes[0] = index

    if 1 not in indexes:
        if 1.75 < i < 2.5 and 3.21 < j < 3.71:
            indexes[1] = index

    if 2 not in indexes:
        if -2.25 < i < -1.75 and 3.21 < j < 3.71:
            indexes[2] = index

    if 3 not in indexes:
        if -4.25 < i < -3.75 and -0.5 < j < 0.5:
            indexes[3] = index

    if 4 not in indexes:
        if 1.75 < i < 2.25 and -3.71 < j < -3.21:
            indexes[4] = index
    
    if 5 not in indexes:
        if -2.25 < i < -1.75 and -3.71 < j < -3.21:
            indexes[5] = index


# print(indexes)
indexes = np.array([i for i in indexes.values()], dtype=np.int32)

# xy = baseline[indexes, :2]

# plt.scatter(xy.T[0], xy.T[1])
# plt.show()

sub_genotypes = genotypes[indexes]
# print(sub_genotypes.shape)
np.save("selected_genotypes.npy", sub_genotypes)




