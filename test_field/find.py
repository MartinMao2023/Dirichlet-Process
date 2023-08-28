import os
import sys
import re
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import numpy.linalg as lg

from util import Repertoire, decode_genotype, modify, calculate_arc



# def extract_genotypes(path):
#     repertoire = Repertoire(path=path)
#     policy_num = len(repertoire.occupied_cells)
#     genotypes = np.zeros((policy_num, 24), dtype=np.float32)

#     for index, (i, j) in enumerate(repertoire.occupied_cells):
#         genotypes[index, :] = repertoire.genotypes[i, j]
        
#     return genotypes

# repertoire = extract_genotypes("repertoire.npy")
# baseline = np.load("baseline_results.npy")

# print(repertoire.shape)
# filtered_indexes = np.load("filtered_indexes.npy")

# filter_repertoire = []
# for index in filtered_indexes:
#     filter_repertoire.append(repertoire[index])

# filter_repertoire = np.array(filter_repertoire)
# print(filter_repertoire.shape)

# np.save("filtered_repertoire.npy", filter_repertoire)



# print(np.load("filtered_repertoire.npy").shape)


# filtered_indexes = np.load("filtered_indexes.npy")
# baseline = np.load("baseline_results.npy")
# print(baseline.shape)

# a = baseline[filtered_indexes]
# print(a.shape)

# b = np.load("filtered_baseline.npy")

filtered_baseline = np.load("filtered_baseline.npy")
original_repertoire = np.load("repertoire.npy")

# first_indexes = np.ones(len(filtered_baseline))
# second_indexes = np.ones(len(filtered_baseline))

girds = np.zeros((32, 32), dtype=np.float32)


print(np.min(filtered_baseline[:, 1]), np.max(filtered_baseline[:, 1]))
for index, (x, y) in enumerate(filtered_baseline[:, :2]):
    second_index = int(np.clip((x + 4.8) / 0.3, 0, 31))
    first_index = int(np.clip((4.8 - y) / 0.3, 0, 31))
    girds[first_index, second_index] = original_repertoire[first_index, second_index, 26]
    # first_indexes[index] = first_index
    # second_indexes[index] = second_index


# plt.hist(second_indexes)
# print(np.max(second_indexes))
# plt.show()


fig, ax = plt.subplots()

grid_image = ax.imshow(girds)
# plt.xlim((-4.8, 4.8))
# plt.ylim((-4.8, 4.8))
ax.set_xticks(np.linspace(0, 31, 7))
ax.set_yticks(np.linspace(0, 31, 7))

labels = np.round(np.linspace(-4.8, 4.8, 7), 2)
ax.set_xticklabels([str(i) for i in labels])
ax.set_yticklabels([str(-i) for i in labels])
plt.colorbar(grid_image)
plt.show()


# print(filtered_baseline.shape)

# c = np.abs(a - b)
# print(np.mean(np.mean(c)), np.max(c))




