import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt



filtered_baseline = np.load("filtered_baseline.npy")
original_repertoire = np.load("repertoire.npy")

# first_indexes = np.ones(len(filtered_baseline))
# second_indexes = np.ones(len(filtered_baseline))

girds = np.zeros((32, 32), dtype=np.float32)


# print(np.min(filtered_baseline[:, 1]), np.max(filtered_baseline[:, 1]))
for index, (x, y) in enumerate(filtered_baseline[:, :2]):
    second_index = int(np.clip((x + 4.8) / 0.3, 0, 31))
    first_index = int(np.clip((4.8 - y) / 0.3, 0, 31))
    girds[first_index, second_index] = original_repertoire[first_index, second_index, 26]


# original_size = np.sum(np.where(original_repertoire[:, :, 26].reshape(-1, 1) != 0 ,1, 0))
# print(original_size)
# filtered_size = len(filtered_baseline)
# print(filtered_size)



# # original repertoire
# fig, ax = plt.subplots()

# grid_image = ax.imshow(original_repertoire[:, :, 26])
# ax.set_xticks(np.linspace(0, 31, 7))
# ax.set_yticks(np.linspace(0, 31, 7))

# xlabels = np.linspace(-4.8, 4.8, 7)
# ylabels = np.linspace(4.8, -4.8, 7)
# ax.set_xticklabels([str(round(i, 1)) for i in xlabels], fontsize=14)
# ax.set_yticklabels([str(round(i, 1)) for i in ylabels], fontsize=14)
# plt.colorbar(grid_image)
# # plt.savefig("original_map_elites.pdf", dpi=500)
# plt.show()





# # filtered repertoire
# fig, ax = plt.subplots()

# grid_image = ax.imshow(girds)
# ax.set_xticks(np.linspace(0, 31, 7))
# ax.set_yticks(np.linspace(0, 31, 7))

# xlabels = np.linspace(-4.8, 4.8, 7)
# ylabels = np.linspace(4.8, -4.8, 7)
# ax.set_xticklabels([str(round(i, 1)) for i in xlabels], fontsize=14)
# ax.set_yticklabels([str(round(i, 1)) for i in ylabels], fontsize=14)
# plt.colorbar(grid_image)
# # plt.savefig("filtered_map_elites.pdf", dpi=500)
# plt.show()



# print(filtered_baseline.shape)

# c = np.abs(a - b)
# print(np.mean(np.mean(c)), np.max(c))




