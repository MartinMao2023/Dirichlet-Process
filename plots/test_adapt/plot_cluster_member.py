import numpy as np
import matplotlib.pyplot as plt


xy_cluster_data = np.load("cluster_data_xy.npy")
arc_cluster_data = np.load("cluster_data_arc.npy")

fig, ax = plt.subplots()



# ax.set_xticks(np.linspace(0, 31, 7))
# ax.set_yticks(np.linspace(0, 31, 7))


# plot xy

xlabels = np.arange(len(xy_cluster_data))
ax.set_xticks(np.arange(len(xy_cluster_data)))
ax.set_xticklabels([str(round(i, 1)) for i in xlabels], fontsize=12)

plt.bar(np.arange(len(xy_cluster_data)), xy_cluster_data[:, 0], label="env 1")
plt.bar(np.arange(len(xy_cluster_data)), xy_cluster_data[:, 1], label="env 2")
plt.bar(np.arange(len(xy_cluster_data)), xy_cluster_data[:, 2], label="env 3")
plt.bar(np.arange(len(xy_cluster_data)), xy_cluster_data[:, 3], label="env 4")
plt.xlabel("cluster index", fontsize=16)
plt.ylabel("episode amount", fontsize=16)
plt.legend(fontsize=14)
# plt.show()
plt.tight_layout()
plt.savefig("xy_cluster_data.pdf" , dpi=500)





# # plot arc
# xlabels = np.arange(len(arc_cluster_data))
# ax.set_xticks(np.arange(len(arc_cluster_data)))
# ax.set_xticklabels([str(round(i, 1)) for i in xlabels], fontsize=12)

# plt.bar(np.arange(len(arc_cluster_data)), arc_cluster_data[:, 0], label="env 1")
# plt.bar(np.arange(len(arc_cluster_data)), arc_cluster_data[:, 1], label="env 2")
# plt.bar(np.arange(len(arc_cluster_data)), arc_cluster_data[:, 2], label="env 3")
# plt.bar(np.arange(len(arc_cluster_data)), arc_cluster_data[:, 3], label="env 4")
# plt.xlabel("cluster index", fontsize=16)
# # plt.ylabel("episode amount", fontsize=16)
# plt.legend(fontsize=14)
# # plt.show()
# plt.tight_layout()
# plt.savefig("arc_cluster_data.pdf" , dpi=500)


