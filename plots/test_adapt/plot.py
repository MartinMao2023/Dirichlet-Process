import numpy as np
import matplotlib.pyplot as plt

case_index = 0


fig, ax = plt.subplots()
rte_data = np.load(f"rte_case{case_index}.npy")
rte_means = np.mean(rte_data, axis=0)
rte_stds = np.std(rte_data, axis=0)


# Case 0 to 4
xy_data = np.load(f"xy_case{case_index}.npy")
xy_means = np.mean(xy_data, axis=0)
xy_stds = np.std(xy_data, axis=0)
arc_data = np.load(f"arc_case{case_index}.npy")
arc_means = np.mean(arc_data, axis=0)
arc_stds = np.std(arc_data, axis=0)

ax.fill_between(np.arange(26), rte_means-rte_stds, rte_means+rte_stds, alpha=0.2)
ax.plot(np.arange(26), rte_means, label="RTE")
ax.fill_between(np.arange(26), xy_means-xy_stds, xy_means+xy_stds, color='#2ca02c', alpha=0.2)
ax.plot(np.arange(26), xy_means, color='#2ca02c', label="Collab (XY-bases)")
ax.fill_between(np.arange(26), arc_means-arc_stds, arc_means+arc_stds, color='#9467bd', alpha=0.2)
ax.plot(np.arange(26), arc_means, color='#9467bd', label="Collab (arc-bases)")
plt.xlabel("step taken", fontsize=14)
plt.ylabel("MSE loss", fontsize=14)
plt.legend(fontsize=14)
plt.show()
# plt.savefig(f"regular_case{case_index}.pdf", dpi=500)





# xy_data = np.load(f"new_xy_case{case_index}.npy")
# xy_means = np.mean(xy_data, axis=0)
# xy_stds = np.std(xy_data, axis=0)
# arc_data = np.load(f"new_arc_case{case_index}.npy")
# arc_means = np.mean(arc_data, axis=0)
# arc_stds = np.std(arc_data, axis=0)

# ax.fill_between(np.arange(26), rte_means-rte_stds, rte_means+rte_stds, color="#ff7f0e", alpha=0.2)
# ax.plot(np.arange(26), rte_means,color="#ff7f0e", label="RTE")
# ax.fill_between(np.arange(26), xy_means-xy_stds, xy_means+xy_stds, color='#2ca02c', alpha=0.2)
# ax.plot(np.arange(26), xy_means, color='#2ca02c', label="Collab (XY-bases)")
# ax.fill_between(np.arange(26), arc_means-arc_stds, arc_means+arc_stds, color='#9467bd', alpha=0.2)
# ax.plot(np.arange(26), arc_means, color='#9467bd', label="Collab (arc-bases)")
# plt.xlabel("step taken", fontsize=14)
# plt.ylabel("MSE loss", fontsize=14)
# plt.legend(fontsize=14)
# # plt.savefig(f"new_case{case_index}.pdf", dpi=500)
# plt.show()



# xy_data = np.load(f"new_long_xy_case{case_index}.npy")
# xy_means = np.mean(xy_data, axis=0)
# xy_stds = np.std(xy_data, axis=0)
# arc_data = np.load(f"new_long_arc_case{case_index}.npy")
# arc_means = np.mean(arc_data, axis=0)
# arc_stds = np.std(arc_data, axis=0)

# ax.fill_between(np.arange(26), rte_means-rte_stds, rte_means+rte_stds, alpha=0.2)
# ax.plot(np.arange(26), rte_means, label="RTE")
# ax.fill_between(np.arange(26), xy_means-xy_stds, xy_means+xy_stds, color='#2ca02c', alpha=0.2)
# ax.plot(np.arange(26), xy_means, color='#2ca02c', label="Collab (XY-bases)")
# ax.fill_between(np.arange(26), arc_means-arc_stds, arc_means+arc_stds, color='#9467bd', alpha=0.2)
# ax.plot(np.arange(26), arc_means, color='#9467bd', label="Collab (arc-bases)")
# plt.xlabel("step taken", fontsize=14)
# plt.ylabel("MSE loss", fontsize=14)
# plt.legend(fontsize=14)
# plt.show()





