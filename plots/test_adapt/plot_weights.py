import numpy as np
import matplotlib.pyplot as plt

case_index = 4


fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(8)
rte_data = np.load(f"rte_case{case_index}.npy")
rte_means = np.mean(rte_data, axis=0)
rte_stds = np.std(rte_data, axis=0)


data = np.load(f"weights_{case_index}.npy")
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)

color_dict = {0: '#1f77b4', 
              1: '#ff7f0e', 
              2: '#2ca02c', 
              3: '#d62728', 
              4: '#9467bd', 
              5: '#8c564b', 
              6: '#e377c2', 
              7: '#7f7f7f', 
              8: '#bcbd22', 
              9: '#17becf',
              10: "#ff7f0e"}


# for i, (mean, std) in enumerate(zip(data_mean.T, data_std.T)):
#     ax.fill_between(np.arange(26), mean-std, mean+std, color=color_dict[i], alpha=0.2)


for i, (mean, std) in enumerate(zip(data_mean.T, data_std.T)):
    linestyle = "--" if i == 10 else "-"
    label = "new cluster" if i == 10 else f"cluster {i+1}"
    ax.plot(np.arange(26), mean, color=color_dict[i], linestyle=linestyle, linewidth=2, label=label)


legend = plt.legend(fontsize=14)
legend.get_frame().set_edgecolor("none")
plt.xlabel("step taken", fontsize=24)
plt.ylabel("posterior weight", fontsize=24)
plt.tick_params(labelsize=16)
# plt.xlim((-3, 23.5))
plt.ylim((0, 1))
plt.show()


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



