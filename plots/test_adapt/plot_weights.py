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
plt.tick_params(labelsize=18)
# plt.xlim((-3, 23.5))
plt.ylim((0, 1))
if case_index == 4:
    plt.title("new env", fontsize=24)
else:
    plt.ylabel("posterior weight", fontsize=24)
    plt.title(f"env {case_index+1}", fontsize=24)
plt.tight_layout()
plt.savefig(f"weight_{case_index}.pdf", dpi=500)
# plt.show()







