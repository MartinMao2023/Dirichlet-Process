import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.path as mpath


budget = 60

case = 3

# Plot projections
# oracle_projections = np.load("oracle_projections.npy")
rte_projections = np.load(f"rte_projections_{case}.npy")
xy_projections = np.load(f"xy_projections_{case}.npy")
arc_projections = np.load(f"arc_projections_{case}.npy")

# orcale_mean = np.mean(oracle_projections, axis=0)
# orcale_std = np.std(oracle_projections, axis=0)
rte_mean = np.mean(rte_projections, axis=0)
rte_std = np.std(rte_projections, axis=0)
xy_mean = np.mean(xy_projections, axis=0)
xy_std = np.std(xy_projections, axis=0)
arc_mean = np.mean(arc_projections, axis=0)
arc_std = np.std(arc_projections, axis=0)


plt.fill_between(np.arange(budget + 1), rte_mean - rte_std, rte_mean + rte_std, color="#ff7f0e", alpha=0.2)

plt.fill_between(np.arange(budget + 1), xy_mean - xy_std, xy_mean + xy_std, color='#2ca02c', alpha=0.2)

plt.fill_between(np.arange(budget + 1), arc_mean - arc_std, arc_mean + arc_std, color='#9467bd', alpha=0.2)
plt.plot(np.arange(budget + 1), rte_mean, color="#ff7f0e", label="RTE", linewidth=2)
plt.plot(np.arange(budget + 1), xy_mean, color='#2ca02c', label="Collab (XY-bases)", linewidth=2)
plt.plot(np.arange(budget + 1), arc_mean, color='#9467bd', label="Collab (arc-bases)", linewidth=2, linestyle="-.")
plt.plot(np.arange(budget + 1), np.zeros(budget + 1)+55.38624, color='#d62728', linestyle="--")
plt.xlabel("step taken", fontsize=16)
plt.ylabel("effective travelled distance", fontsize=16)
# plt.arrow(48, 35, 9, 18, width=0.3, color="black")
# plt.text(40, 30, "Reached exit", fontsize=14)
plt.arrow(5, 47, 2, 6, width=0.3, color="black")
plt.text(0, 39, "Reached \n exit", fontsize=14)
plt.legend(fontsize=14)
plt.show()
# plt.savefig(f"navigation_results_{case}.pdf", dpi=500)












