import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath


budget = 60

# Plot trajectory
# oracle_trajectory = np.load("oracle_trajectory.npy")
rte_trajectory = np.load("rte_trajectory.npy")
xy_trajectory = np.load("xy_trajectory.npy")
arc_trajectory = np.load("arc_trajectory.npy")

rte_hits = np.load("rte_hits.npy")
xy_hits = np.load("xy_hits.npy")
arc_hits = np.load("arc_hits.npy")


circle_1 = mpath.Path.circle(radius=2)
circle_2 = mpath.Path.circle(radius=0.6)
circle_3 = mpath.Path.circle(radius=0.45)
circle_4 = mpath.Path.circle(radius=0.3)
circle_5 = mpath.Path.circle(radius=0.15)
verts = np.concatenate([circle_1.vertices, circle_2.vertices, circle_3.vertices, circle_4.vertices, circle_5.vertices])
codes = np.concatenate([circle_1.codes, circle_2.codes, circle_3.codes, circle_4.codes, circle_5.codes])
marker = mpath.Path(verts, codes)


# print(len(oracle_trajectory))
# print(len(rte_trajectory))
# print(len(xy_trajectory))
# print(len(arc_trajectory))


# plt.plot(rte_trajectory.T[0], rte_trajectory.T[1], marker=marker, markersize=14, markerfacecolor='#ff7f0e', 
#          markeredgecolor='black', markeredgewidth=1, label="RTE", color="black", linewidth=2)

# plt.plot(xy_trajectory.T[0], xy_trajectory.T[1], marker=marker, markersize=14, markerfacecolor='#2ca02c', 
#          markeredgecolor='black', markeredgewidth=1, label="Collab (XY-bases)", color="black", linewidth=2)

# plt.plot(arc_trajectory.T[0], arc_trajectory.T[1], marker=marker, markersize=14, markerfacecolor='#9467bd', 
#          markeredgecolor='black', markeredgewidth=1, label="Collab (arc-bases)", color="black", linewidth=2)


fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(7)
ax.plot(np.linspace(-2, 22.5, 16), -np.ones(16)*2.5, c="black", linewidth=4)
ax.plot(np.linspace(-2, 20, 16), np.ones(16)*2.5, c="black", linewidth=4)
ax.plot(np.linspace(-2, 20, 16), np.ones(16)*12.5, c="black", linewidth=4)
ax.plot(np.linspace(-2, 22.5, 16), np.ones(16)*15, c="black", linewidth=4)
ax.plot(np.ones(16)*20, np.linspace(2.5, 12.5, 16), c="black", linewidth=4)
ax.plot(np.ones(16)*22.5, np.linspace(-2.5, 15, 16), c="black", linewidth=4)





plt.plot(rte_trajectory.T[0], rte_trajectory.T[1], color="black", linewidth=1.5)
plt.plot(xy_trajectory.T[0], xy_trajectory.T[1], color="black", linewidth=1.5)
plt.plot(arc_trajectory.T[0], arc_trajectory.T[1], color="black", linewidth=1.5)

for hit_data in (rte_hits, xy_hits, arc_hits):
    for hit in hit_data:
        plt.plot(hit.T[0], hit.T[1], color="black", linewidth=1.5, alpha=0.3, linestyle="--")


plt.scatter(rte_trajectory.T[0], rte_trajectory.T[1], marker=marker, s=200, c='#ff7f0e', 
            edgecolors='black', linewidths=1, label="RTE", alpha=0.9)
plt.scatter(rte_hits[:, 1, 0], rte_hits[:, 1, 1], marker=marker, s=200, c='#ff7f0e', 
            edgecolors='black', linewidths=1, alpha=0.3)


plt.scatter(xy_trajectory.T[0], xy_trajectory.T[1], marker=marker, s=200, c='#2ca02c', 
            edgecolors='black', linewidths=1, label="Collab (XY-bases)", alpha=0.9)
plt.scatter(xy_hits[:, 1, 0], xy_hits[:, 1, 1], marker=marker, s=200, c='#2ca02c', 
            edgecolors='black', linewidths=1, alpha=0.3)

plt.scatter(arc_trajectory.T[0], arc_trajectory.T[1], marker=marker, s=200, c='#9467bd', 
            edgecolors='black', linewidths=1, label="Collab (arc-bases)", alpha=0.9)
plt.scatter(arc_hits[:, 1, 0], arc_hits[:, 1, 1], marker=marker, s=200, c='#9467bd', 
            edgecolors='black', linewidths=1, alpha=0.3)





plt.scatter(0, 0, marker=marker, s=200, c='cyan', 
            edgecolors='black', linewidths=1)

plt.arrow(-1, -1.2, 0.4, 0.45, width=0.1, color="black")
plt.text(-2, -1.9, "Start", fontsize=16)

plt.plot(np.zeros(8), np.linspace(11.5, 16, 8), color='#d62728', linestyle="--")
plt.arrow(3.5, 10, -2.5, 1.5, width=0.1, color="black")
plt.text(2.5, 9, "Finish line", fontsize=16)

legend = plt.legend(loc=(0.5, 0.4), fontsize=14)
legend.get_frame().set_edgecolor("none")
ax.axis("off")
plt.xlim((-3.5, 23.5))
plt.ylim((-3.5, 15.5))
plt.show()
# plt.savefig("trajectories.png", dpi=500)

