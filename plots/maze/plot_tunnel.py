import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# plt.figure(figsize=(10, 8))
fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(7)

ax.add_patch(Rectangle(xy=(-2.1, 12.5), width=2.1, height=2.5, fc="#2ca02c", ec=None, alpha=0.4, linewidth=2))
ax.plot(np.linspace(-2, 22.5, 16), -np.ones(16)*2.5, c="black", linewidth=4)
ax.plot(np.linspace(-2, 20, 16), np.ones(16)*2.5, c="black", linewidth=4)
ax.plot(np.linspace(-2, 20, 16), np.ones(16)*12.5, c="black", linewidth=4)
ax.plot(np.linspace(-2, 22.5, 16), np.ones(16)*15, c="black", linewidth=4)
ax.plot(np.ones(16)*20, np.linspace(2.5, 12.5, 16), c="black", linewidth=4)
ax.plot(np.ones(16)*22.5, np.linspace(-2.5, 15, 16), c="black", linewidth=4)

ax.add_patch(Rectangle(xy=(0, -1), width=1, height=2, fc="#ff7f0e", ec=None, alpha=0.4, linewidth=2))

guide_x = [np.linspace(2, 20, 16)] + [20.41 + 0.84*np.sin(np.linspace(0, 0.5*np.pi, 8))] + [np.ones(16)*21.25]
guide_y = [np.zeros(16)] + [0.84 - 0.84*np.cos(np.linspace(0, 0.5*np.pi, 8))] + [np.linspace(0.84, 12.91, 16)]
guide_x = guide_x + [21.25 - 0.8 + 0.84*np.cos(np.linspace(0, 0.5*np.pi, 8))] + [np.linspace(21.25-0.84, 2, 16)]
guide_y = guide_y + [12.91 + 0.84*np.sin(np.linspace(0, 0.5*np.pi, 8))] + [np.ones(16)*13.75]

guide_x = np.concatenate(guide_x)
guide_y = np.concatenate(guide_y)

ax.scatter(0, -10, color="#2ca02c", marker="s", s=200, alpha=0.5, label="Exit")
ax.scatter(0, -10, color="#ff7f0e", marker="s", s=200, alpha=0.5, label="Start area")
ax.plot(guide_x, guide_y, color="#9467bd", linewidth=2, linestyle="--", label="Suggested path")

ax.quiver(1.5, 0, 1, 0, angles="xy", scale_units="xy", scale=0.6, color="#9467bd")
ax.quiver(2, 13.75, -1, 0, angles="xy", scale_units="xy", scale=0.6, color="#9467bd")
ax.quiver(21.25, 6.25, 0, 1, angles="xy", scale_units="xy", scale=0.6, color="#9467bd")

arrow_dict = {"facecolor": "w",
              "edgecolor": "#7f7f7f",
              "linewidth": 2, 
              "linestyle": "--",
              "arrowstyle": '<->'}

ax.plot(np.linspace(-4, -3, 4), np.ones(4)*2.5, linewidth=2, color="#7f7f7f")
ax.plot(np.linspace(-4, -3, 4), -np.ones(4)*2.5, linewidth=2, color="#7f7f7f")
ax.annotate('', xy=(-3.5,2), xytext=(-3.5,-2), arrowprops=arrow_dict)


ax.plot(np.zeros(4), np.linspace(-4, -3, 4), linewidth=2, color="#7f7f7f")
ax.plot(np.ones(4)*22.5, np.linspace(-4, -3, 4), linewidth=2, color="#7f7f7f")
ax.annotate('', xy=(0.5, -3.5), xytext=(22, -3.5), arrowprops=arrow_dict)


ax.plot(np.zeros(4), np.linspace(3, 4, 4), linewidth=2, color="#7f7f7f")
ax.annotate('', xy=(0.5, 3.5), xytext=(19.5, 3.5), arrowprops=arrow_dict)


ax.plot(np.linspace(23, 24, 4), np.ones(4)*15, linewidth=2, color="#7f7f7f")
ax.plot(np.linspace(23, 24, 4), -np.ones(4)*2.5, linewidth=2, color="#7f7f7f")
ax.annotate('', xy=(23.5, -2), xytext=(23.5, 14.5), arrowprops=arrow_dict)

ax.annotate('', xy=(-1, 3), xytext=(-1, 12), arrowprops=arrow_dict)

ax.text(10, -4.5, "22.5", fontsize=14)
ax.text(9, 4, "20", fontsize=14)
ax.text(24, 6.25, "17.5", fontsize=14)
ax.text(-2.5, 7.5, "10", fontsize=14)
ax.text(-4.5, 0, "5", fontsize=14)

# ax.legend(loc=(0.2, 0.55), fontsize=16)
legend = plt.legend(loc=(0.53, 0.62), fontsize=14)
legend.get_frame().set_edgecolor("none")
ax.axis("off")
plt.xlim((-5, 25))
plt.ylim((-5, 16))
plt.tight_layout()
# plt.show()
plt.savefig("tunnel_floorplan.pdf", dpi=500)






