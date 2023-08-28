import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# plt.figure(figsize=(10, 8))
fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(8)

ax.plot(np.ones(10), np.linspace(0.5, 7.5, 10), c="black", linewidth=4)
ax.plot(np.linspace(1, 2, 10), np.ones(10)*7.5, c="black", linewidth=4)
ax.plot(np.linspace(1, 2, 10), np.ones(10)*0.5, c="black", linewidth=4)


ax.plot(np.ones(10)*9, np.linspace(0.5, 7.5, 10), c="black", linewidth=4)
ax.plot(np.linspace(8, 9, 10), np.ones(10)*7.5, c="black", linewidth=4)
ax.plot(np.linspace(8, 9, 10), np.ones(10)*0.5, c="black", linewidth=4)


# ax.add_patch(Rectangle(xy=(6, 1), width=2, height=2, fc="w", ec="black"))
ax.add_patch(Rectangle(xy=(2, 5), width=2, height=2, fc="w", ec="black", linewidth=2))
ax.text(2.25, 6.5, "Cov 1", dict(fontsize=16, fontweight="bold"))

ax.add_patch(Rectangle(xy=(4.25, 3.55), width=1.2, height=1.2, fc="w", ec="black", linewidth=2))
ax.text(4.35, 4.25, "Cov 2", dict(fontsize=16, fontweight="bold"))

ax.add_patch(Rectangle(xy=(6.25, 1), width=1.75, height=1.75, fc="w", ec="black", linewidth=2))
ax.text(6.5, 2.25, "Cov n", dict(fontsize=16, fontweight="bold"))


ax.plot(np.linspace(5.6, 6.1, 10), np.linspace(3.4, 2.9, 10), c="black", linestyle=":", linewidth=3)
arrow_dict = {"facecolor": "darkorange",
              "shrink": 0.1, 
              "width": 8, 
              "headwidth": 17,
              "headlength": 20}

ax.annotate("", xy=(4.1, 6.6), xytext=(5, 6.6), arrowprops=arrow_dict)
ax.annotate("", xy=(5.55, 4.4), xytext=(6.45, 4.4), arrowprops=arrow_dict)
ax.annotate("", xy=(6.15, 2.4), xytext=(5.25, 2.4), arrowprops=arrow_dict)
# ax.annotate("experience 1", xy=(1, 1), xytext=(3, 3.5), arrowprops={"arrowstyle": "->", "color": "red"})

ax.text(5.2, 6.5, "Experience 1", dict(fontsize=16, fontweight="bold"))
ax.text(6.65, 4.3, "Experience 2", dict(fontsize=16, fontweight="bold"))
ax.text(2.9, 2.3, "Experience n", dict(fontsize=16, fontweight="bold"))

ax.axis("off")
# ax.add_patch(Rectangle(xy=(1, 4), width=3, height=3, fc="w", ec="black"))
# ax.add_patch(Rectangle(xy=(1, 4), width=3, height=3, fc="w", ec="black"))
plt.xlim((0, 10))
plt.ylim((0, 8))
plt.show()






