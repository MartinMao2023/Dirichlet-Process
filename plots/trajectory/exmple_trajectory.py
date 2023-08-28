import numpy as np
import matplotlib.pyplot as plt
from util import calculate_arc
import math

fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(12.5)

color_dict = {0: "#1f77b4", 1: "#ff7f0e", 
              2: "#2ca02c", 3: "#d62728", 
              4: "#9467bd", 5: "#8c564b"}


t = np.linspace(0, 4, 100)


# curve 1:
index = 1
v = 0.5
w = 30 / 57.296
phi = 90 / 57.296

x = v / w * (np.sin(t*w + phi) - math.sin(phi))
y = v / w * (math.cos(phi) - np.cos(t*w + phi))
dx = math.cos(phi + 4 * w)
dy = math.sin(phi + 4 * w)
ax.quiver(x[-1], y[-1], dx, dy, angles="xy", scale_units="xy", scale=15, color=color_dict[index])
ax.plot(x, y, color=color_dict[index], label="curve 1", linewidth=3)



# curve 2:
index = 2
v = 0.25
w = 0.01 / 57.296
phi = 90 / 57.296

x = v / w * (np.sin(t*w + phi) - math.sin(phi))
y = v / w * (math.cos(phi) - np.cos(t*w + phi))

dx = math.cos(phi + 4 * w)
dy = math.sin(phi + 4 * w)
ax.quiver(x[-1], y[-1], dx, dy, angles="xy", scale_units="xy", scale=15, color=color_dict[index])
ax.plot(x, y, color=color_dict[index], label="curve 2", linewidth=3)



# curve 3:
index = 3
v = 0.5
w = 30 / 57.296
phi = 0 / 57.296

x = v / w * (np.sin(t*w + phi) - math.sin(phi))
y = v / w * (math.cos(phi) - np.cos(t*w + phi))
dx = math.cos(phi + 4 * w)
dy = math.sin(phi + 4 * w)
ax.quiver(x[-1], y[-1], dx, dy, angles="xy", scale_units="xy", scale=15, color=color_dict[index])
ax.plot(x, y, color=color_dict[index], label="curve 3", linewidth=3)

ax.axis("off")
plt.xlim((-1.5, 1))
plt.ylim((-0.1, 1.5))
plt.legend(fontsize=24)
# plt.show()
plt.savefig("example_curves.pdf", dpi=500)


# # original_trajectories 1
# original_trajectories = np.load("trajectories.npy")
# original_arcs = np.load("arcs.npy")
# plt.figure(figsize=(6, 5.4))

# for index, (trajectory, arc) in enumerate(zip(original_trajectories, original_arcs)):
#     x, y = trajectory
#     v, w, phi = arc

#     dx =  np.mean(x[375: 400]) - np.mean(x[350: 375])
#     dy =  np.mean(y[375: 400]) - np.mean(y[350: 375])

#     dx_ = dx / math.sqrt(dx**2 + dy**2)
#     dy_ = dy / math.sqrt(dx**2 + dy**2)
    
#     theta = np.linspace(0, 4, 100)*w + phi
#     arc_x = v / w * (np.sin(theta) - math.sin(phi))
#     arc_y = v / w * (math.cos(phi) - np.cos(theta))

#     plt.plot(x, y, c=color_dict[index], label=f"action {index+1}", linewidth=2)
#     plt.quiver(x[375], y[375], dx_, dy_, angles="xy", scale_units="xy", scale=2, color=color_dict[index])
#     plt.plot(arc_x, arc_y, c=color_dict[index], linestyle="--")

# plt.xlim((-5, 5))
# plt.ylim((-4.5, 4.5))
# plt.legend()
# plt.show()
# plt.savefig("original_trajectories_1.pdf", dpi=500)



