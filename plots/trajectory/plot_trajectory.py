import numpy as np
import matplotlib.pyplot as plt
from util import calculate_arc
import math


color_dict = {0: "#1f77b4", 1: "#ff7f0e", 
              2: "#2ca02c", 3: "#d62728", 
              4: "#9467bd", 5: "#8c564b"}


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

#     plt.plot(x, y, c=color_dict[index], label=f"policy {index+1}", linewidth=2)
#     plt.quiver(x[375], y[375], dx_, dy_, angles="xy", scale_units="xy", scale=2, color=color_dict[index])
#     plt.plot(arc_x, arc_y, c=color_dict[index], linestyle="--")

# plt.xlim((-5, 5))
# plt.ylim((-4.5, 4.5))
# plt.legend()
# # plt.show()
# plt.savefig("original_trajectories_1.pdf", dpi=500)



# # original_trajectories 2
# original_trajectories = np.load("trajectories_2.npy")
# original_arcs = np.load("arcs_2.npy")
# plt.figure(figsize=(6, 5.4))

# for index, (trajectory, arc) in enumerate(zip(original_trajectories, original_arcs)):
#     x, y = trajectory
#     v, w, phi = arc

#     dx =  np.mean(x[375: 400]) - np.mean(x[350: 375])
#     dy =  np.mean(y[375: 400]) - np.mean(y[350: 375])

#     dx /= math.sqrt(dx**2 + dy**2)
#     dy /= math.sqrt(dx**2 + dy**2)
    
#     theta = np.linspace(0, 4, 100)*w + phi
#     arc_x = v / w * (np.sin(theta) - math.sin(phi))
#     arc_y = v / w * (math.cos(phi) - np.cos(theta))

#     plt.plot(x, y, c=color_dict[index], label=f"action {index+1}", linewidth=2)
#     plt.quiver(x[375], y[375], dx, dy, angles="xy", scale_units="xy", scale=2, color=color_dict[index])
#     plt.plot(arc_x, arc_y, c=color_dict[index], linestyle="--")

# plt.xlim((-5, 5))
# plt.ylim((-4.5, 4.5))
# plt.legend()
# plt.show()



# # damaged case
# original_trajectories = np.load("new_trajectories.npy")
# plt.figure(figsize=(6, 5.4))

# for index, trajectory in enumerate(original_trajectories):
#     x, y = trajectory
#     v, w, phi = calculate_arc(np.mean(x.reshape(-1, 4), axis=-1), 
#                               np.mean(y.reshape(-1, 4), axis=-1),
#                               np.linspace(0, 4, 100))

#     dx =  np.mean(x[375: 400]) - np.mean(x[350: 375])
#     dy =  np.mean(y[375: 400]) - np.mean(y[350: 375])

#     dx /= math.sqrt(dx**2 + dy**2)
#     dy /= math.sqrt(dx**2 + dy**2)
    
#     theta = np.linspace(0, 4, 100)*w + phi
#     arc_x = v / w * (np.sin(theta) - math.sin(phi))
#     arc_y = v / w * (math.cos(phi) - np.cos(theta))

#     plt.plot(x, y, c=color_dict[index], label=f"policy {index+1}", linewidth=2)
#     plt.quiver(x[375], y[375], dx, dy, angles="xy", scale_units="xy", scale=2, color=color_dict[index])
#     plt.plot(arc_x, arc_y, c=color_dict[index], linestyle="--")

# plt.xlim((-5, 5))
# plt.ylim((-4.5, 4.5))
# plt.legend()
# plt.show()



# slippery case
original_trajectories = np.load("new_trajectories2.npy")
plt.figure(figsize=(6, 5.4))

for index, trajectory in enumerate(original_trajectories):
    x, y = trajectory
    v, w, phi = calculate_arc(np.mean(x.reshape(-1, 4), axis=-1), 
                              np.mean(y.reshape(-1, 4), axis=-1),
                              np.linspace(0, 4, 100))

    dx =  np.mean(x[375: 400]) - np.mean(x[350: 375])
    dy =  np.mean(y[375: 400]) - np.mean(y[350: 375])

    dx /= math.sqrt(dx**2 + dy**2)
    dy /= math.sqrt(dx**2 + dy**2)
    
    theta = np.linspace(0, 4, 100)*w + phi
    arc_x = v / w * (np.sin(theta) - math.sin(phi))
    arc_y = v / w * (math.cos(phi) - np.cos(theta))

    plt.plot(x, y, c=color_dict[index], label=f"policy {index+1}", linewidth=2)
    plt.quiver(x[375], y[375], dx, dy, angles="xy", scale_units="xy", scale=2, color=color_dict[index])
    plt.plot(arc_x, arc_y, c=color_dict[index], linestyle="--")

plt.xlim((-5, 5))
plt.ylim((-4.5, 4.5))
plt.legend()
plt.show()



