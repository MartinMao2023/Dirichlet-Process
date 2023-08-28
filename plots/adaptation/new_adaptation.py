from Archives import Archive_xy, Archive_arc
import numpy as np

archive_xy = Archive_xy()
archive_xy.load("archive_xy")

coor_x = np.load("coor_x.npy")
coor_y = np.load("coor_y.npy")
coor_v = np.load("coor_v.npy")
coor_w = np.load("coor_w.npy")
coor_phi = np.load("coor_phi.npy")
baseline = np.load("filtered_baseline.npy")

new_data = np.load("f=0.77_s=0.82_dmg4.npy")
# np.random.seed(111)
np.random.seed(11)
xy_exps = []
arc_exp = []
for i in range(2):
    selected_steps = np.random.choice(len(new_data), 35)

    x_coors = coor_x[selected_steps]
    y_coors = coor_y[selected_steps]
    x_targets = (new_data[selected_steps, 0] - baseline[selected_steps, 0]).reshape(-1, 1)
    y_targets = (new_data[selected_steps, 1] - baseline[selected_steps, 1]).reshape(-1, 1)

    xy_exps.append({"x": np.hstack((x_coors, x_targets)), "y": np.hstack((y_coors, y_targets))})


    exp = {}
    v_targets = (new_data[selected_steps, 2] - baseline[selected_steps, 2]).reshape(-1, 1)
    v_coors = coor_v[selected_steps]
    w_targets = (new_data[selected_steps, 3] - baseline[selected_steps, 3]).reshape(-1, 1)
    w_coors = coor_w[selected_steps]
    phi_targets = np.mod((new_data[selected_steps, 4] - baseline[selected_steps, 4] + 3*np.pi), 2*np.pi).reshape(-1, 1) - np.pi
    phi_coors = coor_phi[selected_steps]

    v = new_data[selected_steps, 2]
    w = new_data[selected_steps, 3]
    phi = new_data[selected_steps, 4]
    sin_theta = np.sin(w*4 + phi)
    cos_theta = np.cos(w*4 + phi)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    x_ = v/w*(sin_theta - sin_phi)
    y_ = v/w*(cos_phi - cos_theta)

    noise_v = v**2 / (x_**2 + y_**2 + 1e-6) + 1e-6
    noise_w = (w**2 + 0.01) / (np.square(4*v*cos_theta - x_) + np.square(4*v*sin_theta - y_) + 1e-8)
    noise_phi = 1 / (x_**2 + y_**2 + 16e-6*v**2)

    exp["v"] = np.hstack((v_coors, noise_v.reshape(-1, 1), v_targets))
    exp["w"] = np.hstack((w_coors, noise_w.reshape(-1, 1), w_targets))
    exp["phi"] = np.hstack((phi_coors, noise_phi.reshape(-1, 1), phi_targets))
    arc_exp.append(exp)


# for file in os.listdir(r"../exps/dmg1"):
#     data = np.load(os.path.join(r"../exps/dmg1", file))
#     exp = {}
#     indexes = np.int32(data[:, 0])
#     x_targets = (data[:, 1] - baseline[indexes, 0]).reshape(-1, 1)
#     # print(baseline[indexes, 0])
#     x_coors = coor_x[indexes]
#     # print(x_targets.shape, x_coors.shape)
#     exp["x"] = np.hstack((x_coors, x_targets))
#     y_targets = (data[:, 2] - baseline[indexes, 1]).reshape(-1, 1)
#     y_coors = coor_y[indexes]
#     exp["y"] = np.hstack((y_coors, y_targets))

#     exps.append(exp)



# next_index = archive_xy.next_exp_id
# print(next_index)

archive_xy.read_experiences(xy_exps)
# print(len(archive_xy.clusters))
for i in range(10):
    archive_xy.gibbs_sweep()


archive_xy.MAP(20)


# print(len(archive_xy.clusters))

adaptor_xy = archive_xy.build_adaptor()
# print(adaptor_xy)
adaptor_xy.save("new_long_adaptor_xy.json")



archive_arc = Archive_arc()
archive_arc.load("archive_arc")

archive_arc.read_experiences(arc_exp)

# for i, j in archive_arc.clusters.items():
#     print(j["w_coef"].shape)

# print(len(archive_xy.clusters))


for i in range(10):
    archive_arc.gibbs_sweep()


archive_arc.MAP(20)

adaptor_arc = archive_arc.build_adaptor()
adaptor_arc.save("new_long_adaptor_arc.json")

