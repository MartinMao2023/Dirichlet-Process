from new_tools import reward, no_collision, projection
import numpy as np
from Adaptor import Adaptor_rte, Combined_adaptor_xy, Combined_adaptor_arc
import math
import matplotlib.pyplot as plt

coor_x = np.load("coor_x.npy")
coor_y = np.load("coor_y.npy")
coor_v = np.load("coor_v.npy")
coor_w = np.load("coor_w.npy")
coor_phi = np.load("coor_phi.npy")
baselines = np.load("filtered_baseline.npy")
case = np.load("f=0.85_s=0.9_dmg1.npy")

budget = 60
np.random.seed(111)

# # Oracle
# oracle_data = [np.zeros((1, 2))]

# state_x = 0
# state_y = 0
# state_theta = 0
# outcomes = np.zeros((len(baselines), 2), dtype=np.float32)

# i = 0
# while i < budget:   
#     R_matrix = np.array([
#         [math.cos(state_theta), -math.sin(state_theta)],
#         [math.sin(state_theta), math.cos(state_theta)]
#     ], dtype=np.float32)

#     predicted_x, predicted_y = case[:, :2].T # damage
#     predicted_x = predicted_x.reshape(-1, 1)
#     predicted_y = predicted_y.reshape(-1, 1)

#     outcomes[:, 0] = predicted_x[:, 0]
#     outcomes[:, 1] = predicted_y[:, 0]
#     outcomes = outcomes @ R_matrix.T
#     outcomes[:, 0] += state_x
#     outcomes[:, 1] += state_y

#     rewards = reward((state_x, state_y), outcomes)

#     next_action_index = np.argmax(rewards)

#     true_x , true_y, true_v, true_w, true_phi = case[next_action_index]
#     new_x = math.cos(state_theta)*true_x - math.sin(state_theta)*true_y + state_x
#     new_y = math.sin(state_theta)*true_x + math.cos(state_theta)*true_y + state_y


#     if no_collision((state_x, state_y), (new_x, new_y)):
#         state_x = new_x
#         state_y = new_y
#         state_theta += 4*true_w
#         oracle_data.append(np.array([[state_x, state_y]]))
#     else:
#         i += 1
    
#     if state_y > 12.91 and state_x < 0:
#         break
#     i += 1


# oracle_data = np.vstack(oracle_data)

# np.save("oracle_trajectory.npy", oracle_data)



# RTE
rte_data = [np.zeros((1, 2))]
rte_hits = []
adaptor_rte = Adaptor_rte(v=0.5)
state_x = 0
state_y = 0
state_theta = 0
outcomes = np.zeros((len(baselines), 2), dtype=np.float32)

i = 0
invalid_moves = []
while i < budget:    
    R_matrix = np.array([
        [math.cos(state_theta), -math.sin(state_theta)],
        [math.sin(state_theta), math.cos(state_theta)]
    ], dtype=np.float32)

    predicted_x, predicted_y = adaptor_rte.predict(baselines[:, :2])

    outcomes[:, 0] = predicted_x[:, 0]
    outcomes[:, 1] = predicted_y[:, 0]
    outcomes = outcomes @ R_matrix.T
    outcomes[:, 0] += state_x
    outcomes[:, 1] += state_y

    rewards = reward((state_x, state_y), outcomes)

    for action_index in np.argsort(-rewards):
        if action_index not in invalid_moves:
            next_action_index = action_index
            break
    else:
        raise Exception("stuck")

    true_x , true_y, true_v, true_w, true_phi = case[next_action_index]
    new_x = math.cos(state_theta)*true_x - math.sin(state_theta)*true_y + state_x
    new_y = math.sin(state_theta)*true_x + math.cos(state_theta)*true_y + state_y

    step_data = np.zeros((2, 3))
    step_data[:, 0] = baselines[next_action_index, 0]
    step_data[:, 1] = baselines[next_action_index, 1]
    step_data[0, 2] = true_x - baselines[next_action_index, 0]
    step_data[1, 2] = true_y - baselines[next_action_index, 1]
    adaptor_rte.read_data(step_data)

    if no_collision((state_x, state_y), (new_x, new_y)):
        state_x = new_x
        state_y = new_y
        state_theta += 4*true_w
        rte_data.append(np.array([[state_x, state_y]]))
        invalid_moves = []
    else:
        invalid_moves.append(next_action_index)
        i += 1
        rte_hits.append(np.array([[[state_x, state_y], [new_x, new_y]]]))
    
    if state_y > 12.91 and state_x < 0:
        break
    i += 1


print("rte hit:", len(rte_hits))
np.save("rte_trajectory.npy", np.vstack(rte_data))





# XY_Adaptor
xy_data = [np.zeros((1, 2))]

adaptor_xy = Combined_adaptor_xy()
adaptor_xy.load("combined_xy")
state_x = 0
state_y = 0
state_theta = 0
outcomes = np.zeros((len(baselines), 2), dtype=np.float32)
xy_hits = []

i = 0
invalid_moves = []
while i < budget:    
    R_matrix = np.array([
        [math.cos(state_theta), -math.sin(state_theta)],
        [math.sin(state_theta), math.cos(state_theta)]
    ], dtype=np.float32)

    predicted_x, predicted_y = adaptor_xy.predict((coor_x, coor_y), baselines.T)

    outcomes[:, 0] = predicted_x[:, 0]
    outcomes[:, 1] = predicted_y[:, 0]
    outcomes = outcomes @ R_matrix.T
    outcomes[:, 0] += state_x
    outcomes[:, 1] += state_y

    rewards = reward((state_x, state_y), outcomes)

    for action_index in np.argsort(-rewards):
        if action_index not in invalid_moves:
            next_action_index = action_index
            break
    else:
        raise Exception("stuck")

    true_x, true_y, true_v, true_w, true_phi = case[next_action_index]
    
    new_x = math.cos(state_theta)*true_x - math.sin(state_theta)*true_y + state_x
    new_y = math.sin(state_theta)*true_x + math.cos(state_theta)*true_y + state_y


    coor = np.vstack((coor_x[next_action_index], coor_y[next_action_index]))
    baseline = baselines[next_action_index].reshape(-1, 1)
    result = case[next_action_index].reshape(-1, 1)
    adaptor_xy.read_data(coor, baseline, result)

    if no_collision((state_x, state_y), (new_x, new_y)):
        state_x = new_x
        state_y = new_y
        state_theta += 4*true_w
        xy_data.append(np.array([[state_x, state_y]]))
        invalid_moves = [] # clear
    else:
        invalid_moves.append(next_action_index)
        xy_hits.append(np.array([[[state_x, state_y], [new_x, new_y]]]))
        i += 1
    

    if state_y > 12.91 and state_x < 0:
        break
    i += 1

print("xy hit:", len(xy_hits))
xy_data = np.vstack(xy_data)
np.save("xy_trajectory.npy", xy_data)





# ARC_Adaptor
arc_data = [np.zeros((1, 2))]

adaptor_arc = Combined_adaptor_arc()
adaptor_arc.load("combined_arc")
state_x = 0
state_y = 0
state_theta = 0
outcomes = np.zeros((len(baselines), 2), dtype=np.float32)
arc_hits = []

i = 0
invalid_moves = []

while i < budget:    
    R_matrix = np.array([
        [math.cos(state_theta), -math.sin(state_theta)],
        [math.sin(state_theta), math.cos(state_theta)]
    ], dtype=np.float32)

    predicted_x, predicted_y = adaptor_arc.predict((coor_v, coor_w, coor_phi), baselines.T)

    outcomes[:, 0] = predicted_x[:, 0]
    outcomes[:, 1] = predicted_y[:, 0]
    outcomes = outcomes @ R_matrix.T
    outcomes[:, 0] += state_x
    outcomes[:, 1] += state_y

    rewards = reward((state_x, state_y), outcomes)

    for action_index in np.argsort(-rewards):
        if action_index not in invalid_moves:
            next_action_index = action_index
            break
    else:
        raise Exception("stuck")

    true_x, true_y, true_v, true_w, true_phi = case[next_action_index]
    
    new_x = math.cos(state_theta)*true_x - math.sin(state_theta)*true_y + state_x
    new_y = math.sin(state_theta)*true_x + math.cos(state_theta)*true_y + state_y


    coor = np.vstack((coor_v[next_action_index], coor_w[next_action_index], coor_phi[next_action_index]))
    baseline = baselines[next_action_index].reshape(-1, 1)
    result = case[next_action_index].reshape(-1, 1)
    adaptor_arc.read_data(coor, baseline, result)

    if no_collision((state_x, state_y), (new_x, new_y)):
        state_x = new_x
        state_y = new_y
        state_theta += 4*true_w
        arc_data.append(np.array([[state_x, state_y]]))
        invalid_moves = [] # clear
    else:
        invalid_moves.append(next_action_index)
        i += 1
        arc_hits.append(np.array([[[state_x, state_y], [new_x, new_y]]]))
    

    if state_y > 12.91 and state_x < 0:
        break
    i += 1

print("arc hit:", len(arc_hits))
arc_data = np.vstack(arc_data)
np.save("arc_trajectory.npy", arc_data)


if rte_hits:
    rte_hits = np.concatenate(rte_hits, axis=0)
    # print(rte_hits.shape)
    np.save("rte_hits.npy", rte_hits)


if xy_hits:
    xy_hits = np.concatenate(xy_hits, axis=0)
    # print(xy_hits.shape)
    np.save("xy_hits.npy", xy_hits)


if arc_hits:
    arc_hits = np.concatenate(arc_hits, axis=0)
    # print(arc_hits.shape)
    np.save("arc_hits.npy", arc_hits)



