from util import Repertoire, decode_genotype, modify, calculate_arc
import numpy as np
import gym
import QDgym_extended
from multiprocessing import Pool



def extract_genotypes(path):
    repertoire = Repertoire(path=path)
    policy_num = len(repertoire.occupied_cells)
    genotypes = np.zeros((policy_num, 24), dtype=np.float32)

    for index, (i, j) in enumerate(repertoire.occupied_cells):
        genotypes[index, :] = repertoire.genotypes[i, j]
        
    return genotypes




def eval_repertoire(genotypes, strength=0.9, damage=0):
    strength = np.ones(8) * strength
    env = gym.make("QDCustomAntBulletEnv-v0", robot_file="custom_ant.xml")
    kds = np.array([0.072, 0.082, 0.072, 0.082, 0.072, 0.082, 0.072, 0.082], dtype=np.float32)
    kps = np.array([1.43, 1.637, 1.43, 1.637, 1.43, 1.637, 1.43, 1.637], dtype=np.float32)
    constants = np.array([0.0, 1.857176, 0.0, -1.857176, 0.0, -1.857176, 0.0, 1.857176], dtype=np.float32)
    positions = np.zeros(8, dtype=np.float32)
    speeds = np.zeros(8, dtype=np.float32)

    results = np.zeros((len(genotypes), 5), dtype=np.float32)
    trajectory_data = np.zeros((400, 2), dtype=np.float32)
    t = np.linspace(0, 4, 100, dtype=np.float64)

    for index, genotype in enumerate(genotypes):
        
        targets, scales = decode_genotype(genotype)

        env.reset()
        # Evaluation loop
        for n in range(420):
            target = targets[n % 100] # cycle of 100 steps (1 second)

            if damage == 1:
                target[1] = -1
                strength[1] = 1
            elif damage == 2:
                target[3] = 1
                strength[3] = 1
            elif damage == 3:
                target[5] = 1
                strength[5] = 1
            elif damage == 4:
                target[7] = -1
                strength[7] = 1

            for j, joint in enumerate(env.ordered_joints): # read the joints status
                positions[j], speeds[j] = joint.current_position()
            action = np.clip(constants + scales*target - positions*kps - kds*speeds, -1, 1)*strength # PID positional control

            env.step(action)

            if n >= 20:
                x, y, _ = env.robot_body.pose().xyz()
                trajectory_data[n-20] = x, y
        
        trajectory_data -= trajectory_data[0]
        results[index, :2] = trajectory_data[-1]
        x, y = np.mean(trajectory_data.reshape(-1, 4, 2), axis=1).T

        v, w, phi = calculate_arc(np.float64(x), np.float64(y), t)
        results[index, 2:] = v, w, phi


    return results.copy()



def job(genotypes, strength, damage, index):
    result = eval_repertoire(genotypes, 
                             strength=strength, 
                             damage=damage)
    return index, result



if __name__ == "__main__":
    # genotypes = extract_genotypes("repertoire.npy")
    genotypes = np.load("filtered_repertoire.npy")


    pool = Pool(processes=10)
    results = []
    # friction = np.random.uniform(0.7, 1.0)
    # strength = np.random.uniform(0.7, 1.0)
    dmg = 4
    friction = 0.77
    strength = 0.82
    modify(5, friction)

    for i in range(16):
        sub_genotypes = genotypes[i*25: i*25+25].copy()
        results.append(pool.apply_async(job, args=(sub_genotypes, strength, dmg, i)))

    pool.close()
    pool.join()

    fragements = [res.get() for res in results]
    fragements.sort(key=lambda x: x[0])
    combined_results = np.concatenate([x[1] for x in fragements], axis=0)

    np.save(rf"final_eval/f={round(friction, 2)}_s={round(strength, 2)}_dmg{dmg}.npy", combined_results)


    # for dmg in range(5):
    #     for j in range(6):
    #         pool = Pool(processes=10)
    #         results = []
    #         friction = np.random.uniform(0.7, 1.0)
    #         strength = np.random.uniform(0.7, 1.0)
    #         # friction = 1
    #         # strength = 1
    #         modify(5, friction)

    #         for i in range(16):
    #             sub_genotypes = genotypes[i*48: i*48+48].copy()
    #             results.append(pool.apply_async(job, args=(sub_genotypes, strength, dmg, i)))
    
    #         pool.close()
    #         pool.join()

    #         fragements = [res.get() for res in results]
    #         fragements.sort(key=lambda x: x[0])
    #         combined_results = np.concatenate([x[1] for x in fragements], axis=0)

    #         np.save(rf"evaluations/new/f={round(friction, 2)}_s={round(strength, 2)}_dmg{dmg}.npy", combined_results)
    #         # np.save(rf"evaluations/new/baseline_results.npy", combined_results)






# genotypes = extract_genotypes("repertoire.npy")
# np.save("genotypes.npy", genotypes)
# print(len(genotypes))

# results = eval_repertoire(genotypes, friction=1.0, strength=1.0)
# np.save("baseline_results.npy", results)

# results = np.load("baseline_results.npy")

# x, y = results[:, :2].T

# import matplotlib.pyplot as plt


# plt.scatter(x, y)
# plt.show()


# plt.hist(results[:, 2])
# plt.show()

# v, w, phi = results[:, -3:].T
# final_theta = w*4
# R = v / w
# x_final = R * (np.sin(final_theta + phi) - np.sin(phi))
# y_final = R * (np.cos(phi) - np.cos(final_theta + phi))

# original_dis = np.sqrt(x**2 + y**2)
# dis = np.sqrt((x - x_final)**2 + (y - y_final)**2)
# print("dis", np.mean(dis), np.std(dis))
# print("percentage", np.mean(dis / (original_dis + 1e-6)), np.mean(dis) / np.mean(original_dis + 1e-6))

# print(np.mean(results[:, -2])*57.296, np.std(results[:, -2])*57.296)






