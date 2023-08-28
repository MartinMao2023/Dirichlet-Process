from util import Repertoire, decode_genotype, modify, calculate_arc
import numpy as np
import gym
import QDgym_extended
from multiprocessing import Pool
import datetime
import os



def extract_genotypes(path):
    repertoire = Repertoire(path=path)
    policy_num = len(repertoire.occupied_cells)
    genotypes = np.zeros((policy_num, 24), dtype=np.float32)

    for index, (i, j) in enumerate(repertoire.occupied_cells):
        genotypes[index, :] = repertoire.genotypes[i, j]
        
    return genotypes




def eval_genotypes(genotypes, strength=0.9, damage=0):
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

    env.close()

    return results.copy()



def job(genotypes, strength, damage, index):
    result = eval_genotypes(genotypes, 
                            strength=strength, 
                            damage=damage)
    return index, result



if __name__ == "__main__":
    starting_time = datetime.datetime.now()

    # genotypes = extract_genotypes("repertoire.npy")
    genotypes = np.load("filtered_repertoire.npy")
    repertoire_size = len(genotypes)


    for batch_num in range(128):
        pool = Pool(processes=10)
        sample_index = np.random.choice(repertoire_size, size=64)
        sample_batch = genotypes[sample_index]
        # print(sample_batch.shape)

        friction = np.random.uniform(0.7, 1.0)
        strength = np.random.uniform(0.7, 1.0)
        dmg = np.random.choice(5)
        # dmg = 0
        modify(5, friction)
        
        results = []
        for index, sub_batch in enumerate(sample_batch.reshape(16, 4, 24)):
            results.append(pool.apply_async(job, args=(sub_batch, strength, dmg, index)))
        
        pool.close()
        pool.join()

        fragements = [res.get() for res in results]
        fragements.sort(key=lambda x: x[0])
        combined_results = np.concatenate([x[1] for x in fragements], axis=0)
        combined_results = np.hstack((np.float32(sample_index).reshape(-1, 1), combined_results))
        np.save(os.path.join("evaluations", "prior", f"{batch_num}.npy"), combined_results)

    ending_time = datetime.datetime.now()
    print("time cost:", round((ending_time - starting_time).total_seconds()-1, 4) , "seconds")

