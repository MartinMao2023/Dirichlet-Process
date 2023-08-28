import gym
import QDgym_extended
import datetime
from multiprocessing import Pool
import numpy as np
import scipy.ndimage as ndimage
from crossover import crossover
from repertoire import Repertoire
import matplotlib.pyplot as plt
from modifier import modify
import time



def decode_genotype(genotype):
    """genotype: shape (24,) array, 3 parameters for each of the 8 motors.
    each motor genotype in the form of (duty_cycle, phase, scale).

    duty_cycle : [0, 1)
    phase: [0, 1)
    scale: [0, 1]
    """
    
    targets = -np.ones((100, 8), dtype=np.float32)
    phase = np.int32(genotype[1::3] * 100)
    for index, duty_cycle in enumerate(genotype[0::3]):
        start = phase[index]
        end = int(duty_cycle*101) + start
        targets[start: end, index] = 1
        if end > 100:
            targets[: end-100, index] = 1

    guassian_kernel = np.array([0.0008880585, 0.0015861066, 0.00272177, 0.00448744, 0.007108437, 0.010818767, 
                                0.015820118, 0.022226436, 0.03000255, 0.03891121, 0.048486352, 0.058048703, 
                                0.0667719, 0.073794365, 0.078357555, 0.07994048, 0.078357555, 0.073794365, 
                                0.0667719, 0.058048703, 0.048486352, 0.03891121, 0.03000255, 0.022226436, 
                                0.015820118, 0.010818767, 0.007108437, 0.00448744, 0.00272177, 0.0015861066, 
                                0.0008880585], dtype=np.float32)

    targets = ndimage.convolve1d(targets, guassian_kernel, axis=0, mode="wrap")
    
    scales = np.float32(genotype[2::3])
    return targets, scales



def eval_genotypes(genotypes):
    """genotypes in the form of N x 24 array.

    results in the form of N x 3 array,
    each row in the form of [x, y, fitness]

    return the tuple of (genotypes, results)
    """

    # Build the env and ready the constant vectors 
    # env = gym.make("QDAntOmnidirectionalBulletEnv-v0")

    env = gym.make("QDCustomAntBulletEnv-v0", robot_file="custom_ant.xml")

    kds = np.array([0.072, 0.082, 0.072, 0.082, 0.072, 0.082, 0.072, 0.082], dtype=np.float32)
    kps = np.array([1.43, 1.637, 1.43, 1.637, 1.43, 1.637, 1.43, 1.637], dtype=np.float32)
    constants = np.array([0.0, 1.857176, 0.0, -1.857176, 0.0, -1.857176, 0.0, 1.857176], dtype=np.float32)

    # Pre-allocate the matrices
    positions = np.zeros(8, dtype=np.float32)
    speeds = np.zeros(8, dtype=np.float32)
    results = np.zeros((len(genotypes), 3), dtype=np.float32)

    for i, genotype in enumerate(genotypes):
        targets, scales = decode_genotype(genotype) # decode the genotypes
        env.reset()
        done = False
        n = 0

        # Evaluation loop
        while not done:
            target = targets[n % 100] # cycle of 100 steps (1 second)
            for j, joint in enumerate(env.ordered_joints): # read the joints status
                positions[j], speeds[j] = joint.current_position()
            action = np.clip(constants + scales*target - positions*kps - kds*speeds, -1, 1) # PID positional control

            state, reward, done, info = env.step(action)
            n += 1
            if n == 400 and not done: # if survived 4 seconds
                results[i, :] = env.robot_body.pose().xyz()
                # results[i, 2] = -env.tot_reward
                results[i, 2] = env.tot_reward
                break

    env.close()

    return genotypes, results



if __name__ == "__main__":
    starting_time = datetime.datetime.now()
    iterations = 1000

    # original_repertoire = np.load("repertoire.npy")
    # print(original_repertoire.shape, original_repertoire.dtype)
    # indexes = np.load("filtered_indexes.npy")
    # print(indexes.dtype)

    # new_repertore = np.zeros(original_repertoire.shape, dtype=np.float32)
    # for i, j in zip(indexes, np.load("filtered_repertoire.npy").reshape(-1, 24)):
    #     print(np.sum(np.abs(original_repertoire.reshape(-1, 28)[i, :24] - j)))
    # original_repertoire.reshape(-1, 28):






    repertoire = Repertoire(bins=(32, 32), 
                            bounds=((-4.8, 4.8), (-4.8, 4.8)),
                            path="repertoire.npy", 
                            inherit=True)
    

    # repertoire = Repertoire()
    # log_period = 10

    ############################
    ##   initiate genotypes   ##
    ############################


    # corresponding_fitness = np.zeros(8)

    # modify(5, 1.0)

    # for iteration in range(iterations):
    #     results = []
    #     pool = Pool(processes=10)
        
    #     ##########################################
    #     ##   segament genotypes to 16 batches   ##
    #     ##########################################

    #     parents = repertoire.sample_parents(mode="uniform")
    #     off_springs = crossover(parents)

    #     for off_spring in off_springs:
    #         results.append(pool.apply_async(eval_genotypes, args=(off_spring,)))

    #     pool.close()
    #     pool.join()


    #     ########################
    #     ##   select parents   ##
    #     ########################

    #     # genotypes = np.concatenate([parents] + [res.get()[0] for res in results], axis=0)
    #     genotypes = np.concatenate([res.get()[0] for res in results], axis=0)
    #     outcomes = np.concatenate([res.get()[1] for res in results], axis=0)
    #     # fitness = np.concatenate([corresponding_fitness, outcomes[:, -1]])

    #     # indexes = np.argsort(fitness)[-8:]
    #     # parents = genotypes[indexes]
    #     # corresponding_fitness = fitness[indexes]

    #     repertoire.update(genotypes, outcomes)

    #     if (iteration + 1) % log_period == 0:
    #         ################################
    #         ##   record current results   ##
    #         ################################
    #         repertoire.save()
        


    ending_time = datetime.datetime.now()
    # print("total eval:", sum([res.get() for res in results]))

    # distances = np.concatenate([res.get()[1] for res in results], axis=0)
    # min_dis = np.min(distances[:, 2])
    # max_dis = np.max(distances[:, 2])
    # print("evaluation amount:", len(distances))
    # print(round(max_dis, 4), round(min_dis, 4))


    # print("max distance:", np.max(fitness))
    # print("min of max:", np.min(fitness))
    # print("best individual:", end="  [")
    # for i in parents[-1]:
    #     print(round(i, 6), end=", ")
    # print(']')

    # plt.figure()
    plt.imshow(repertoire.outcomes[:, :, 2])
    plt.colorbar()
    # plt.savefig("fitness.png", dpi=300)
    plt.show()

    # time.sleep(1.0)

    # plt.figure()
    # plt.imshow(repertoire.outcomes[:, :, 3])
    # plt.colorbar()
    # plt.savefig("visits.png", dpi=300)

    print("time cost:", round((ending_time - starting_time).total_seconds()-1, 4) , "seconds")





