import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from collections.abc import Iterable

prior_amount = 128 # number of sampled priors
np.random.seed(123)

# prior sampling
row_xy = np.random.uniform(-0.99, 0.99, prior_amount)
mu_x, mu_y = np.random.normal(0, 5, (2, prior_amount))
sigma_x, sigma_y = np.random.uniform(0.5, 2, (2, prior_amount))
inverse_matrices = np.zeros((prior_amount, 2, 2))
inverse_matrices[:, 0, 0] = 1 / sigma_x**2
inverse_matrices[:, 1, 1] = 1 / sigma_y**2
inverse_matrices[:, 0, 1] = -row_xy / (sigma_x*sigma_y)
inverse_matrices[:, 1, 0] = -row_xy / (sigma_x*sigma_y)
inverse_matrices /= (1 - row_xy**2)[:, None, None]

# data generation, 6 clusters
np.random.seed(123)
total_data_size = 1000

row_xy_data = np.random.uniform(-0.99, 0.99, 6)
mu_x_data, mu_y_data = np.random.normal(0, 5, (2, 6))
sigma_x_data, sigma_y_data = np.random.uniform(0.5, 2, (2, 6))
p = (0.2, 0.15, 0.05, 0.1, 0.1, 0.4)

total_data = np.zeros((total_data_size, 2), dtype=np.float32)
cumulated_size = 0

visualize = False


for i in range(6):
    data_size = int(p[i] * total_data_size)
    data_x = np.random.normal(mu_x_data[i], sigma_x_data[i], data_size)
    data_y = np.random.normal(mu_y_data[i] + row_xy_data[i]*sigma_y_data[i]/sigma_x_data[i]*(data_x - mu_x_data[i]),
                              sigma_y_data[i]*np.sqrt(1 - row_xy_data[i]**2), data_size)

    if visualize:
        plt.scatter(data_x, data_y, label=f"cluster {i+1}")

    total_data[cumulated_size: cumulated_size+data_size, 0] = data_x
    total_data[cumulated_size: cumulated_size+data_size, 1] = data_y
    cumulated_size += data_size


if visualize:
    plt.legend()
    plt.show()

# plt.scatter(total_data[:, 0], total_data[:, 1])
# plt.show()




class Experience:
    def __init__(self, data):
        self.data = data.reshape(-1, 2)
        power = self.data @ (inverse_matrices @ self.data.T).reshape(-1, 2).T
        self.log_likelihoods = -0.5*power - np.log(sigma_x*sigma_y*np.sqrt(1 - row_xy**2))
        self.log_likelihoods = self.log_likelihoods.reshape(-1)

        self.id = 0
        self.cluster_id = 0



class Cluster:
    def __init__(self, ids, archive, threshold=40):
        self.id = 0
        self.threshold = threshold
        self.archive = archive
        if isinstance(ids, int):
            self.members = {ids: archive.experiences[ids]}
        else:
            self.members = {}
            for id in ids:
                self.members[id] = archive.experiences[id]
        
        self.log_likelihoods = np.zeros(prior_amount, dtype=np.float32)
        self.weights = np.zeros(prior_amount, dtype=np.float32)
        self.size = len(self.members)
        self.mean = np.zeros((2, 1), dtype=np.float32)
        self.inverse_covariance = np.zeros((2, 2), dtype=np.float32)
        self.det = 1
    

    def update(self):
        self.size = len(self.members)
        if self.size == 0:
            return False

        if self.size >= self.threshold:
            # | <-- fit mean and inverse covariance --> | #
            data = np.zeros((self.size, 2), dtype=np.float32)
            for index, exp in enumerate(self.members.values()):
                data[index] = exp.data
            
            self.mean[:, 0] = np.mean(data, axis=0)
            cov = np.cov(data.T)
            self.det = lg.det(cov)
            self.inverse_covariance[:, :] = lg.inv(cov)
        else:
            log_likelihoods = np.zeros(prior_amount, dtype=np.float32)
            for member in self.members.values():
                log_likelihoods += member.log_likelihoods
            self.log_likelihoods[:] = log_likelihoods
            log_likelihoods -= np.max(log_likelihoods)
            weights = np.exp(log_likelihoods)
            weights /= np.sum(weights)
            self.weights[:] = weights
        return True


    def eval_prob(self, id):
        exp = self.archive.experiences[id]
        if self.size >= self.threshold:
            data = exp.data
            likelihood = 1
            log_factor = -0.5*data @ self.inverse_covariance @ data.T - 0.5*np.log(self.det)
        else:
            log_factor = np.max(exp.log_likelihoods)
            # log_likelihood = exp.log_likelihoods
            if id in self.members:
                if self.size == 1:
                    return 0, None
                log_likelihood = self.log_likelihoods - exp.log_likelihoods
                weights = np.exp(log_likelihood - np.max(log_likelihood))
                weights /= np.sum(weights)
                likelihood = np.sum(weights * np.exp(exp.log_likelihoods - log_factor)) * (self.size - 1)
            else:
                likelihood = np.sum(self.weights * np.exp(exp.log_likelihoods - log_factor)) * self.size

        return log_factor, likelihood
    




class Archive:
    def __init__(self, alpha):
        self.alpha = alpha # concentration parameter
        self.experiences = {}
        self.empty_exp_id = []
        self.next_exp_id = 1

        self.clusters = {}
        self.empty_cluster_id = []
        self.next_cluster_id = 1
    

    def read_experiences(self, exps):
        if not isinstance(exps, Iterable):
            exps = (exps,)

        for exp in exps:
            if self.empty_exp_id:
                id = self.empty_exp_id[0]
                self.empty_exp_id.remove(id)
            else:
                id = self.next_exp_id
                self.next_exp_id += 1
            
            exp.id = id
            self.experiences[id] = exp
            exp.cluster_id = self.generate_cluster(id)
    

    def generate_cluster(self, ids):
        if self.empty_cluster_id:
            id = self.empty_cluster_id[0]
            self.empty_cluster_id.remove(id)
        else:
            id = self.next_cluster_id
            self.next_cluster_id += 1
        
        cluster = Cluster(ids, self)
        cluster.id = id
        self.clusters[id] = cluster
        self.update_cluster(id)
        return id


    def update_cluster(self, id):
        exist = self.clusters[id].update()
        if not exist:
            self.remove_cluster(id)
            # return
        
        # cluster = self.clusters[id]
        # cluster.size = len(cluster.members)

        # log_likelihoods = np.zeros(prior_amount, dtype=np.float32)
        # for member in cluster.members.values():
        #     log_likelihoods += member.log_likelihoods
        # cluster.log_likelihoods[:] = log_likelihoods
        # log_likelihoods -= np.max(log_likelihoods)
        # weights = np.exp(log_likelihoods)
        # weights /= np.sum(weights)
        # cluster.weights[:] = weights


    def remove_exp(self, id):
        del self.experiences[id] # unregister this experience
        self.empty_exp_id.append(id) # record this empty id 


    def remove_cluster(self, id):
        del self.clusters[id] # unregister this cluster
        self.empty_cluster_id.append(id) # record this empty id 


    def gibbs_loop(self):
        denominator = self.alpha + len(self.experiences) - 1
        for exp_id, exp in self.experiences.items():
            likelihoods = np.zeros(len(self.clusters)+1, dtype=np.float32)
            log_factors = np.zeros(len(self.clusters)+1, dtype=np.float32)
            cluster_ids = np.zeros(len(self.clusters)+1, dtype=np.int32)

            log_factors[:] = np.max(exp.log_likelihoods)
            exp_likelihoods = np.exp(exp.log_likelihoods - np.max(exp.log_likelihoods))
            likelihoods[0] = np.mean(exp_likelihoods) * self.alpha / denominator

            for index, (cluster_id, cluster) in enumerate(self.clusters.items()):
                cluster_ids[index+1] = cluster_id
                log_factor, likelihood = cluster.eval_prob(exp_id)

                if likelihood is None:
                    likelihoods[index+1] = likelihoods[0]
                    likelihoods[0] = 0
                else:
                    likelihoods[index+1] = likelihood / denominator
                    log_factors[index+1] = log_factor

                # if cluster_id == exp.cluster_id:
                #     if len(cluster.members) == 1:
                #         likelihoods[index+1] = likelihoods[0]
                #         likelihoods[0] = 0
                #     else:
                #         new_log_likelihoods = cluster.log_likelihoods - exp.log_likelihoods
                #         new_log_likelihoods -= (np.max(new_log_likelihoods) - 60)
                #         new_weights = np.exp(new_log_likelihoods)
                #         new_weights /= np.sum(new_weights)
                #         likelihoods[index+1] = np.sum(new_weights * exp_likelihoods) * (len(cluster.members)-1) / denominator
                # else:
                #     likelihoods[index+1] = np.sum(cluster.weights * exp_likelihoods) * len(cluster.members) / denominator
            
            likelihoods *= np.exp(log_factors - np.max(log_factors))
            likelihoods /= np.sum(likelihoods)
            




            selected_cluster = cluster_ids[np.random.choice(len(self.clusters)+1, p=likelihoods)]
            if selected_cluster and selected_cluster != exp.cluster_id:
                new_cluster = self.clusters[selected_cluster]
                old_cluster = self.clusters[exp.cluster_id]

                new_cluster.members[exp_id] = exp
                self.update_cluster(selected_cluster)

                del old_cluster.members[exp_id]
                self.update_cluster(exp.cluster_id)

                exp.cluster_id = selected_cluster
            elif selected_cluster == 0:
                self.generate_cluster(exp_id)



archive = Archive(alpha=1.0)

exps = [Experience(data) for data in total_data]
archive.read_experiences(exps)

iter = 30
for i in range(iter):
    archive.gibbs_loop()
    print(f"iter {1+i}:", len(archive.clusters))

print()
for i, j in archive.clusters.items():
    print("Size:", len(j.members), "\t", "Max weight:", np.max(j.weights))
    if j.size > 1:
        xy = np.zeros((j.size, 2))
        for index, exp in enumerate(j.members.values()):
            xy[index, :] = exp.data[0]

        plt.scatter(xy.T[0], xy.T[1], label=f"cluster {index+1}")

plt.legend()
plt.show()


# for i, j in archive.clusters.items():
#     if len(j.members) > 100:
#         print(np.max(j.weights))

# print(archive.next_cluster_id, archive.next_exp_id)

# np.random.seed()

# a = Experience(total_data[0])
# archive.read_experiences(a)





