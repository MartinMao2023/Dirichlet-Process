import numpy as np
import matplotlib.pyplot as plt


prior_amount = 64 # number of sampled priors


class Experience:
    def __init__(self, data_x, data_y):

        # data size checking
        if len(data_x) != len(data_y):
            raise Exception("Inconsistent data dim")
        elif len(data_y) == 0:
            raise Exception("No input data")
        else:
            self.data_x = data_x
            self.data_y = data_y
            self.data_dim = len(data_y)
            self.data_size = len(data_y[0])
            
        self.self_prob = 0 # waiting to be calculated
        self.cluster_index = 0 # no cluster assigned initially
        self.prior_likelihoods = np.zeros(prior_amount, dtype=np.float32)
        pass


    # def fit_mean(self):
        
    #     # return weights, bias
    #     pass


    # def fit_kernel(self):

    #     # return length_scale, noise
    #     pass


    # def calculate_self_prob(self):

    #     # weight, bias = self.fit_mean()
    #     # length_scale, noise = self.fit_kernel()
    #     covariance_matrix = np.eye(self.data_size)
    #     # return self.prob
    #     pass


a = np.array([1,2,3, 4, 5])
print(np.diag(a))


class Cluster:
    def __init__(self, archive, id):
        self.archive = archive
        self.id = id
        self.exp_set = set()
    

    def update(self):
        pass
    
    
    def add(self, id):
        self.exp_set.add(id)

        self.update()
        pass
    

    def remove(self, id):
        self.exp_set.discard(id)

        self.update()
        pass


    def fit_mean(self):
        pass


    def fit_kernel(self):
        pass



class Archive:
    def __init__(self):
        self.prior_amount = prior_amount
        self.dim_num = 3
        self.dim_name = ("v", "w", "phi")

        self.experience_dict = {}
        self.clusters = {}

        pass


