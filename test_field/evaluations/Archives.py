import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from Matern import Matern
from Adaptor import Adaptor_xy, Adaptor_arc
import math
import os
import copy
import json


class Archive(metaclass=ABCMeta):
    
    @abstractmethod
    def load_priors(self, paths: dict):
        pass


    @abstractmethod
    def read_experiences(self, exps):
        pass


    @abstractmethod
    def gibbs_sweep(self):
        pass


    @abstractmethod
    def MAP(self):
        pass


    @abstractmethod
    def save(self, dir_path=None):
        pass


    @abstractmethod
    def load(self, dir_path):
        pass



class Archive_xy(Archive):
    def __init__(self, alpha=1.0, threshold=40):
        self.alpha = alpha
        self.kernel = Matern(v=1.5)
        self.exps = {}
        self.next_exp_id = 1
        self.next_cluster_id = 1
        self.empty_cluster_ids = []
        self.empty_exp_ids = []
        self.clusters = {}
        self.threshold = threshold
        self.backup_clusters = {}
        self.x_priors = None
        self.y_priors = None
        self.log_prob = -np.inf

    
    def load_priors(self, paths):
        x_priors = np.load(paths["x"])
        self.x_prior_data = x_priors.copy()
        y_priors = np.load(paths["y"])
        self.y_prior_data = y_priors.copy()

        self.x_priors = []
        for x_prior in x_priors:
            coef = x_prior[:5].reshape(-1, 1)
            length_scale = x_prior[5:10]
            prior_var, noise_ratio = x_prior[10:12]
            noisy = True if x_prior[-1] else False
            self.x_priors.append((noisy, coef, length_scale, prior_var, noise_ratio))
        
        self.y_priors = []
        for y_prior in y_priors:
            coef = y_prior[:5].reshape(-1, 1)
            length_scale = y_prior[5:10]
            prior_var, noise_ratio = y_prior[10:12]
            noisy = True if y_prior[-1] else False
            self.y_priors.append((noisy, coef, length_scale, prior_var, noise_ratio))
    

    def read_experiences(self, exps, ids=None):
        if ids is None and exps:
            self.log_prob = -np.inf

        for id_index, exp in enumerate(exps):
            exp_dict = {}
            x_data = exp["x"]
            y_data = exp["y"]
            exp_dict["data_size"] = len(x_data)
            exp_dict["x_data"] = x_data.copy()
            exp_dict["y_data"] = y_data.copy()

            if not ids is None:
                exp_id = ids[id_index]
            elif self.empty_exp_ids:
                exp_id = self.empty_exp_ids[0]
                del self.empty_exp_ids[0]
            else:
                exp_id = self.next_exp_id
                self.next_exp_id += 1

            exp_dict["id"] = exp_id
            self.exps[exp_id] = exp_dict
            x_loglikelihoods = np.zeros(len(self.x_priors), dtype=np.float32)
            y_loglikelihoods = np.zeros(len(self.y_priors), dtype=np.float32)

            x_coors = x_data[:, :-1]
            x_target = x_data[:, -1].reshape(-1, 1)
            y_coors = y_data[:, :-1]
            y_target = y_data[:, -1].reshape(-1, 1)
            
            for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(self.x_priors):
                diff = x_target - x_coors @ coef
                data_size = len(x_data)
                if noisy:
                    x_loglikelihood = -0.5*(diff / (prior_var*noise_ratio + prior_var)).T @ diff - \
                        0.5*data_size*math.log(prior_var*noise_ratio + prior_var)
                else:
                    cov = self.kernel(x_coors, length_scales=length_scale) + np.eye(data_size)*noise_ratio
                    cov *= prior_var
                    x_loglikelihood = -0.5*(diff.T @ lg.inv(cov) @ diff)
                    sign, log_det = lg.slogdet(cov)
                    log_det = log_det if sign else -100
                    x_loglikelihood -= 0.5*log_det
                x_loglikelihoods[index] = float(x_loglikelihood)

            for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(self.y_priors):
                diff = y_target - y_coors @ coef
                data_size = len(y_data)
                if noisy:
                    y_loglikelihood = -0.5*(diff / (prior_var*noise_ratio + prior_var)).T @ diff - \
                        0.5*data_size*math.log(prior_var*noise_ratio + prior_var)
                else:
                    cov = self.kernel(y_coors, length_scales=length_scale) + np.eye(data_size)*noise_ratio
                    cov *= prior_var
                    y_loglikelihood = -0.5*(diff.T @ lg.inv(cov) @ diff)
                    sign, log_det = lg.slogdet(cov)
                    log_det = log_det if sign else -100
                    y_loglikelihood -= 0.5*log_det
                y_loglikelihoods[index] = float(y_loglikelihood)
            exp_dict["x_loglikelihoods"] = x_loglikelihoods
            exp_dict["y_loglikelihoods"] = y_loglikelihoods
            log_self_prob = np.log(np.mean(
                np.exp(x_loglikelihoods - np.max(x_loglikelihoods))
                )*np.mean(np.exp(y_loglikelihoods - np.max(y_loglikelihoods))))
            log_self_prob += np.max(x_loglikelihoods) + np.max(y_loglikelihoods)
            exp_dict["log_self_prob"] = log_self_prob

            if ids is None:
                exp_dict["cluster"] = self.alloc_cluster(exp_id)
            else:
                exp_dict["cluster"] = 0
        
    

    def drop_exp(self, exp_id):
        exp = self.exps[exp_id]
        self.remove_exp_from_cluster(exp_id, exp["cluster"])
        del self.exps[exp_id]
        self.empty_exp_ids.append(exp_id)
        self.log_prob = -np.inf



    def alloc_cluster(self, exp_id):
        if self.empty_cluster_ids:
            cluster_id = self.empty_cluster_ids[0]
            del self.empty_cluster_ids[0]
        else:
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
        
        cluster_dict = {"id": cluster_id}
        cluster_dict["members"] = {exp_id}
        x_loglikelihoods = self.exps[exp_id]["x_loglikelihoods"].copy()
        y_loglikelihoods = self.exps[exp_id]["y_loglikelihoods"].copy()
        cluster_dict["x_loglikelihoods"] = x_loglikelihoods
        cluster_dict["y_loglikelihoods"] = y_loglikelihoods
        x_weights = np.exp(x_loglikelihoods - np.max(x_loglikelihoods))
        y_weights = np.exp(y_loglikelihoods - np.max(y_loglikelihoods))
        cluster_dict["x_weights"] = x_weights / np.sum(x_weights)
        cluster_dict["y_weights"] = y_weights / np.sum(y_weights)
        cluster_dict["size"] = 1
        cluster_dict["data_size"] = self.exps[exp_id]["data_size"]
        self.clusters[cluster_id] = cluster_dict
        if cluster_dict["data_size"] >= self.threshold:
            cluster_dict["large"] = True
            cluster_dict["x_kernel_index"] = int(np.argmax(x_weights))
            cluster_dict["y_kernel_index"] = int(np.argmax(y_weights))
            x_coef, y_coef = self.fit_coef(cluster_id)
            cluster_dict["x_coef"] = x_coef
            cluster_dict["y_coef"] = y_coef
        else:
            cluster_dict["large"] = False
            cluster_dict["x_coef"] = np.ones((5, 1), dtype=np.float32)
            cluster_dict["y_coef"] = np.ones((5, 1), dtype=np.float32)
            cluster_dict["x_kernel_index"] = 0
            cluster_dict["y_kernel_index"] = 0

        return cluster_id

    

    def add_exp_to_cluster(self, exp_id, cluster_id):
        cluster = self.clusters[cluster_id]
        cluster["members"].add(exp_id)
        self.exps[exp_id]["cluster"] = cluster_id
        x_loglikelihoods = self.exps[exp_id]["x_loglikelihoods"]
        y_loglikelihoods = self.exps[exp_id]["y_loglikelihoods"]
        cluster["x_loglikelihoods"] += x_loglikelihoods
        cluster["y_loglikelihoods"] += y_loglikelihoods
        new_x_loglikelihoods = cluster["x_loglikelihoods"]
        new_y_loglikelihoods = cluster["y_loglikelihoods"]
        x_weights = np.exp(new_x_loglikelihoods - np.max(new_x_loglikelihoods))
        y_weights = np.exp(new_y_loglikelihoods - np.max(new_y_loglikelihoods))
        cluster["x_weights"] = x_weights / np.sum(x_weights)
        cluster["y_weights"] = y_weights / np.sum(y_weights)
        cluster["size"] += 1
        cluster["data_size"] += self.exps[exp_id]["data_size"]

        if cluster["data_size"] >= self.threshold:
            cluster["large"] = True
            cluster["x_kernel_index"] = int(np.argmax(x_weights))
            cluster["y_kernel_index"] = int(np.argmax(y_weights))
            x_coef, y_coef = self.fit_coef(cluster_id)
            cluster["x_coef"] = x_coef
            cluster["y_coef"] = y_coef



    def remove_exp_from_cluster(self, exp_id, cluster_id):
        cluster = self.clusters[cluster_id]
        if cluster["size"] == 1:
            self.empty_cluster_ids.append(cluster_id)
            del self.clusters[cluster_id]
            return

        cluster["members"].discard(exp_id)
        x_loglikelihoods = self.exps[exp_id]["x_loglikelihoods"]
        y_loglikelihoods = self.exps[exp_id]["y_loglikelihoods"]
        cluster["x_loglikelihoods"] -= x_loglikelihoods
        cluster["y_loglikelihoods"] -= y_loglikelihoods
        new_x_loglikelihoods = cluster["x_loglikelihoods"]
        new_y_loglikelihoods = cluster["y_loglikelihoods"]
        x_weights = np.exp(new_x_loglikelihoods - np.max(new_x_loglikelihoods))
        y_weights = np.exp(new_y_loglikelihoods - np.max(new_y_loglikelihoods))
        cluster["x_weights"] = x_weights / np.sum(x_weights)
        cluster["y_weights"] = y_weights / np.sum(y_weights)
        cluster["size"] -= 1
        cluster["data_size"] -= self.exps[exp_id]["data_size"]

        if cluster["data_size"] >= self.threshold:
            cluster["x_kernel_index"] = int(np.argmax(x_weights))
            cluster["y_kernel_index"] = int(np.argmax(y_weights))
            x_coef, y_coef = self.fit_coef(cluster_id)
            cluster["x_coef"] = x_coef
            cluster["y_coef"] = y_coef
        else:
            cluster["large"] = False


    
    def fit_coef(self, cluster_id):
        cluster = self.clusters[cluster_id]
        x_noisy, _, x_length_scale, x_prior_var, x_noise_ratio = self.x_priors[cluster["x_kernel_index"]]
        y_noisy, _, y_length_scale, y_prior_var, y_noise_ratio = self.y_priors[cluster["y_kernel_index"]]
        x_coors = [self.exps[exp_id]["x_data"][:, :-1] for exp_id in cluster["members"]]
        x_targets = [self.exps[exp_id]["x_data"][:, -1].reshape(-1, 1) for exp_id in cluster["members"]]
        y_coors = [self.exps[exp_id]["y_data"][:, :-1] for exp_id in cluster["members"]]
        y_targets = [self.exps[exp_id]["y_data"][:, -1].reshape(-1, 1) for exp_id in cluster["members"]]

        if x_noisy:
            A = sum([x_coor.T @ x_coor for x_coor in x_coors])
            B = sum([x_coor.T @ x_target for x_coor, x_target in zip(x_coors, x_targets)])
            x_coef = lg.inv(A) @ B
        else:
            inverse_covs = [lg.inv(
                (self.kernel(x_coor, length_scales=x_length_scale) + np.eye(len(x_coor))*x_noise_ratio)*x_prior_var
                ) for x_coor in x_coors]
            A = sum([x_coor.T @ inverse_cov @ x_coor for x_coor, inverse_cov in zip(x_coors, inverse_covs)])
            B = sum([x_coor.T @ inverse_cov @ x_target for x_coor, x_target, inverse_cov in zip(x_coors, x_targets, inverse_covs)])
            x_coef = lg.inv(A) @ B

        if y_noisy:
            A = sum([y_coor.T @ y_coor for y_coor in y_coors])
            B = sum([y_coor.T @ y_target for y_coor, y_target in zip(y_coors, y_targets)])
            y_coef = lg.inv(A) @ B
        else:
            inverse_covs = [lg.inv(
                (self.kernel(y_coor, length_scales=y_length_scale) + np.eye(len(y_coor))*y_noise_ratio)*y_prior_var
                ) for y_coor in y_coors]
            A = sum([y_coor.T @ inverse_cov @ y_coor for y_coor, inverse_cov in zip(y_coors, inverse_covs)])
            B = sum([y_coor.T @ inverse_cov @ y_target for y_coor, y_target, inverse_cov in zip(y_coors, y_targets, inverse_covs)])
            y_coef = lg.inv(A) @ B

        return x_coef, y_coef



    def gibbs_sweep(self, return_log_prob=False):
        for exp_id in self.exps:
            exp = self.exps[exp_id]
            log_probs = np.zeros(len(self.clusters)+1, dtype=np.float32)
            clusters = np.zeros(len(self.clusters)+1, dtype=np.int32)
            log_probs[0] = exp["log_self_prob"] + math.log(self.alpha)
            for index, cluster_id in enumerate(self.clusters, 1):
                solo_cluster, log_prob = self.calculate_likelihood(exp_id, cluster_id)
                if solo_cluster:
                    log_probs[index] = log_probs[0]
                    log_probs[0] = -np.inf
                elif exp["cluster"] == cluster_id:
                    log_probs[index] = log_prob + math.log(self.clusters[cluster_id]["size"] - 1)
                else:
                    log_probs[index] = log_prob + math.log(self.clusters[cluster_id]["size"])

                clusters[index] = cluster_id
            
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)
            selected_cluster = np.random.choice(clusters, p=probs)
            
            original_cluster_id = exp["cluster"]
            if selected_cluster != original_cluster_id:
                self.remove_exp_from_cluster(exp_id, original_cluster_id)
                if selected_cluster:
                    self.add_exp_to_cluster(exp_id, int(selected_cluster))
                else:
                    exp["cluster"] = self.alloc_cluster(exp_id)
        
        if not return_log_prob:
            return 
        
        log_prob = 0
        for cluster in self.clusters.values():
            log_prob += math.lgamma(cluster["size"]) + math.log(self.alpha)
            if cluster["large"]:
                x_noisy, _, x_length_scale, x_prior_var, x_noise_ratio = self.x_priors[cluster["x_kernel_index"]]
                y_noisy, _, y_length_scale, y_prior_var, y_noise_ratio = self.y_priors[cluster["y_kernel_index"]]
                x_coef = cluster["x_coef"]
                y_coef = cluster["y_coef"]
                for exp_id in cluster["members"]:

                    x_coor = self.exps[exp_id]["x_data"][:, :-1]
                    x_target = self.exps[exp_id]["x_data"][:, -1].reshape(-1, 1)
                    y_coor = self.exps[exp_id]["y_data"][:, :-1]
                    y_target = self.exps[exp_id]["y_data"][:, -1].reshape(-1, 1)

                    x_diff = x_target - x_coor @ x_coef
                    y_diff = y_target - y_coor @ y_coef
                    data_size = self.exps[exp_id]["data_size"]

                    if x_noisy:
                        x_loglikelihood = -0.5*(x_diff / (x_prior_var*x_noise_ratio + x_prior_var)).T @ x_diff - \
                            0.5*data_size*math.log(x_prior_var*x_noise_ratio + x_prior_var)
                    else:
                        cov = x_prior_var*self.kernel(x_coor, length_scales=x_length_scale) + np.eye(data_size)*x_noise_ratio*x_prior_var
                        x_loglikelihood = -0.5*(x_diff.T @ lg.inv(cov) @ x_diff)
                        sign, log_det = lg.slogdet(cov)
                        log_det = log_det if sign else -100
                        x_loglikelihood -= 0.5*log_det

                    if y_noisy:
                        y_loglikelihood = -0.5*(y_diff / (y_prior_var*y_noise_ratio + y_prior_var)).T @ y_diff - \
                            0.5*data_size*math.log(y_prior_var*y_noise_ratio + y_prior_var)
                    else:
                        cov = y_prior_var*self.kernel(y_coor, length_scales=y_length_scale) + np.eye(data_size)*y_noise_ratio*y_prior_var
                        y_loglikelihood = -0.5*(y_diff.T @ lg.inv(cov) @ y_diff)
                        sign, log_det = lg.slogdet(cov)
                        log_det = log_det if sign else -100
                        y_loglikelihood -= 0.5*log_det

                    log_prob += float(x_loglikelihood + y_loglikelihood)
            else:
                x_log_likelihoods = cluster["x_loglikelihoods"]
                x_weights = cluster["x_weights"]
                log_prob += np.max(x_log_likelihoods)
                log_prob += np.log(np.sum(x_weights * np.exp(x_log_likelihoods - np.max(x_log_likelihoods))))

                y_log_likelihoods = cluster["y_loglikelihoods"]
                y_weights = cluster["y_weights"]
                log_prob += np.max(y_log_likelihoods)
                log_prob += np.log(np.sum(y_weights * np.exp(y_log_likelihoods - np.max(y_log_likelihoods))))

        return np.float32(log_prob)
        


    def calculate_likelihood(self, exp_id, cluster_id):
        exp = self.exps[exp_id]
        cluster = self.clusters[cluster_id]
        solo_cluster = False

        if exp["cluster"] == cluster_id and cluster["size"] == 1:
            solo_cluster = True
            log_prob = 0
        elif exp["cluster"] == cluster_id and (cluster["data_size"] - exp["data_size"]) < self.threshold:
            x_loglikelihoods = cluster["x_loglikelihoods"] - exp["x_loglikelihoods"]
            y_loglikelihoods = cluster["y_loglikelihoods"] - exp["y_loglikelihoods"]

            x_weights = np.exp(x_loglikelihoods - np.max(x_loglikelihoods))
            y_weights = np.exp(y_loglikelihoods - np.max(y_loglikelihoods))
            x_weights /= np.sum(x_weights)
            y_weights /= np.sum(y_weights)

            x_log_scale = np.max(exp["x_loglikelihoods"])
            y_log_scale = np.max(exp["y_loglikelihoods"])
            x_prob = np.sum(x_weights * np.exp(exp["x_loglikelihoods"] - x_log_scale))
            y_prob = np.sum(y_weights * np.exp(exp["y_loglikelihoods"] - y_log_scale))
            prob = x_prob * y_prob
            log_prob = x_log_scale + y_log_scale + (math.log(prob) if prob else -40)

        elif not cluster["large"]:
            x_log_scale = np.max(exp["x_loglikelihoods"])
            y_log_scale = np.max(exp["y_loglikelihoods"])
            x_prob = np.sum(cluster["x_weights"] * np.exp(exp["x_loglikelihoods"] - x_log_scale))
            y_prob = np.sum(cluster["y_weights"] * np.exp(exp["y_loglikelihoods"] - y_log_scale))
            prob = x_prob * y_prob
            log_prob = x_log_scale + y_log_scale + (math.log(prob) if prob else -40)
        else:
            x_noisy, _, x_length_scale, x_prior_var, x_noise_ratio = self.x_priors[cluster["x_kernel_index"]]
            y_noisy, _, y_length_scale, y_prior_var, y_noise_ratio = self.y_priors[cluster["y_kernel_index"]]

            x_coor = self.exps[exp_id]["x_data"][:, :-1]
            x_target = self.exps[exp_id]["x_data"][:, -1].reshape(-1, 1)
            y_coor = self.exps[exp_id]["y_data"][:, :-1]
            y_target = self.exps[exp_id]["y_data"][:, -1].reshape(-1, 1)

            x_diff = x_target - x_coor @ cluster["x_coef"]
            y_diff = y_target - y_coor @ cluster["y_coef"]
            data_size = exp["data_size"]

            if x_noisy:
                x_loglikelihood = -0.5*(x_diff / (x_prior_var*x_noise_ratio + x_prior_var)).T @ x_diff - \
                    0.5*data_size*math.log(x_prior_var*x_noise_ratio + x_prior_var)
            else:
                cov = x_prior_var*self.kernel(x_coor, length_scales=x_length_scale) + np.eye(data_size)*x_noise_ratio*x_prior_var
                x_loglikelihood = -0.5*(x_diff.T @ lg.inv(cov) @ x_diff)
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                x_loglikelihood -= 0.5*log_det

            if y_noisy:
                y_loglikelihood = -0.5*(y_diff / (y_prior_var*y_noise_ratio + y_prior_var)).T @ y_diff - \
                    0.5*data_size*math.log(y_prior_var*y_noise_ratio + y_prior_var)
            else:
                cov = y_prior_var*self.kernel(y_coor, length_scales=y_length_scale) + np.eye(data_size)*y_noise_ratio*y_prior_var
                y_loglikelihood = -0.5*(y_diff.T @ lg.inv(cov) @ y_diff)
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                y_loglikelihood -= 0.5*log_det

            log_prob = x_loglikelihood + y_loglikelihood

        return solo_cluster, log_prob


    def freeze(self):
        self.backup_clusters = {int(key): {i: copy.deepcopy(j) for i, j in value.items()} for key, value in self.clusters.items()}
        self.backup_next_cluster_id = np.int32(self.next_cluster_id)
        self.backup_empty_cluster_ids = self.empty_cluster_ids.copy()
    


    def MAP(self, rounds=20):
        for r in range(rounds):
            new_log_prob = self.gibbs_sweep(return_log_prob=True)
            if new_log_prob > self.log_prob:
                self.freeze()
                self.log_prob = new_log_prob
        
        self.clusters = {int(key): {i: copy.deepcopy(j) for i, j in value.items()} for key, value in self.backup_clusters.items()}
        self.next_cluster_id = int(self.backup_next_cluster_id)
        self.empty_cluster_ids = [i for i in self.backup_empty_cluster_ids]
        for cluster_id, cluster in self.clusters.items():
            for exp_id in cluster["members"]:
                self.exps[exp_id]["cluster"] = cluster_id



    def save(self, dir_path=None):
        dir_path = "archive_xy" if dir_path is None else dir_path
        base_configs = {"alpha": self.alpha, 
                        "next_exp_id": self.next_exp_id, 
                        "next_cluster_id": self.next_cluster_id, 
                        "empty_cluster_ids": self.empty_cluster_ids.copy(), 
                        "empty_exp_ids": self.empty_exp_ids.copy(), 
                        "threshold": self.threshold} # <--
        
        exp_seps = [] 
        exp_length = 0
        exps_x_data = []
        exps_y_data = []

        for exp_id, exp in self.exps.items():
            exp_seps.append((exp_id, exp_length, exp_length + exp["data_size"]))
            exp_length += exp["data_size"]
            exps_x_data.append(exp["x_data"])
            exps_y_data.append(exp["y_data"])
        base_configs["exp_seps"] = exp_seps
        
        exps_x_data = np.concatenate(exps_x_data, axis=0)[None, :, :]
        exps_y_data = np.concatenate(exps_y_data, axis=0)[None, :, :]
        exp_data = np.concatenate((exps_x_data, exps_y_data), axis=0) # <--

        clusters_data = {} # <--
        for cluster_id, cluster in self.clusters.items():
            cluster_info = {}
            for key, value in cluster.items():
                if type(value) == np.ndarray:
                    cluster_info[key] = value.tolist()
                elif type(value) == set:
                    cluster_info[key] = list(value)
                else:
                    cluster_info[key] = value
            
            clusters_data[cluster_id] = cluster_info
        
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        np.save(os.path.join(dir_path, "exp_data.npy"), exp_data)
        np.save(os.path.join(dir_path, "GP_x.npy"), self.x_prior_data)
        np.save(os.path.join(dir_path, "GP_y.npy"), self.y_prior_data)

        with open(os.path.join(dir_path, "base_configs.json"), "w") as file:
            json.dump(base_configs, file)

        with open(os.path.join(dir_path, "cluster_data.json"), "w") as file:
            json.dump(clusters_data, file)
        


    def load(self, dir_path):
        prior_path = {"x": os.path.join(dir_path, "GP_x.npy"), 
                      "y": os.path.join(dir_path, "GP_y.npy")}
        self.load_priors(prior_path)

        with open(os.path.join(dir_path, "base_configs.json"), "r") as file:
            base_configs = json.load(file)

        with open(os.path.join(dir_path, "cluster_data.json"), "r") as file:
            cluster_data = json.load(file)
        
        exp_data = np.load(os.path.join(dir_path, "exp_data.npy"))

        self.alpha = base_configs["alpha"]
        self.exps = {}
        self.threshold = base_configs["threshold"]

        exp_seps = base_configs["exp_seps"]
        exps = []
        ids = []

        for exp_id, start, end in exp_seps:
            x_exp, y_exp = exp_data[:, start: end, :].copy()
            exps.append({"x": x_exp, "y": y_exp})
            ids.append(exp_id)

        self.read_experiences(exps, ids)
        self.next_exp_id = base_configs["next_exp_id"]
        self.next_cluster_id = base_configs["next_cluster_id"]
        self.empty_cluster_ids = base_configs["empty_cluster_ids"]
        self.empty_exp_ids = base_configs["empty_exp_ids"]
        self.clusters = {}

        for cluster_id, cluster in cluster_data.items():
            updated_dict = {}
            for key, value in cluster.items():
                if key == "members":
                    updated_dict[key] = set(value)
                    for exp_id in value:
                        self.exps[exp_id]["cluster"] = int(cluster_id)
                elif type(value) == list:
                    updated_dict[key] = np.array(value)
            
            cluster.update(updated_dict)
            self.clusters[int(cluster_id)] = cluster
    

    def build_adaptor(self):
        total_size = 0
        x_gps = []
        y_gps = []
        prior_weights = []
        for cluster in self.clusters.values():
            if cluster["data_size"] >= 15:
                total_size += cluster["size"]

        if not total_size:
            raise Exception("Not enough data to build adaptor")
        
        for cluster in self.clusters.values():
            if cluster["data_size"] < 15:
                continue
            # print(cluster["id"])

            prior_weights.append(cluster["size"]/total_size)
            x_gp = [copy.deepcopy(i) for i in self.x_priors[cluster["x_kernel_index"]]]
            y_gp = [copy.deepcopy(i) for i in self.y_priors[cluster["y_kernel_index"]]]

            if cluster["large"]:
                x_gp[1] = cluster["x_coef"].copy()
                y_gp[1] = cluster["y_coef"].copy()

            x_gps.append(x_gp)
            y_gps.append(y_gp)
        
        adaptor = Adaptor_xy((x_gps, y_gps), np.array(prior_weights))
        return adaptor
                




class Archive_arc(Archive):
    def __init__(self, alpha=1.0, threshold=30):
        self.alpha = alpha
        self.kernel = Matern(v=1.5)
        self.exps = {}
        self.next_exp_id = 1
        self.next_cluster_id = 1
        self.empty_cluster_ids = []
        self.empty_exp_ids = []
        self.clusters = {}
        self.threshold = threshold
        self.backup_clusters = {}

        self.v_priors = None
        self.w_priors = None
        self.phi_priors = None
        self.log_prob = -np.inf

    
    def load_priors(self, paths):
        v_priors = np.load(paths["v"])
        self.v_prior_data = v_priors.copy()
        w_priors = np.load(paths["w"])
        self.w_prior_data = w_priors.copy()
        phi_priors = np.load(paths["phi"])
        self.phi_prior_data = phi_priors.copy()


        self.v_priors = []
        for v_prior in v_priors:
            coef = v_prior[:5].reshape(-1, 1)
            length_scale = v_prior[5:10]
            prior_var, noise_ratio = v_prior[10:12]
            noisy = True if v_prior[-1] else False
            self.v_priors.append((noisy, coef, length_scale, prior_var, noise_ratio))
        
        self.w_priors = []
        for w_prior in w_priors:
            coef = w_prior[:5].reshape(-1, 1)
            length_scale = w_prior[5:10]
            prior_var, noise_ratio = w_prior[10:12]
            noisy = True if w_prior[-1] else False
            self.w_priors.append((noisy, coef, length_scale, prior_var, noise_ratio))

        self.phi_priors = []
        for phi_prior in phi_priors:
            coef = phi_prior[:5].reshape(-1, 1)
            length_scale = phi_prior[5:10]
            prior_var, noise_ratio = phi_prior[10:12]
            noisy = True if phi_prior[-1] else False
            self.phi_priors.append((noisy, coef, length_scale, prior_var, noise_ratio))
    


    def read_experiences(self, exps, ids=None):
        if ids is None and exps:
            self.log_prob = -np.inf

        for id_index, exp in enumerate(exps):
            exp_dict = {}
            v_data = exp["v"]
            w_data = exp["w"]
            phi_data = exp["phi"]
            exp_dict["data_size"] = len(v_data)
            exp_dict["v_data"] = v_data.copy()
            exp_dict["w_data"] = w_data.copy()
            exp_dict["phi_data"] = phi_data.copy()

            if not ids is None:
                exp_id = ids[id_index]
            elif self.empty_exp_ids:
                exp_id = self.empty_exp_ids[0]
                del self.empty_exp_ids[0]
            else:
                exp_id = self.next_exp_id
                self.next_exp_id += 1

            exp_dict["id"] = exp_id
            self.exps[exp_id] = exp_dict
            v_loglikelihoods = np.zeros(len(self.v_priors), dtype=np.float32)
            w_loglikelihoods = np.zeros(len(self.w_priors), dtype=np.float32)
            phi_loglikelihoods = np.zeros(len(self.phi_priors), dtype=np.float32)

            v_coors = v_data[:, :-2]
            v_noise = v_data[:, -2]
            v_target = v_data[:, -1].reshape(-1, 1)
            w_coors = w_data[:, :-2]
            w_noise = w_data[:, -2]
            w_target = w_data[:, -1].reshape(-1, 1)
            phi_coors = phi_data[:, :-2]
            phi_noise = phi_data[:, -2]
            phi_target = phi_data[:, -1].reshape(-1, 1)
            
            for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(self.v_priors):
                diff = v_target - v_coors @ coef
                if noisy:
                    v_loglikelihood = -0.5*(diff / (prior_var*v_noise.reshape(-1, 1)*noise_ratio + prior_var)).T @ diff - \
                        0.5*np.sum(np.log(prior_var*v_noise*noise_ratio + prior_var))
                else:
                    cov = self.kernel(v_coors, length_scales=length_scale) + np.diag(v_noise)*noise_ratio
                    cov *= prior_var
                    v_loglikelihood = -0.5*(diff.T @ lg.inv(cov) @ diff)
                    sign, log_det = lg.slogdet(cov)
                    log_det = log_det if sign else -100
                    v_loglikelihood -= 0.5*log_det
                v_loglikelihoods[index] = float(v_loglikelihood)
            

            for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(self.w_priors):
                diff = w_target - w_coors @ coef
                if noisy:
                    w_loglikelihood = -0.5*(diff / (prior_var*w_noise.reshape(-1, 1)*noise_ratio + prior_var)).T @ diff - \
                        0.5*np.sum(np.log(prior_var*w_noise*noise_ratio + prior_var))
                else:
                    cov = self.kernel(w_coors, length_scales=length_scale) + np.diag(w_noise)*noise_ratio
                    cov *= prior_var
                    w_loglikelihood = -0.5*(diff.T @ lg.inv(cov) @ diff)
                    sign, log_det = lg.slogdet(cov)
                    log_det = log_det if sign else -100
                    w_loglikelihood -= 0.5*log_det
                w_loglikelihoods[index] = float(w_loglikelihood)
            

            for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(self.phi_priors):
                diff = phi_target - phi_coors @ coef
                if noisy:
                    phi_loglikelihood = -0.5*(diff / (prior_var*phi_noise.reshape(-1, 1)*noise_ratio + prior_var)).T @ diff - \
                        0.5*np.sum(np.log(prior_var*phi_noise*noise_ratio + prior_var))
                else:
                    cov = self.kernel(phi_coors, length_scales=length_scale) + np.diag(phi_noise)*noise_ratio
                    cov *= prior_var
                    phi_loglikelihood = -0.5*(diff.T @ lg.inv(cov) @ diff)
                    sign, log_det = lg.slogdet(cov)
                    log_det = log_det if sign else -100
                    phi_loglikelihood -= 0.5*log_det
                phi_loglikelihoods[index] = float(phi_loglikelihood)

            exp_dict["v_loglikelihoods"] = v_loglikelihoods
            exp_dict["w_loglikelihoods"] = w_loglikelihoods
            exp_dict["phi_loglikelihoods"] = phi_loglikelihoods
            log_self_prob = np.log(np.mean(np.exp(v_loglikelihoods - np.max(v_loglikelihoods))))
            log_self_prob += np.log(np.mean(np.exp(w_loglikelihoods - np.max(w_loglikelihoods))))
            log_self_prob += np.log(np.mean(np.exp(phi_loglikelihoods - np.max(phi_loglikelihoods))))
            log_self_prob += np.max(v_loglikelihoods) + np.max(w_loglikelihoods) + np.max(phi_loglikelihoods)
            exp_dict["log_self_prob"] = log_self_prob

            if ids is None:
                exp_dict["cluster"] = self.alloc_cluster(exp_id)
            else:
                exp_dict["cluster"] = 0
        
    

    def drop_exp(self, exp_id):
        exp = self.exps[exp_id]
        self.remove_exp_from_cluster(exp_id, exp["cluster"])
        del self.exps[exp_id]
        self.empty_exp_ids.append(exp_id)
        self.log_prob = -np.inf



    def alloc_cluster(self, exp_id):
        if self.empty_cluster_ids:
            cluster_id = self.empty_cluster_ids[0]
            del self.empty_cluster_ids[0]
        else:
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
        
        cluster_dict = {"id": cluster_id}
        cluster_dict["members"] = {exp_id}
        v_loglikelihoods = self.exps[exp_id]["v_loglikelihoods"].copy()
        w_loglikelihoods = self.exps[exp_id]["w_loglikelihoods"].copy()
        phi_loglikelihoods = self.exps[exp_id]["phi_loglikelihoods"].copy()
        cluster_dict["v_loglikelihoods"] = v_loglikelihoods
        cluster_dict["w_loglikelihoods"] = w_loglikelihoods
        cluster_dict["phi_loglikelihoods"] = phi_loglikelihoods
        v_weights = np.exp(v_loglikelihoods - np.max(v_loglikelihoods))
        w_weights = np.exp(w_loglikelihoods - np.max(w_loglikelihoods))
        phi_weights = np.exp(phi_loglikelihoods - np.max(phi_loglikelihoods))
        cluster_dict["v_weights"] = v_weights / np.sum(v_weights)
        cluster_dict["w_weights"] = w_weights / np.sum(w_weights)
        cluster_dict["phi_weights"] = phi_weights / np.sum(phi_weights)
        cluster_dict["size"] = 1
        cluster_dict["data_size"] = self.exps[exp_id]["data_size"]
        self.clusters[cluster_id] = cluster_dict
        if cluster_dict["data_size"] >= self.threshold:
            cluster_dict["large"] = True
            cluster_dict["v_kernel_index"] = int(np.argmax(v_weights))
            cluster_dict["w_kernel_index"] = int(np.argmax(w_weights))
            cluster_dict["phi_kernel_index"] = int(np.argmax(phi_weights))
            v_coef, w_coef, phi_coef = self.fit_coef(cluster_id)
            cluster_dict["v_coef"] = v_coef
            cluster_dict["w_coef"] = w_coef
            cluster_dict["phi_coef"] = phi_coef
        else:
            cluster_dict["large"] = False
            cluster_dict["v_coef"] = np.ones((5, 1), dtype=np.float32)
            cluster_dict["w_coef"] = np.ones((5, 1), dtype=np.float32)
            cluster_dict["phi_coef"] = np.ones((5, 1), dtype=np.float32)
            cluster_dict["v_kernel_index"] = 0
            cluster_dict["w_kernel_index"] = 0
            cluster_dict["phi_kernel_index"] = 0

        return cluster_id

    

    def add_exp_to_cluster(self, exp_id, cluster_id):
        cluster = self.clusters[cluster_id]
        cluster["members"].add(exp_id)
        self.exps[exp_id]["cluster"] = cluster_id
        v_loglikelihoods = self.exps[exp_id]["v_loglikelihoods"]
        w_loglikelihoods = self.exps[exp_id]["w_loglikelihoods"]
        phi_loglikelihoods = self.exps[exp_id]["phi_loglikelihoods"]
        cluster["v_loglikelihoods"] += v_loglikelihoods
        cluster["w_loglikelihoods"] += w_loglikelihoods
        cluster["phi_loglikelihoods"] += phi_loglikelihoods
        new_v_loglikelihoods = cluster["v_loglikelihoods"]
        new_w_loglikelihoods = cluster["w_loglikelihoods"]
        new_phi_loglikelihoods = cluster["phi_loglikelihoods"]
        v_weights = np.exp(new_v_loglikelihoods - np.max(new_v_loglikelihoods))
        w_weights = np.exp(new_w_loglikelihoods - np.max(new_w_loglikelihoods))
        phi_weights = np.exp(new_phi_loglikelihoods - np.max(new_phi_loglikelihoods))
        cluster["v_weights"] = v_weights / np.sum(v_weights)
        cluster["w_weights"] = w_weights / np.sum(w_weights)
        cluster["phi_weights"] = phi_weights / np.sum(phi_weights)
        cluster["size"] += 1
        cluster["data_size"] += self.exps[exp_id]["data_size"]

        if cluster["data_size"] >= self.threshold:
            cluster["large"] = False
            cluster["v_kernel_index"] = int(np.argmax(v_weights))
            cluster["w_kernel_index"] = int(np.argmax(w_weights))
            cluster["phi_kernel_index"] = int(np.argmax(phi_weights))
            v_coef, w_coef, phi_coef = self.fit_coef(cluster_id)
            cluster["v_coef"] = v_coef
            cluster["w_coef"] = w_coef
            cluster["phi_coef"] = phi_coef



    def remove_exp_from_cluster(self, exp_id, cluster_id):
        cluster = self.clusters[cluster_id]
        if cluster["size"] == 1:
            self.empty_cluster_ids.append(cluster_id)
            del self.clusters[cluster_id]
            return

        cluster["members"].discard(exp_id)
        v_loglikelihoods = self.exps[exp_id]["v_loglikelihoods"]
        w_loglikelihoods = self.exps[exp_id]["w_loglikelihoods"]
        phi_loglikelihoods = self.exps[exp_id]["phi_loglikelihoods"]
        cluster["v_loglikelihoods"] -= v_loglikelihoods
        cluster["w_loglikelihoods"] -= w_loglikelihoods
        cluster["phi_loglikelihoods"] -= phi_loglikelihoods
        new_v_loglikelihoods = cluster["v_loglikelihoods"]
        new_w_loglikelihoods = cluster["w_loglikelihoods"]
        new_phi_loglikelihoods = cluster["phi_loglikelihoods"]
        v_weights = np.exp(new_v_loglikelihoods - np.max(new_v_loglikelihoods))
        w_weights = np.exp(new_w_loglikelihoods - np.max(new_w_loglikelihoods))
        phi_weights = np.exp(new_phi_loglikelihoods - np.max(new_phi_loglikelihoods))
        cluster["v_weights"] = v_weights / np.sum(v_weights)
        cluster["w_weights"] = w_weights / np.sum(w_weights)
        cluster["phi_weights"] = w_weights / np.sum(phi_weights)
        cluster["size"] -= 1
        cluster["data_size"] -= self.exps[exp_id]["data_size"]

        if cluster["data_size"] >= self.threshold:
            cluster["v_kernel_index"] = int(np.argmax(v_weights))
            cluster["w_kernel_index"] = int(np.argmax(w_weights))
            cluster["phi_kernel_index"] = int(np.argmax(phi_weights))
            v_coef, w_coef, phi_coef = self.fit_coef(cluster_id)
            cluster["v_coef"] = v_coef
            cluster["w_coef"] = w_coef
            cluster["phi_coef"] = phi_coef
        else:
            cluster["large"] = False


    
    def fit_coef(self, cluster_id):
        cluster = self.clusters[cluster_id]
        v_noisy, _, v_length_scale, __, v_noise_ratio = self.v_priors[cluster["v_kernel_index"]]
        w_noisy, _, w_length_scale, __, w_noise_ratio = self.w_priors[cluster["w_kernel_index"]]
        phi_noisy, _, phi_length_scale, __, phi_noise_ratio = self.phi_priors[cluster["phi_kernel_index"]]

        v_coors = [self.exps[exp_id]["v_data"][:, :-2] for exp_id in cluster["members"]]
        v_noises = [self.exps[exp_id]["v_data"][:, -2] for exp_id in cluster["members"]]
        v_targets = [self.exps[exp_id]["v_data"][:, -1].reshape(-1, 1) for exp_id in cluster["members"]]

        w_coors = [self.exps[exp_id]["w_data"][:, :-2] for exp_id in cluster["members"]]
        w_noises = [self.exps[exp_id]["w_data"][:, -2] for exp_id in cluster["members"]]
        w_targets = [self.exps[exp_id]["w_data"][:, -1].reshape(-1, 1) for exp_id in cluster["members"]]

        phi_coors = [self.exps[exp_id]["phi_data"][:, :-2] for exp_id in cluster["members"]]
        phi_noises = [self.exps[exp_id]["phi_data"][:, -2] for exp_id in cluster["members"]]
        phi_targets = [self.exps[exp_id]["phi_data"][:, -1].reshape(-1, 1) for exp_id in cluster["members"]]

        if v_noisy:
            inverse_covs = [np.diag(1/(1 + v_noise*v_noise_ratio)) for v_noise in v_noises]
        else:
            inverse_covs = [lg.inv(
                self.kernel(v_coor, length_scales=v_length_scale) + np.diag(v_noise)*v_noise_ratio
                ) for v_coor, v_noise in zip(v_coors, v_noises)]
            
        A = sum([v_coor.T @ inverse_cov @ v_coor for v_coor, inverse_cov in zip(v_coors, inverse_covs)])
        B = sum([v_coor.T @ inverse_cov @ v_target for v_coor, v_target, inverse_cov in zip(v_coors, v_targets, inverse_covs)])
        v_coef = lg.inv(A) @ B

        if w_noisy:
            inverse_covs = [np.diag(1/(1 + w_noise*w_noise_ratio)) for w_noise in w_noises]
        else:
            inverse_covs = [lg.inv(
                self.kernel(w_coor, length_scales=w_length_scale) + np.diag(w_noise)*w_noise_ratio
                ) for w_coor, w_noise in zip(w_coors, w_noises)]
            
        A = sum([w_coor.T @ inverse_cov @ w_coor for w_coor, inverse_cov in zip(w_coors, inverse_covs)])
        B = sum([w_coor.T @ inverse_cov @ w_target for w_coor, w_target, inverse_cov in zip(w_coors, w_targets, inverse_covs)])
        w_coef = lg.inv(A) @ B

        if phi_noisy:
            inverse_covs = [np.diag(1/(1 + phi_noise*phi_noise_ratio)) for phi_noise in phi_noises]
        else:
            inverse_covs = [lg.inv(
                self.kernel(phi_coor, length_scales=phi_length_scale) + np.diag(phi_noise)*phi_noise_ratio
                ) for phi_coor, phi_noise in zip(phi_coors, phi_noises)]
            
        A = sum([phi_coor.T @ inverse_cov @ phi_coor for phi_coor, inverse_cov in zip(phi_coors, inverse_covs)])
        B = sum([phi_coor.T @ inverse_cov @ phi_target for phi_coor, phi_target, inverse_cov in zip(phi_coors, phi_targets, inverse_covs)])
        phi_coef = lg.inv(A) @ B

        return v_coef, w_coef, phi_coef



    def gibbs_sweep(self, return_log_prob=False):
        for exp_id in self.exps:
            exp = self.exps[exp_id]
            log_probs = np.zeros(len(self.clusters)+1, dtype=np.float32)
            clusters = np.zeros(len(self.clusters)+1, dtype=np.int32)
            log_probs[0] = exp["log_self_prob"] + math.log(self.alpha)
            for index, cluster_id in enumerate(self.clusters, 1):
                solo_cluster, log_prob = self.calculate_likelihood(exp_id, cluster_id)
                if solo_cluster:
                    log_probs[index] = log_probs[0]
                    log_probs[0] = -np.inf
                elif exp["cluster"] == cluster_id:
                    log_probs[index] = log_prob + math.log(self.clusters[cluster_id]["size"] - 1)
                else:
                    log_probs[index] = log_prob + math.log(self.clusters[cluster_id]["size"])

                clusters[index] = cluster_id
            
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)
            selected_cluster = np.random.choice(clusters, p=probs)
            
            original_cluster_id = exp["cluster"]
            if selected_cluster != original_cluster_id:
                self.remove_exp_from_cluster(exp_id, original_cluster_id)
                if selected_cluster:
                    self.add_exp_to_cluster(exp_id, int(selected_cluster))
                else:
                    exp["cluster"] = self.alloc_cluster(exp_id)
        
        if not return_log_prob:
            return 
        
        log_prob = 0
        for cluster in self.clusters.values():
            log_prob += math.lgamma(cluster["size"]) + math.log(self.alpha)
            if cluster["large"]:
                v_noisy, _, v_length_scale, v_prior_var, v_noise_ratio = self.v_priors[cluster["v_kernel_index"]]
                w_noisy, _, w_length_scale, w_prior_var, w_noise_ratio = self.w_priors[cluster["w_kernel_index"]]
                phi_noisy, _, phi_length_scale, phi_prior_var, phi_noise_ratio = self.phi_priors[cluster["phi_kernel_index"]]

                v_coef = cluster["v_coef"]
                w_coef = cluster["w_coef"]
                phi_coef = cluster["phi_coef"]

                for exp_id in cluster["members"]:
                    v_coor = self.exps[exp_id]["v_data"][:, :-2]
                    v_noise = self.exps[exp_id]["v_data"][:, -2]
                    v_target = self.exps[exp_id]["v_data"][:, -1].reshape(-1, 1)

                    w_coor = self.exps[exp_id]["w_data"][:, :-2]
                    w_noise = self.exps[exp_id]["w_data"][:, -2]
                    w_target = self.exps[exp_id]["w_data"][:, -1].reshape(-1, 1)

                    phi_coor = self.exps[exp_id]["phi_data"][:, :-2]
                    phi_noise = self.exps[exp_id]["phi_data"][:, -2]
                    phi_target = self.exps[exp_id]["phi_data"][:, -1].reshape(-1, 1)

                    v_diff = v_target - v_coor @ v_coef
                    w_diff = w_target - w_coor @ w_coef
                    phi_diff = phi_target - phi_coor @ phi_coef

                    if v_noisy:
                        v_loglikelihood = -0.5*(v_diff / (v_prior_var*v_noise.reshape(-1, 1)*v_noise_ratio + v_prior_var)
                                                ).T @ v_diff - 0.5*np.sum(np.log(v_prior_var*v_noise*v_noise_ratio + v_prior_var))
                    else:
                        cov = v_prior_var*self.kernel(v_coor, length_scales=v_length_scale) + \
                            np.diag(v_noise)*v_noise_ratio*v_prior_var
                        v_loglikelihood = -0.5*(v_diff.T @ lg.inv(cov) @ v_diff)
                        sign, log_det = lg.slogdet(cov)
                        log_det = log_det if sign else -100
                        v_loglikelihood -= 0.5*log_det

                    if w_noisy:
                        w_loglikelihood = -0.5*(w_diff / (w_prior_var*w_noise.reshape(-1, 1)*w_noise_ratio + w_prior_var)
                                                ).T @ w_diff - 0.5*np.sum(np.log(w_prior_var*w_noise*w_noise_ratio + w_prior_var))
                    else:
                        cov = w_prior_var*self.kernel(w_coor, length_scales=w_length_scale) + \
                            np.diag(w_noise)*w_noise_ratio*w_prior_var
                        w_loglikelihood = -0.5*(w_diff.T @ lg.inv(cov) @ w_diff)
                        sign, log_det = lg.slogdet(cov)
                        log_det = log_det if sign else -100
                        w_loglikelihood -= 0.5*log_det
                    
                    if phi_noisy:
                        phi_loglikelihood = -0.5*(phi_diff / (phi_prior_var*phi_noise.reshape(-1, 1)*phi_noise_ratio + phi_prior_var)
                                                  ).T @ phi_diff - 0.5*np.sum(np.log(phi_prior_var*phi_noise*phi_noise_ratio + phi_prior_var))
                    else:
                        cov = phi_prior_var*self.kernel(phi_coor, length_scales=phi_length_scale) + \
                            np.diag(phi_noise)*phi_noise_ratio*phi_prior_var
                        phi_loglikelihood = -0.5*(phi_diff.T @ lg.inv(cov) @ phi_diff)
                        sign, log_det = lg.slogdet(cov)
                        log_det = log_det if sign else -100
                        phi_loglikelihood -= 0.5*log_det

                    log_prob += float(v_loglikelihood + w_loglikelihood + phi_loglikelihood)
            else:
                v_log_likelihoods = cluster["v_loglikelihoods"]
                v_weights = cluster["v_weights"]
                log_prob += np.max(v_log_likelihoods)
                log_prob += np.log(np.sum(v_weights * np.exp(v_log_likelihoods - np.max(v_log_likelihoods))))

                w_log_likelihoods = cluster["w_loglikelihoods"]
                w_weights = cluster["w_weights"]
                log_prob += np.max(w_log_likelihoods)
                log_prob += np.log(np.sum(w_weights * np.exp(w_log_likelihoods - np.max(w_log_likelihoods))))

                phi_log_likelihoods = cluster["phi_loglikelihoods"]
                phi_weights = cluster["phi_weights"]
                log_prob += np.max(phi_log_likelihoods)
                log_prob += np.log(np.sum(phi_weights * np.exp(phi_log_likelihoods - np.max(phi_log_likelihoods))))

        return np.float32(log_prob)
        


    def calculate_likelihood(self, exp_id, cluster_id):
        exp = self.exps[exp_id]
        cluster = self.clusters[cluster_id]
        solo_cluster = False

        if exp["cluster"] == cluster_id and cluster["size"] == 1:
            solo_cluster = True
            log_prob = 0
        elif exp["cluster"] == cluster_id and (cluster["data_size"] - exp["data_size"]) < self.threshold:
            v_loglikelihoods = cluster["v_loglikelihoods"] - exp["v_loglikelihoods"]
            w_loglikelihoods = cluster["w_loglikelihoods"] - exp["w_loglikelihoods"]
            phi_loglikelihoods = cluster["phi_loglikelihoods"] - exp["phi_loglikelihoods"]

            v_weights = np.exp(v_loglikelihoods - np.max(v_loglikelihoods))
            w_weights = np.exp(w_loglikelihoods - np.max(w_loglikelihoods))
            phi_weights = np.exp(phi_loglikelihoods - np.max(phi_loglikelihoods))
            v_weights /= np.sum(v_weights)
            w_weights /= np.sum(w_weights)
            phi_weights /= np.sum(phi_weights)

            v_log_scale = np.max(exp["v_loglikelihoods"])
            w_log_scale = np.max(exp["w_loglikelihoods"])
            phi_log_scale = np.max(exp["phi_loglikelihoods"])
            v_prob = np.sum(v_weights * np.exp(exp["v_loglikelihoods"] - v_log_scale))
            w_prob = np.sum(w_weights * np.exp(exp["w_loglikelihoods"] - w_log_scale))
            phi_prob = np.sum(phi_weights * np.exp(exp["phi_loglikelihoods"] - phi_log_scale))

            prob = (math.log(v_prob) if v_prob else -40) + (math.log(w_prob) if w_prob else -40) + (math.log(phi_prob) if phi_prob else -40) 
            log_prob = v_log_scale + w_log_scale + phi_log_scale + prob
        elif not cluster["large"]:
            v_log_scale = np.max(exp["v_loglikelihoods"])
            w_log_scale = np.max(exp["w_loglikelihoods"])
            phi_log_scale = np.max(exp["phi_loglikelihoods"])
            v_prob = np.sum(cluster["v_weights"] * np.exp(exp["v_loglikelihoods"] - v_log_scale))
            w_prob = np.sum(cluster["w_weights"] * np.exp(exp["w_loglikelihoods"] - w_log_scale))
            phi_prob = np.sum(cluster["phi_weights"] * np.exp(exp["phi_loglikelihoods"] - phi_log_scale))
            prob = (math.log(v_prob) if v_prob else -40) + (math.log(w_prob) if w_prob else -40) + (math.log(phi_prob) if phi_prob else -40) 
            log_prob = v_log_scale + w_log_scale + phi_log_scale + prob
        else:
            v_noisy, _, v_length_scale, v_prior_var, v_noise_ratio = self.v_priors[cluster["v_kernel_index"]]
            w_noisy, _, w_length_scale, w_prior_var, w_noise_ratio = self.w_priors[cluster["w_kernel_index"]]
            phi_noisy, _, phi_length_scale, phi_prior_var, phi_noise_ratio = self.phi_priors[cluster["phi_kernel_index"]]

            v_coor = self.exps[exp_id]["v_data"][:, :-2]
            v_noise = self.exps[exp_id]["v_data"][:, -2]
            v_target = self.exps[exp_id]["v_data"][:, -1].reshape(-1, 1)

            w_coor = self.exps[exp_id]["w_data"][:, :-2]
            w_noise = self.exps[exp_id]["w_data"][:, -2]
            w_target = self.exps[exp_id]["w_data"][:, -1].reshape(-1, 1)

            phi_coor = self.exps[exp_id]["phi_data"][:, :-2]
            phi_noise = self.exps[exp_id]["phi_data"][:, -2]
            phi_target = self.exps[exp_id]["phi_data"][:, -1].reshape(-1, 1)

            v_diff = v_target - v_coor @ cluster["v_coef"]
            w_diff = w_target - w_coor @ cluster["w_coef"]
            phi_diff = phi_target - phi_coor @ cluster["phi_coef"]

            if v_noisy:
                v_loglikelihood = -0.5*(v_diff / (v_prior_var*v_noise.reshape(-1, 1)*v_noise_ratio + v_prior_var)).T @ v_diff - \
                    0.5*np.sum(np.log(v_prior_var*v_noise*v_noise_ratio + v_prior_var))
            else:
                cov = v_prior_var*self.kernel(v_coor, length_scales=v_length_scale) + np.diag(v_noise)*v_noise_ratio*v_prior_var
                v_loglikelihood = -0.5*(v_diff.T @ lg.inv(cov) @ v_diff)
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                v_loglikelihood -= 0.5*log_det

            if w_noisy:
                w_loglikelihood = -0.5*(w_diff / (w_prior_var*w_noise.reshape(-1, 1)*w_noise_ratio + w_prior_var)).T @ w_diff - \
                    0.5*np.sum(np.log(w_prior_var*w_noise*w_noise_ratio + w_prior_var))
            else:
                cov = w_prior_var*self.kernel(w_coor, length_scales=w_length_scale) + np.diag(w_noise)*w_noise_ratio*w_prior_var
                w_loglikelihood = -0.5*(w_diff.T @ lg.inv(cov) @ w_diff)
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                w_loglikelihood -= 0.5*log_det
            
            if phi_noisy:
                phi_loglikelihood = -0.5*(phi_diff / (phi_prior_var*phi_noise.reshape(-1, 1)*phi_noise_ratio + phi_prior_var)).T @ phi_diff - \
                    0.5*np.sum(np.log(phi_prior_var*phi_noise*phi_noise_ratio + phi_prior_var))
            else:
                cov = phi_prior_var*self.kernel(phi_coor, length_scales=phi_length_scale) + np.diag(phi_noise)*phi_noise_ratio*phi_prior_var
                phi_loglikelihood = -0.5*(phi_diff.T @ lg.inv(cov) @ phi_diff)
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                phi_loglikelihood -= 0.5*log_det

            log_prob = v_loglikelihood + w_loglikelihood + phi_loglikelihood

        return solo_cluster, log_prob


    def freeze(self):
        self.backup_clusters = {int(key): {i: copy.deepcopy(j) for i, j in value.items()} for key, value in self.clusters.items()}
        self.backup_next_cluster_id = np.int32(self.next_cluster_id)
        self.backup_empty_cluster_ids = self.empty_cluster_ids.copy()
    


    def MAP(self, rounds=20):
        for r in range(rounds):
            new_log_prob = self.gibbs_sweep(return_log_prob=True)
            if new_log_prob > self.log_prob:
                self.freeze()
                self.log_prob = new_log_prob
        
        self.clusters = {int(key): {i: copy.deepcopy(j) for i, j in value.items()} for key, value in self.backup_clusters.items()}
        self.next_cluster_id = int(self.backup_next_cluster_id)
        self.empty_cluster_ids = [i for i in self.backup_empty_cluster_ids]
        for cluster_id, cluster in self.clusters.items():
            for exp_id in cluster["members"]:
                self.exps[exp_id]["cluster"] = cluster_id



    def save(self, dir_path=None):
        dir_path = "archive_arc" if dir_path is None else dir_path
        base_configs = {"alpha": self.alpha, 
                        "next_exp_id": self.next_exp_id, 
                        "next_cluster_id": self.next_cluster_id, 
                        "empty_cluster_ids": self.empty_cluster_ids.copy(), 
                        "empty_exp_ids": self.empty_exp_ids.copy(), 
                        "threshold": self.threshold} # <--
        
        exp_seps = [] 
        exp_length = 0
        exps_v_data = []
        exps_w_data = []
        exps_phi_data = []

        for exp_id, exp in self.exps.items():
            exp_seps.append((exp_id, exp_length, exp_length + exp["data_size"]))
            exp_length += exp["data_size"]
            exps_v_data.append(exp["v_data"])
            exps_w_data.append(exp["w_data"])
            exps_phi_data.append(exp["phi_data"])
        base_configs["exp_seps"] = exp_seps
        
        exps_v_data = np.concatenate(exps_v_data, axis=0)[None, :, :]
        exps_w_data = np.concatenate(exps_w_data, axis=0)[None, :, :]
        exps_phi_data = np.concatenate(exps_phi_data, axis=0)[None, :, :]
        exp_data = np.concatenate((exps_v_data, exps_w_data, exps_phi_data), axis=0) # <--

        clusters_data = {} # <--
        for cluster_id, cluster in self.clusters.items():
            cluster_info = {}
            for key, value in cluster.items():
                if type(value) == np.ndarray:
                    cluster_info[key] = value.tolist()
                elif type(value) == set:
                    cluster_info[key] = list(value)
                else:
                    cluster_info[key] = value
            
            clusters_data[cluster_id] = cluster_info
        
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        np.save(os.path.join(dir_path, "exp_data.npy"), exp_data)
        np.save(os.path.join(dir_path, "GP_v.npy"), self.v_prior_data)
        np.save(os.path.join(dir_path, "GP_w.npy"), self.w_prior_data)
        np.save(os.path.join(dir_path, "GP_phi.npy"), self.phi_prior_data)

        with open(os.path.join(dir_path, "base_configs.json"), "w") as file:
            json.dump(base_configs, file)

        with open(os.path.join(dir_path, "cluster_data.json"), "w") as file:
            json.dump(clusters_data, file)
        


    def load(self, dir_path):
        prior_path = {"v": os.path.join(dir_path, "GP_v.npy"), 
                      "w": os.path.join(dir_path, "GP_w.npy"),
                      "phi": os.path.join(dir_path, "GP_phi.npy")}
        self.load_priors(prior_path)

        with open(os.path.join(dir_path, "base_configs.json"), "r") as file:
            base_configs = json.load(file)

        with open(os.path.join(dir_path, "cluster_data.json"), "r") as file:
            cluster_data = json.load(file)
        
        exp_data = np.load(os.path.join(dir_path, "exp_data.npy"))

        self.alpha = base_configs["alpha"]
        self.exps = {}
        self.threshold = base_configs["threshold"]

        exp_seps = base_configs["exp_seps"]
        exps = []
        ids = []

        for exp_id, start, end in exp_seps:
            v_exp, w_exp, phi_exp = exp_data[:, start: end, :].copy()
            exps.append({"v": v_exp, "w": w_exp, "phi": phi_exp})
            ids.append(exp_id)

        self.read_experiences(exps, ids)
        self.next_exp_id = base_configs["next_exp_id"]
        self.next_cluster_id = base_configs["next_cluster_id"]
        self.empty_cluster_ids = base_configs["empty_cluster_ids"]
        self.empty_exp_ids = base_configs["empty_exp_ids"]
        self.clusters = {}

        for cluster_id, cluster in cluster_data.items():
            updated_dict = {}
            for key, value in cluster.items():
                if key == "members":
                    updated_dict[key] = set(value)
                    for exp_id in value:
                        self.exps[exp_id]["cluster"] = int(cluster_id)
                elif type(value) == list:
                    updated_dict[key] = np.array(value)
            
            cluster.update(updated_dict)
            self.clusters[int(cluster_id)] = cluster
    

    def build_adaptor(self):
        total_size = 0
        v_gps = []
        w_gps = []
        phi_gps = []
        prior_weights = []
        for cluster in self.clusters.values():
            if cluster["data_size"] >= 15:
                total_size += cluster["size"]

        if not total_size:
            raise Exception("Not enough data to build adaptor")
        
        for cluster in self.clusters.values():
            if cluster["data_size"] < 15:
                continue

            # print(cluster["id"])

            prior_weights.append(cluster["size"]/total_size)
            v_gp = [copy.deepcopy(i) for i in self.v_priors[cluster["v_kernel_index"]]]
            w_gp = [copy.deepcopy(i) for i in self.w_priors[cluster["w_kernel_index"]]]
            phi_gp = [copy.deepcopy(i) for i in self.phi_priors[cluster["phi_kernel_index"]]]

            if cluster["large"]:
                v_gp[1] = cluster["v_coef"].copy()
                w_gp[1] = cluster["w_coef"].copy()
                phi_gp[1] = cluster["phi_coef"].copy()

            v_gps.append(v_gp)
            w_gps.append(w_gp)
            phi_gps.append(phi_gp)
        
        adaptor = Adaptor_arc((v_gps, w_gps, phi_gps), np.array(prior_weights))
        return adaptor
                    

    











