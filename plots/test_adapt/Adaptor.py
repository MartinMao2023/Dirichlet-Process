from abc import ABCMeta, abstractmethod
from Matern import Matern
import numpy as np
import numpy.linalg as lg
import math
import json
import os


class Adaptor(metaclass=ABCMeta):

    @abstractmethod
    def read_data(self, *data):
        pass


    @abstractmethod
    def predict(self, coors, baselines):
        pass




class Adaptor_xy(Adaptor):
    def __init__(self, gps=None, weights=None):
        self.kernel = Matern()
        self.gps = gps
        if weights is not None:
            self.prior_weights = weights.copy()
            self.posterior_weights = weights.copy()
            self.posteriors = [(i, weights[i]) for i in range(len(weights))]
        self.x_data = []
        self.y_data = []
    

    def read_data(self, data):
        """
        data: 2 x 6 or 2 x 1 x 6 array
        """
        x_data, y_data = data
        self.x_data.append(x_data.copy().reshape(1, -1))
        self.y_data.append(y_data.copy().reshape(1, -1))
        x_data = np.concatenate(self.x_data, axis=0)
        y_data = np.concatenate(self.y_data, axis=0)
        x_coor = x_data[:, :-1]
        x_target = x_data[:, -1].reshape(-1, 1)
        y_coor = y_data[:, :-1]
        y_target = y_data[:, -1].reshape(-1, 1)

        x_log_likelihoods = np.zeros(len(self.prior_weights), dtype=np.float32)
        y_log_likelihoods = np.zeros(len(self.prior_weights), dtype=np.float32)

        x_gps, y_gps = self.gps

        for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(x_gps):
            diff = x_target - x_coor @ coef
            prior_var += 1e-6
            if noisy:
                log_likelihood = -0.5*diff.T @ diff / (noise_ratio*prior_var + prior_var)
                log_likelihood -= 0.5*len(x_coor)*math.log(prior_var*noise_ratio + prior_var)
            else:
                cov = self.kernel(x_coor, length_scales=length_scale) + np.eye(len(x_coor))*noise_ratio
                cov *= prior_var
                log_likelihood = -0.5*diff.T @ lg.inv(cov) @ diff
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                log_likelihood -= 0.5*log_det
            x_log_likelihoods[index] = log_likelihood
        
        for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(y_gps):
            diff = y_target - y_coor @ coef
            prior_var += 1e-6
            if noisy:
                log_likelihood = -0.5*diff.T @ diff / (noise_ratio*prior_var + prior_var)
                log_likelihood -= 0.5*len(y_coor)*math.log(prior_var*noise_ratio + prior_var)
            else:
                cov = self.kernel(y_coor, length_scales=length_scale) + np.eye(len(y_coor))*noise_ratio
                cov *= prior_var
                log_likelihood = -0.5*diff.T @ lg.inv(cov) @ diff
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                log_likelihood -= 0.5*log_det
            y_log_likelihoods[index] = log_likelihood
        
        log_likelihoods = x_log_likelihoods + y_log_likelihoods
        log_scale = np.max(log_likelihoods)
        log_likelihoods -= np.max(log_likelihoods)
        posterior_weights = np.exp(log_likelihoods) * self.prior_weights
        log_scale += np.log(np.sum(posterior_weights))
        posterior_weights /= np.sum(posterior_weights)

        self.posterior_weights = posterior_weights.copy()
        sorted_weights = np.argsort(-posterior_weights)
        indexes = []
        total_weights = 0
        for index in sorted_weights:
            indexes.append(index)
            total_weights += posterior_weights[index]
            if total_weights > 0.9:
                break
        self.posteriors = [(index, posterior_weights[index]/total_weights) for index in indexes]

        return log_scale

    

    def predict(self, coors, baselines):
        """
        coors: 2 x N x 5 array
        baselines: 2 x N or 2 x N x 1 array
        """
        x_coors = coors[0].copy() 
        y_coors = coors[1].copy()
        x_baselines = baselines[0].copy()
        y_baselines = baselines[1].copy()
        x_gps, y_gps = self.gps
        predicted_x = x_baselines.reshape(-1, 1)
        predicted_y = y_baselines.reshape(-1, 1)

        if not self.x_data:
            for index, weight in self.posteriors:
                x_coef = x_gps[index][1]
                y_coef = y_gps[index][1]
                predicted_x += weight*(x_coors @ x_coef)
                predicted_y += weight*(y_coors @ y_coef)
            return predicted_x, predicted_y

        x_data = np.concatenate(self.x_data, axis=0)
        y_data = np.concatenate(self.y_data, axis=0)
        x_coor = x_data[:, :-1]
        x_target = x_data[:, -1].reshape(-1, 1)
        y_coor = y_data[:, :-1]
        y_target = y_data[:, -1].reshape(-1, 1)

        for index, weight in self.posteriors:
            x_noisy, x_coef, x_length_scale, _, x_noise_ratio = x_gps[index]
            if x_noisy:
                predicted_x += weight*(x_coors @ x_coef)
            else:
                diff_x = x_target - x_coor @ x_coef
                sigm22 = self.kernel(x_coor, length_scales=x_length_scale) + np.eye(len(x_coor))*x_noise_ratio
                sigma12 = self.kernel(x_coors, x_coor, length_scales=x_length_scale)
                predicted_x += weight*(x_coors @ x_coef + sigma12 @ lg.inv(sigm22) @ diff_x)
        
            y_noisy, y_coef, y_length_scale, _, y_noise_ratio = y_gps[index]
            if y_noisy:
                predicted_y += weight*(y_coors @ y_coef)
            else:
                diff_y = y_target - y_coor @ y_coef
                sigm22 = self.kernel(y_coor, length_scales=y_length_scale) + np.eye(len(y_coor))*y_noise_ratio
                sigma12 = self.kernel(y_coors, y_coor, length_scales=y_length_scale)
                predicted_y += weight*(y_coors @ y_coef + sigma12 @ lg.inv(sigm22) @ diff_y)

        return predicted_x, predicted_y


    def save(self, path="adaptor_xy.json"):
        configs = {"weights": self.prior_weights.tolist()}
        x_gps, y_gps = self.gps
        new_x_gps = []
        new_y_gps = []

        for x_gp in x_gps:
            noisy, coef, length_scale, prior_var, noise_ratio = x_gp
            new_x_gps.append((noisy, 
                              coef.tolist(), 
                              length_scale.tolist(), 
                              float(prior_var), 
                              float(noise_ratio)))
        
        for y_gp in y_gps:
            noisy, coef, length_scale, prior_var, noise_ratio = y_gp
            new_y_gps.append((noisy, 
                              coef.tolist(), 
                              length_scale.tolist(), 
                              float(prior_var), 
                              float(noise_ratio)))
        
        configs["gps"] = (new_x_gps, new_y_gps)

        with open(path, "w") as file:
            json.dump(configs, file)



    def load(self, path):
        self.x_data = []
        self.y_data = []
        with open(path, "r") as file:
            configs = json.load(file)
        
        weights = np.array(configs["weights"])
        self.prior_weights = weights.copy()
        self.posterior_weights = weights.copy()
        self.posteriors = [(i, weights[i]) for i in range(len(weights))]

        x_gps = []
        for x_gp in configs["gps"][0]:
            noisy, coef, length_scale, prior_var, noise_ratio = x_gp
            x_gps.append([noisy, 
                          np.array(coef), 
                          np.array(length_scale), 
                          prior_var, 
                          noise_ratio])
            
        y_gps = []
        for y_gp in configs["gps"][1]:
            noisy, coef, length_scale, prior_var, noise_ratio = y_gp
            y_gps.append([noisy, 
                          np.array(coef), 
                          np.array(length_scale), 
                          prior_var, 
                          noise_ratio])

        self.gps = (x_gps, y_gps)




class Adaptor_arc(Adaptor):
    def __init__(self, gps=None, weights=None):
        self.kernel = Matern()
        self.gps = gps
        if weights is not None:
            self.prior_weights = weights.copy()
            self.posterior_weights = weights.copy()
            self.posteriors = [(i, weights[i]) for i in range(len(weights))]
        self.v_data = []
        self.w_data = []
        self.phi_data = []
    

    def read_data(self, data):
        """
        data: 3 x 7 or 3 x 1 x 7 array
        """
        v_data, w_data, phi_data = data
        self.v_data.append(v_data.copy().reshape(1, -1))
        self.w_data.append(w_data.copy().reshape(1, -1))
        self.phi_data.append(phi_data.copy().reshape(1, -1))
        v_data = np.concatenate(self.v_data, axis=0)
        w_data = np.concatenate(self.w_data, axis=0)
        phi_data = np.concatenate(self.phi_data, axis=0)
        v_coor = v_data[:, :-2]
        v_noise = v_data[:, -2]
        v_target = v_data[:, -1].reshape(-1, 1)
        w_coor = w_data[:, :-2]
        w_noise = w_data[:, -2]
        w_target = w_data[:, -1].reshape(-1, 1)
        phi_coor = phi_data[:, :-2]
        phi_noise = phi_data[:, -2]
        phi_target = phi_data[:, -1].reshape(-1, 1)

        v_log_likelihoods = np.zeros(len(self.prior_weights), dtype=np.float32)
        w_log_likelihoods = np.zeros(len(self.prior_weights), dtype=np.float32)
        phi_log_likelihoods = np.zeros(len(self.prior_weights), dtype=np.float32)
        v_gps, w_gps, phi_gps = self.gps

        for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(v_gps):
            diff = v_target - v_coor @ coef
            prior_var += 1e-6
            if noisy:
                log_likelihood = -0.5*(diff / (noise_ratio*v_noise.reshape(-1, 1)*prior_var + prior_var)).T @ diff
                log_likelihood -= 0.5*np.sum(np.log(noise_ratio*v_noise*prior_var + prior_var))
            else:
                cov = self.kernel(v_coor, length_scales=length_scale) + np.diag(v_noise)*noise_ratio
                cov *= prior_var
                log_likelihood = -0.5*diff.T @ lg.inv(cov) @ diff
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                log_likelihood -= 0.5*log_det
            v_log_likelihoods[index] = log_likelihood
        
        for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(w_gps):
            diff = w_target - w_coor @ coef
            prior_var += 1e-6
            if noisy:
                log_likelihood = -0.5*(diff / (noise_ratio*w_noise.reshape(-1, 1)*prior_var + prior_var)).T @ diff
                log_likelihood -= 0.5*np.sum(np.log(noise_ratio*w_noise*prior_var + prior_var))
            else:
                cov = self.kernel(w_coor, length_scales=length_scale) + np.diag(w_noise)*noise_ratio
                cov *= prior_var
                log_likelihood = -0.5*diff.T @ lg.inv(cov) @ diff
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                log_likelihood -= 0.5*log_det
            w_log_likelihoods[index] = log_likelihood
        
        for index, (noisy, coef, length_scale, prior_var, noise_ratio) in enumerate(phi_gps):
            diff = phi_target - phi_coor @ coef
            prior_var += 1e-6
            if noisy:
                log_likelihood = -0.5*(diff / (noise_ratio*phi_noise.reshape(-1, 1)*prior_var + prior_var)).T @ diff
                log_likelihood -= 0.5*np.sum(np.log(noise_ratio*phi_noise*prior_var + prior_var))
            else:
                cov = self.kernel(phi_coor, length_scales=length_scale) + np.diag(phi_noise)*noise_ratio
                cov *= prior_var
                log_likelihood = -0.5*diff.T @ lg.inv(cov) @ diff
                sign, log_det = lg.slogdet(cov)
                log_det = log_det if sign else -100
                log_likelihood -= 0.5*log_det
            phi_log_likelihoods[index] = log_likelihood
        
        log_likelihoods = v_log_likelihoods + w_log_likelihoods + phi_log_likelihoods
        log_scale = np.max(log_likelihoods)
        log_likelihoods -= np.max(log_likelihoods)
        posterior_weights = np.exp(log_likelihoods) * self.prior_weights
        log_scale += np.log(np.sum(posterior_weights))
        posterior_weights /= np.sum(posterior_weights)

        self.posterior_weights = posterior_weights.copy()
        sorted_weights = np.argsort(-posterior_weights)
        indexes = []
        total_weights = 0
        for index in sorted_weights:
            indexes.append(index)
            total_weights += posterior_weights[index]
            if total_weights > 0.9:
                break
        self.posteriors = [(index, posterior_weights[index]/total_weights) for index in indexes]

        return log_scale


    def predict(self, coors, baselines):
        """
        coors: 3 x N x 5 array
        baselines: 3 x N or 3 x N x 1 array
        """
        # v_coors, w_coors, phi_coors = coors
        # v_baselines, w_baselines, phi_baselines = baselines

        v_coors = coors[0].copy() 
        w_coors = coors[1].copy()
        phi_coors = coors[2].copy()
        v_baselines = baselines[0].copy()
        w_baselines = baselines[1].copy()
        phi_baselines = baselines[2].copy()

        v_gps, w_gps, phi_gps = self.gps
        v = v_baselines.reshape(-1, 1)
        w = w_baselines.reshape(-1, 1)
        phi = phi_baselines.reshape(-1, 1)

        if not self.v_data:
            for index, weight in self.posteriors:
                v_coef = v_gps[index][1]
                w_coef = w_gps[index][1]
                phi_coef = phi_gps[index][1]
                v += weight*(v_coors @ v_coef)
                w += weight*(w_coors @ w_coef)
                phi += weight*(w_coors @ phi_coef)

            w = np.where(np.isclose(w, 0), 1e-6, w)
            sin_theta = np.sin(w*4 + phi)
            cos_theta = np.cos(w*4 + phi)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            predicted_x = v/w * (sin_theta - sin_phi)
            predicted_y = v/w * (cos_phi - cos_theta)
            return predicted_x, predicted_y

        v_data = np.concatenate(self.v_data, axis=0)
        w_data = np.concatenate(self.w_data, axis=0)
        phi_data = np.concatenate(self.phi_data, axis=0)
        v_coor = v_data[:, :-2]
        v_noise = v_data[:, -2]
        v_target = v_data[:, -1].reshape(-1, 1)
        w_coor = w_data[:, :-2]
        w_noise = w_data[:, -2]
        w_target = w_data[:, -1].reshape(-1, 1)
        phi_coor = phi_data[:, :-2]
        phi_noise = phi_data[:, -2]
        phi_target = phi_data[:, -1].reshape(-1, 1)

        for index, weight in self.posteriors:
            v_noisy, v_coef, v_length_scale, _, v_noise_ratio = v_gps[index]
            if v_noisy:
                v += weight*(v_coors @ v_coef)
            else:
                diff_v = v_target - v_coor @ v_coef
                sigm22 = self.kernel(v_coor, length_scales=v_length_scale) + np.diag(v_noise)*v_noise_ratio
                sigma12 = self.kernel(v_coors, v_coor, length_scales=v_length_scale)
                v += weight*(v_coors @ v_coef + sigma12 @ lg.inv(sigm22) @ diff_v)

            w_noisy, w_coef, w_length_scale, _, w_noise_ratio = w_gps[index]
            if w_noisy:
                w += weight*(w_coors @ w_coef)
            else:
                diff_w = w_target - w_coor @ w_coef
                sigm22 = self.kernel(w_coor, length_scales=w_length_scale) + np.diag(w_noise)*w_noise_ratio
                sigma12 = self.kernel(w_coors, w_coor, length_scales=w_length_scale)
                w += weight*(w_coors @ w_coef + sigma12 @ lg.inv(sigm22) @ diff_w)

            phi_noisy, phi_coef, phi_length_scale, _, phi_noise_ratio = phi_gps[index]
            if phi_noisy:
                phi += weight*(phi_coors @ phi_coef)
            else:
                diff_phi = phi_target - phi_coor @ phi_coef
                sigm22 = self.kernel(phi_coor, length_scales=phi_length_scale) + np.diag(phi_noise)*phi_noise_ratio
                sigma12 = self.kernel(phi_coors, phi_coor, length_scales=phi_length_scale)
                phi += weight*(phi_coors @ phi_coef + sigma12 @ lg.inv(sigm22) @ diff_phi)

        w = np.where(np.isclose(w, 0), 1e-6, w)
        sin_theta = np.sin(w*4 + phi)
        cos_theta = np.cos(w*4 + phi)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        predicted_x = v/w * (sin_theta - sin_phi)
        predicted_y = v/w * (cos_phi - cos_theta)

        return predicted_x, predicted_y
    

    def save(self, path="adaptor_arc.json"):
        configs = {"weights": self.prior_weights.tolist()}
        v_gps, w_gps, phi_gps = self.gps
        new_v_gps = []
        new_w_gps = []
        new_phi_gps = []

        for v_gp in v_gps:
            noisy, coef, length_scale, prior_var, noise_ratio = v_gp
            new_v_gps.append((noisy, 
                              coef.tolist(), 
                              length_scale.tolist(), 
                              float(prior_var), 
                              float(noise_ratio)))
        
        for w_gp in w_gps:
            noisy, coef, length_scale, prior_var, noise_ratio = w_gp
            new_w_gps.append((noisy, 
                              coef.tolist(), 
                              length_scale.tolist(), 
                              float(prior_var), 
                              float(noise_ratio)))
        
        for phi_gp in phi_gps:
            noisy, coef, length_scale, prior_var, noise_ratio = phi_gp
            new_phi_gps.append((noisy, 
                                coef.tolist(), 
                                length_scale.tolist(), 
                                float(prior_var), 
                                float(noise_ratio)))
        
        configs["gps"] = (new_v_gps, new_w_gps, new_phi_gps)

        with open(path, "w") as file:
            json.dump(configs, file)



    def load(self, path):
        self.v_data = []
        self.w_data = []
        self.phi_data = []
        with open(path, "r") as file:
            configs = json.load(file)
        
        weights = np.array(configs["weights"])
        self.prior_weights = weights.copy()
        self.posterior_weights = weights.copy()
        self.posteriors = [(i, weights[i]) for i in range(len(weights))]

        v_gps = []
        for v_gp in configs["gps"][0]:
            noisy, coef, length_scale, prior_var, noise_ratio = v_gp
            v_gps.append([noisy, 
                          np.array(coef), 
                          np.array(length_scale), 
                          prior_var, 
                          noise_ratio])
            
        w_gps = []
        for w_gp in configs["gps"][1]:
            noisy, coef, length_scale, prior_var, noise_ratio = w_gp
            w_gps.append([noisy, 
                          np.array(coef), 
                          np.array(length_scale), 
                          prior_var, 
                          noise_ratio])
        
        phi_gps = []
        for phi_gp in configs["gps"][2]:
            noisy, coef, length_scale, prior_var, noise_ratio = phi_gp
            phi_gps.append([noisy, 
                            np.array(coef), 
                            np.array(length_scale), 
                            prior_var, 
                            noise_ratio])

        self.gps = (v_gps, w_gps, phi_gps)





class Adaptor_rte(Adaptor):
    def __init__(self, v=2.5):
        self.kernel = Matern(length_scales=np.ones(2, dtype=np.float32), 
                             v=v)
        self.x_data = []
        self.y_data = []
    

    def read_data(self, data):
        """
        data: 2 x 3 or 2 x 1 x 3 array
        """
        x_data, y_data = data
        self.x_data.append(x_data.copy().reshape(1, -1))
        self.y_data.append(y_data.copy().reshape(1, -1))
        x_data = np.concatenate(self.x_data, axis=0)
        y_data = np.concatenate(self.y_data, axis=0)
        coor = x_data[:, :-1] / 9.6 + 0.5
        x_target = x_data[:, -1].reshape(-1, 1)
        y_target = y_data[:, -1].reshape(-1, 1)

        cov = self.kernel(coor)*4 + np.eye(len(coor))*0.01
        inverse_cov = lg.inv(cov)
        self.inverse_cov = inverse_cov
        log_likelihood = -0.5*x_target.T @ inverse_cov @ x_target
        log_likelihood -= 0.5*y_target.T @ inverse_cov @ y_target

        sign, log_det = lg.slogdet(cov)
        log_det = log_det if sign else -100
        log_likelihood -= log_det
        
        return float(log_likelihood)


    def predict(self, coors):
        """
        coors: N x 2 array
        """
        x_baselines = coors[:, 0].copy().reshape(-1, 1)
        y_baselines = coors[:, 1].copy().reshape(-1, 1)

        if not self.x_data:
            return x_baselines, y_baselines
        
        x_data = np.concatenate(self.x_data, axis=0)
        y_data = np.concatenate(self.y_data, axis=0)
        coor = x_data[:, :-1] / 9.6 + 0.5
        x_target = x_data[:, -1].reshape(-1, 1)
        y_target = y_data[:, -1].reshape(-1, 1)

        matrix = 4*self.kernel((coors/9.6 + 0.5), coor) @ self.inverse_cov
        predicted_x = x_baselines + matrix @ x_target
        predicted_y = y_baselines + matrix @ y_target

        return predicted_x, predicted_y



class Combined_adaptor_xy(Adaptor):
    def __init__(self, adaptor_xy=None, alpha=1, N=0):
        self.configs = {"alpha": alpha, "N": N}
        self.adaptor_xy = adaptor_xy
        self.adaptor_rte = Adaptor_rte(v=0.5)
        self.prior_weights = np.array([N/(alpha + N), alpha/(alpha + N)])
        self.weights = self.prior_weights.copy()


    def read_data(self, coors, baselines, results):
        """
        coors: 2 x 5
        baselines: 5 x 1    <- (x, y, v, w, phi)
        results: 5 x 1    <- (x, y, v, w, phi)
        """
        targets = (results - baselines)[:2]
        log_likelihood_xy = self.adaptor_xy.read_data(np.hstack((coors, targets)))
        baseline_xy = np.zeros((2, 2))
        baseline_xy[:, :2] = baselines[:2, 0]
        log_likelihood_rte = self.adaptor_rte.read_data(np.hstack((baseline_xy, targets)))
        try:
            log_likelihoods = np.array([log_likelihood_xy, log_likelihood_rte])
        except Exception:
            print(log_likelihood_xy, log_likelihood_rte)
            raise
        log_likelihoods -= np.max(log_likelihoods)
        self.weights = np.exp(log_likelihoods) * self.prior_weights
        self.weights /= np.sum(self.weights)
    

    def predict(self, coors, baselines):
        """
        coors: 2 x N x 5
        baselines: 5 x N    <- (x, y, v, w, phi)
        """
        predicted_x_xy, predicted_y_xy = self.adaptor_xy.predict(coors, baselines[:2, :])
        predicted_x_rte, predicted_y_rte = self.adaptor_rte.predict(baselines[:2, :].T)
        weight_xy, weight_rte = self.weights
        predicted_x = weight_xy*predicted_x_xy + weight_rte*predicted_x_rte
        predicted_y = weight_xy*predicted_y_xy + weight_rte*predicted_y_rte
        return predicted_x, predicted_y


    def save(self, dir_path=None):
        dir_path = "combined_xy" if dir_path is None else dir_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        with open(os.path.join(dir_path, "configs.json"), "w") as file:
            json.dump(self.configs, file)
        self.adaptor_xy.save(path=os.path.join(dir_path, "adaptor_xy.json"))
    

    def load(self, dir_path):
        with open(os.path.join(dir_path, "configs.json"), "r") as file:
            configs = json.load(file)
        N = configs["N"]
        alpha = configs["alpha"]
        self.configs = configs.copy()
        self.prior_weights = np.array([N/(alpha + N), alpha/(alpha + N)])
        self.weights = self.prior_weights.copy()
        self.adaptor_xy = Adaptor_xy()
        self.adaptor_xy.load(os.path.join(dir_path, "adaptor_xy.json"))




class Combined_adaptor_arc(Adaptor):
    def __init__(self, adaptor_arc=None, alpha=1, N=0):
        self.configs = {"alpha": alpha, "N": N}
        self.adaptor_arc = adaptor_arc
        self.adaptor_rte = Adaptor_rte(v=0.5)
        self.prior_weights = np.array([N/(alpha + N), alpha/(alpha + N)])
        self.weights = self.prior_weights.copy()


    def read_data(self, coors, baselines, results):
        """
        coors: 3 x 5
        baselines: 5 x 1    <- (x, y, v, w, phi)
        results: 5 x 1    <- (x, y, v, w, phi)
        """
        xy_targets = (results - baselines)[:2]
        arc_targets = (results - baselines)[2:]

        v, w, phi = results[2:, 0]
        sin_theta = math.sin(w*4 + phi)
        cos_theta = math.cos(w*4 + phi)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        x_ = v/w*(sin_theta - sin_phi)
        y_ = v/w*(cos_phi - cos_theta)

        noise_v = v**2 / (x_**2 + y_**2 + 1e-6) + 1e-6
        noise_w = (w**2 + 0.01) / ((4*v*cos_theta - x_)**2 + (4*v*sin_theta - y_)**2 + 1e-8)
        noise_phi = 1 / (x_**2 + y_**2 + 16e-6*v**2)

        noises = np.array([noise_v, noise_w, noise_phi]).reshape(3, 1)
        log_likelihood_arc = self.adaptor_arc.read_data(np.hstack((coors, noises, arc_targets)))
        baseline_xy = np.zeros((2, 2))
        baseline_xy[:, :2] = baselines[:2, 0]
        log_likelihood_rte = self.adaptor_rte.read_data(np.hstack((baseline_xy, xy_targets)))

        log_likelihoods = np.array([log_likelihood_arc, log_likelihood_rte])
        log_likelihoods -= np.max(log_likelihoods)
        self.weights = np.exp(log_likelihoods) * self.prior_weights
        self.weights /= np.sum(self.weights)
    

    def predict(self, coors, baselines):
        """
        coors: 3 x N x 5
        baselines: 5 x N    <- (x, y, v, w, phi)
        """
        predicted_x_arc, predicted_y_arc = self.adaptor_arc.predict(coors, baselines[2:, :])
        predicted_x_rte, predicted_y_rte = self.adaptor_rte.predict(baselines[:2, :].T)
        weight_arc, weight_rte = self.weights
        predicted_x = weight_arc*predicted_x_arc + weight_rte*predicted_x_rte
        predicted_y = weight_arc*predicted_y_arc + weight_rte*predicted_y_rte
        return predicted_x, predicted_y


    def save(self, dir_path=None):
        dir_path = "combined_arc" if dir_path is None else dir_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        with open(os.path.join(dir_path, "configs.json"), "w") as file:
            json.dump(self.configs, file)
        self.adaptor_arc.save(path=os.path.join(dir_path, "adaptor_arc.json"))
    

    def load(self, dir_path):
        with open(os.path.join(dir_path, "configs.json"), "r") as file:
            configs = json.load(file)
        N = configs["N"]
        alpha = configs["alpha"]
        self.configs = configs.copy()
        self.prior_weights = np.array([N/(alpha + N), alpha/(alpha + N)])
        self.weights = self.prior_weights.copy()
        self.adaptor_arc = Adaptor_arc()
        self.adaptor_arc.load(os.path.join(dir_path, "adaptor_arc.json"))






