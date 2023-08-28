import numpy as np
import jax.numpy as jnp
import numpy.linalg as nlg
import jax.numpy.linalg as lg
from scipy.optimize import minimize
# from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
import math
from Matern import Matern
from jax import random

key = random.PRNGKey(0)


data_x = random.uniform(key, (24, 2), minval=-5, maxval=5)
data_y = data_x @ jnp.array([[2.5], [1]]) + jnp.prod(jnp.sin(data_x), axis=1, keepdims=True) + 1.2



def negative_log_likelihood(params, data_x, diff_y):
    length_scale = params[:-2]
    prior_var, noise_var = params[-2:]

    d = jnp.sqrt(
            jnp.sum(
            jnp.square((data_x[:, None, :] - data_x[None, :, :]) / length_scale[None, None, :]), 
            axis=-1)) * math.sqrt(5) * 0.5
    cov = (1 + d + d**2 / 3) * jnp.exp(-d) * prior_var + jnp.eye(24) * noise_var
    
    result =  0.5 * diff_y.T @ lg.inv(cov) @ diff_y + 0.5 * jnp.log(lg.det(cov))
    return result[0, 0]



def fixed_negative_log_likelihood(data_x, diff_y):
    d_matrix = jnp.square(data_x[:, None, :] - data_x[None, :, :]) * 1.25
    size = len(diff_y)
    ndiff_y = np.float32(diff_y)
    diagonal = jnp.eye(size)
    nd_matrix = np.float32(d_matrix)

    def fn(params):
        length_scale = jnp.square(params[:-2].reshape(-1, 1))
        prior_var, noise_var = params[-2:]

        d = jnp.sqrt(d_matrix @ length_scale).reshape(size, size)
        cov = (1 + d + d**2 / 3) * jnp.exp(-d) * prior_var + diagonal * noise_var
        
        result =  0.5 * diff_y.T @ lg.inv(cov) @ diff_y + 0.5 * jnp.log(lg.det(cov))
        return result[0, 0]


    def numpy_fn(params):
        length_scale = np.square(params[:-2].reshape(-1, 1))
        prior_var, noise_var = params[-2:]

        d = np.sqrt(nd_matrix @ length_scale).reshape(size, size)
        cov = (1 + d + d**2 / 3) * np.exp(-d) * prior_var + np.eye(size) * noise_var
        
        result =  0.5 * ndiff_y.T @ nlg.inv(cov) @ ndiff_y + 0.5 * np.log(nlg.det(cov))
        return np.float64(result[0, 0])
    
    return jit(fn), numpy_fn

    # jit_fn = jit(fn)

    # jit_fn(np.ones(4))

    # def numpy_fn(params):
    #     return np.float64(jit_fn(params))

    # return numpy_fn


def fixed_jac(fn, vmap_fn, dim):
    param_matrix = jnp.ones((dim, dim)) + jnp.eye(dim) * 1e-5

    def jac(params):
        original = fn(params)
        new_params = params.reshape(1, -1) * param_matrix
        step_diff = jnp.diagonal(new_params - params)
        step_diff = jnp.where(step_diff == 0, 1e-6, step_diff)

        new_values = vmap_fn(new_params)
        gradient = (new_values - original) / step_diff
        return np.array(gradient)

    return jac


diff_y = data_y-1.2 - data_x @ jnp.array([[2.5], [1]])


nll, numpy_nll = fixed_negative_log_likelihood(data_x, diff_y)
vmap_nll = vmap(nll)
jac = fixed_jac(nll, vmap_nll, 4)
# grad_fn = grad(nll)

params = jnp.array([1,1, np.var(diff_y), 0.01*np.var(diff_y)])
a = nll(params)
# print(a)

# b = jac(params)
# print(b)

# params = jnp.array([1+1e-5, 1, np.var(diff_y), np.var(diff_y)])
# print(nll(params))
# print(jac(params))

# b = nll(params)
# # print( (b - a) * 1e5)
# print(grad_fn(params))




# print(nll(params))
# print(grad_fn(params))

# print(negative_log_likelihood(jnp.array([1, 1, 1, 0.01]), data_x, diff_y))




a = minimize(numpy_nll, 
             np.array([1,1, np.var(diff_y), 0.01*np.var(diff_y)]), 
            #  args=(data_x, diff_y), 
             jac=jac,
            #  method="bfgs"
             bounds=np.array([
                 [1e-5, 1e4],
                 [1e-5, 1e4],
                 [1e-6, 1e4],
                 [1e-8, 1]])
                 ) 

print(a.x)
print(a.fun)

# print(negative_log_likelihood(np.array([0.5, 0.5, np.var(diff_y), 1e-8]), data_x, diff_y))

# print(np.var(diff_y))
