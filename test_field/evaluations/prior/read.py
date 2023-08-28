import numpy as np
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jax import grad, jit
import os

def sth(x):
    a = x[:5]
    b = x[-1]
    return jnp.sum(jnp.square(a)) + b

# sth_grad = grad(sth)
# x = jnp.arange(6, dtype=jnp.float32)
# print(sth_grad(x))

a = np.arange(5)
z = jnp.array(a)

print(z)
