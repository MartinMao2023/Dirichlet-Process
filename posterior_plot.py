import numpy as np
import matplotlib.pyplot as plt
import math


np.random.seed(111)
x = np.linspace(0, 16, 320)
prior = np.exp(-0.5*(x - 4)**2) / math.sqrt(2*math.pi)*0.25 + np.exp(-0.5*(x - 10)**2/4) / math.sqrt(8*math.pi)*0.75


plt.plot(x, prior, linestyle="--", label="prior")
# plt.show()


data = np.random.normal(5.5, 1, 5).reshape(1, -1)

log_likelihood = -0.5*np.sum(((x[:, None] - data)/4)**2, axis=-1)

posterior = np.log(prior) + log_likelihood
posterior -= np.max(posterior)
posterior = np.exp(posterior)
posterior /= np.sum(posterior*0.05)

# plt.plot(x, log_likelihood)
plt.plot(x, posterior, label="posterior (5 data)")


data = np.random.normal(5.5, 1, 50).reshape(1, -1)

log_likelihood = -0.5*np.sum(((x[:, None] - data)/4)**2, axis=-1)

posterior = np.log(prior) + log_likelihood
posterior -= np.max(posterior)
posterior = np.exp(posterior)
posterior /= np.sum(posterior*0.05)
plt.plot(x, posterior, label="posterior (50 data)")

plt.legend(fontsize=14)
# plt.savefig("posterior_convergence.pdf", dpi=500)
plt.show()


