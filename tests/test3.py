import sys
sys.path.append("/home/alberto/RL/RL")

from agents.pqc import PQC
from scipy.stats import qmc
from scipy.special import ndtr
import numpy as np
import tensorflow_quantum as tfq
import tensorflow as tf

def delta(tau,K,S,sigma,r):
    d1 = 1/(sigma*np.sqrt(tau))*(np.log(S/K)+(r+sigma*sigma/2)*tau)
    delta = ndtr(d1)
    return delta

# Define market space
strike = 1
r = 0.05
volatility = 0.5
tau_bounds = [0.1,1]
S_bounds = [0.1,3]


# Define grid of points
n = 10000
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=n)
l_bounds = [tau_bounds[0], S_bounds[0]]
u_bounds = [tau_bounds[1], S_bounds[1]]
x = qmc.scale(sample, l_bounds, u_bounds)
y = delta(x[:,0],strike,x[:,1],volatility,r)


# Convert to tensors
dataset = tf.data.Dataset.from_tensor_slices((x,y))
tf.data.experimental.save(dataset, "dataset2")
