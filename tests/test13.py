import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr, erf
from jax import numpy as jnp

import agents.pqc as pqc
from agents.losses import spvsd_loss

def signal(x,A,P):
    return A*np.mod(x,P)/P

def signal_fourier(x,A,P,n):
    omega_0 = 2*np.pi/P
    result = np.ones_like(x)*A/2
    for i in range(1,n+1):
        result = result-A/(i*np.pi)*np.sin(omega_0*i*x)
    return result

# Domain
P = 1.
A = 1.
x = np.linspace(0,P,1000)

# Exact
exact = signal(x,A,P)
exact = exact-np.mean(exact)


n = 15
# Classical Fourier
fourier = signal_fourier(x,A,P,n)
fourier = fourier-np.mean(fourier)


# Quantum neural network
n_inputs = 1
epochs = 250
n_layers = n
base_frequency = 2*np.pi/P
re_uploading_pqc = pqc.PQC(n_inputs = n_inputs,n_layers = n_layers, base_frequency = base_frequency)

loss = lambda func, weights, x, y: spvsd_loss(func, weights, x, y)
re_uploading_pqc.compile(loss = loss)
re_uploading_pqc.plot()

inputs = jnp.array(x[:,None])
outputs = jnp.array(fourier[:,None])
metric_train_history, metric_test_history = re_uploading_pqc.fit(
                            inputs, outputs,inputs,outputs,
                            epochs = epochs
                            )
qnn = re_uploading_pqc.predict(inputs)

# Plot
plt.plot(x,exact, label = "Exact")
plt.plot(x,fourier, label = "Fourier")
plt.plot(x,qnn[:,0], label = "Qnn")
plt.grid()
plt.ylabel("Signal")
#plt.yscale("log")
plt.legend()
plt.show()

