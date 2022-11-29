########################################################
# Confirmation of appearence of frequencies in the circuit: no base frequency
########################################################

import sys
sys.path.append("./")
sys.path.append("../")
import time

from jax import numpy as jnp
import numpy as np
import src.pqc as pqc
from src.arquitectures import ConstantArquitecture, FourierAnsatz
from src.losses import spvsd_gradients_loss, spvsd_loss
import matplotlib.pyplot as plt

def cos_expansion(x,z):
    amplitude = np.abs(z)
    phase = np.angle(z)

    N = len(z)
    result = np.ones_like(x)*amplitude[0]*np.cos(phase[0])/2
    for i in range(1,N):
        result = result+amplitude[i]*np.cos(i*x+phase[i])
    return result



n = 4
N = 2**n


amplitude = np.abs(np.random.randn(2**n))
amplitude[0] = 0
amplitude = amplitude/np.sqrt(np.sum(np.square(amplitude)))
phase = np.random.randn(2**n)
phase[0] = 0
z = amplitude*np.exp(1j*phase)


# Define market space
N_points = 100
x = np.linspace(0,1,N_points)
y = cos_expansion(x,z)

# Test 1
n_qubits = n
n_inputs = 1
n_layers = 1

arquitecture = FourierAnsatz(n_qubits = n_qubits,n_inputs = n_inputs, n_layers = n_layers)
#arquitecture = ConstantArquitecture(n_qubits = n_qubits,n_inputs = n_inputs, n_layers = n_layers)
start = time.time()
re_uploading_pqc = pqc.PQC(arquitecture = arquitecture)

# Compilation
loss = spvsd_loss
re_uploading_pqc.compile(loss = loss, generator = z)
re_uploading_pqc.plot()

# Train
x_ = jnp.array(x.reshape((N_points,1)))
y_ = jnp.array(y.reshape((N_points,1)))
#metric_train_history, metric_test_history = re_uploading_pqc.fit(
#                            x_,y_,x_,y_,
#                            epochs = 1
#                            )
#
end = time.time()
print("Time: ", end-start)
# Evaluate
y_predict = re_uploading_pqc.predict(x_)

plt.plot(x_,y_,label = "Original")
plt.plot(x,y_predict[:,0],label = "Approximation")
#plt.ylim(-1, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
