########################################################
# Confirmation of appearence of frequencies in the circuit: base frequency
########################################################

import sys
sys.path.append("./")
sys.path.append("../")

from jax import numpy as np
import agents.pqc as pqc
from agents.losses import spvsd_gradients_loss, spvsd_loss
import matplotlib.pyplot as plt


def expansion(x,base_frequency,coeff):
    number_frequencies = len(coeff)
    number_points = len(x)
    
    values = np.ones(number_points)*coeff[0]
    for i in range(1,number_frequencies):
        values = values+coeff[i]*np.sin(base_frequency*i*x)
    return values

# Domain
a = 0
b = 2*np.pi
N_points = 200
x = np.linspace(a,b,N_points)
domain = [[a,b]]

# Define market space
base_frequency = 2.
A =[0]*2+[0.5]
y = expansion(x,base_frequency,A)

# Test 1
n_inputs = 1
n_layers = 2
loss = spvsd_loss

re_uploading_pqc = pqc.PQC(n_inputs,n_layers, base_frequency)
re_uploading_pqc.compile(loss = loss)
re_uploading_pqc.plot()
print(re_uploading_pqc.n_layers)

# Train
x_ = x.reshape((N_points,1))
y_ = y.reshape((N_points,1))
metric_train_history, metric_test_history = re_uploading_pqc.fit(
                            x_,y_,x_,y_,
                            epochs = 200
                            )

# Evaluate
y_predict = re_uploading_pqc.predict(x_)

plt.plot(x_,y_,label = "Original")
plt.plot(x,y_predict[:,0],label = "Approximation")
plt.ylim(-1, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
