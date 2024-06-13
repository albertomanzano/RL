########################################################
# Confirmation of appearence of frequencies in the circuit: no base frequency
########################################################
# The mean of the function that we will approximate has to be zero
# The output must be normalised between -1 and 1

import sys
sys.path.append("./")
sys.path.append("../")

from jax import numpy as np
import matplotlib.pyplot as plt

import agents.pqc as pqc
from agents.losses import spvsd_gradients_loss, spvsd_loss
from agents.arquitectures import ConstantPureCos, ConstantPureSin, ConstantGeneral


def sin_expansion(x,base_frequency,coeff,shift = None):
    number_frequencies = len(coeff)
    number_points = len(x)
    if shift is None:
        shift = [ 0.0 for i in range(number_frequencies) ]
    
    values = np.ones(number_points)*coeff[0]
    for i in range(1,number_frequencies):
        values = values+coeff[i]*np.sin(base_frequency*i*x+shift[i])
    return values

def cos_expansion(x,base_frequency,coeff, shift = None):
    number_frequencies = len(coeff)
    number_points = len(x)
    if shift is None:
        shift = [ 0.0 for i in range(number_frequencies) ]
    
    values = np.ones(number_points)*coeff[0]
    for i in range(1,number_frequencies):
        values = values+coeff[i]*np.cos(base_frequency*i*x+shift[i])
    return values

def cos_expansion_derivative(x,base_frequency,coeff, shift = None):
    number_frequencies = len(coeff)
    number_points = len(x)
    if shift is None:
        shift = [ 0.0 for i in range(number_frequencies) ]
    
    values = np.ones(number_points)*coeff[0]
    for i in range(1,number_frequencies):
        values = values-coeff[i]*base_frequency*i*np.sin(base_frequency*i*x+shift[i])
    return values

# Domain
a = 0
b = 2*np.pi
N_points = 200
x = np.linspace(a,b,N_points)
domain = [[a,b]]

# Define market space
base_frequency = 1.
A =[0.0]+[0.0]+[0.5]+[0.0]+[0.5]+[0.0]+[0.5]+[0.0]+[0.5]
#shift = [0.1]*3+[0.3]
shift = None
y = cos_expansion(x,base_frequency,A, shift)
y_grad = cos_expansion_derivative(x,base_frequency,A, shift)

y = y-np.mean(y)
norm = np.max(np.abs(y)) 
y = y/norm
print(np.max(np.abs(y)))
y = y[:,None]
y_grad = y_grad/norm
y_grad = y_grad[:,None]
outputs = np.concatenate((y,y_grad), axis = 1)

# Test 1
n_inputs = 1
n_layers = 8
arquitecture = ConstantPureCos
loss = lambda func, weights, x, y: spvsd_gradients_loss(func, weights, x, y ,np.array([0.5,0.5]))
#loss = spvsd_loss

re_uploading_pqc = pqc.PQC(n_inputs,n_layers, base_frequency = base_frequency, arquitecture = arquitecture)
re_uploading_pqc.compile(loss = loss)
re_uploading_pqc.plot()
print(re_uploading_pqc.n_layers)

# Train
x_ = x.reshape((N_points,1))
y_ = y.reshape((N_points,1))
metric_train_history, metric_test_history = re_uploading_pqc.fit(
                            x_,outputs,x_,outputs,
                            epochs = 300
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
