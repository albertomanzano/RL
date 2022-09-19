##################################################
# Confirmation of appearence of frequencies in the circuit
##################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./")
import matplotlib
#matplotlib.use('TkAgg')

from agents.re_uploading_pqc import ReUploadingPQC
from scipy.stats import qmc
from scipy.special import ndtr
import numpy as np
import tensorflow_quantum as tfq
import tensorflow as tf
import matplotlib.pyplot as plt

def expansion(x,base_frequency,coeff):
    number_frequencies = len(coeff)
    number_points = len(x)
    coeff = np.array(coeff)
    conj_coeff = np.conjugate(coeff)
    
    values = np.ones(number_points)*(coeff[0] + conj_coeff[0])*0.5
    for i in range(1,number_frequencies):
        exponent = np.complex128(base_frequency * i * x * 1j)
        values = values+coeff[i] * np.exp(exponent) + conj_coeff[i] * np.exp(-exponent)
    values = np.real(values)
    values = values/np.max(np.abs(values))
    return values

# Domain
a = 0
b = 2*np.pi
N_points = 200
x = np.linspace(a,b,N_points)
domain = [[a,b]]

# Define market space
base_frequency = 1
A = [0]*6+[0.15+0.15j]
y = expansion(x,base_frequency,A)
print("Reference Mean Absolute Error: ",np.mean(np.abs(y-np.mean(y))))

# Arquitecture
n_inputs = 1
n_outputs = 1
n_layers = 3
schedule = 'rx_linear'
entangling = 'cyclic'
arquitecture = 'rot'
repetitions = 1



# Initialization
model = ReUploadingPQC(n_inputs,n_outputs,n_layers,schedule,entangling,arquitecture,repetitions)
model.summary()

# Train
x_ = x.reshape((N_points,1))
y_ = y.reshape((N_points,1))
history = model.fit(x_,y_,epochs = 300,validation_split = 0.)

# Evaluate
y_predict = model(x_)

plt.plot(x,y,label = "Original")
plt.plot(x,y_predict,label = "Approximation")
plt.ylim(-1, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
