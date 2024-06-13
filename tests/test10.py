#################################################################
# Training with gradient for the bi-dimensional gaussian function
#################################################################

import sys
sys.path.append("./")
sys.path.append("../")

from pennylane import numpy as np
import matplotlib.pyplot as plt

import agents.pqc as pqc
from agents.losses import spvsd_loss, spvsd_gradients_loss

# Data
size = 100
x = np.linspace(-1,1,size,requires_grad = False)
y = np.linspace(-1,1,size,requires_grad = False)
input_train, input_test = pqc.mesh((x,y), test_size = 0.2)

# Define outputs
output_train = np.exp(-input_train[:,0]**2-input_train[:,1]**2) 
output_test = np.exp(-input_test[:,0]**2-input_test[:,1]**2) 
output_train = output_train[:,None]
output_test = output_test[:,None]

output_train_grad0 = -2*input_train[:,0]*np.exp(-input_train[:,0]**2-input_train[:,1]**2)
output_train_grad0 = output_train_grad0[:,None]
output_train_grad1 = -2*input_train[:,1]*np.exp(-input_train[:,0]**2-input_train[:,1]**2)
output_train_grad1 = output_train_grad1[:,None]

output_test_grad0 = -2*input_test[:,0]*np.exp(-input_test[:,0]**2-input_test[:,1]**2)
output_test_grad0 = output_test_grad0[:,None]
output_test_grad1 = -2*input_test[:,1]*np.exp(-input_test[:,0]**2-input_test[:,1]**2)
output_test_grad1 = output_test_grad1[:,None]

output_train = np.hstack((output_train,output_train_grad0, output_train_grad1))
output_test = np.hstack((output_test,output_test_grad0,output_test_grad1))

# Net configuration
n_inputs = 2
n_layers = 4
epochs = 1000

# Test 1
loss1 = lambda func, weights, x, y: spvsd_gradients_loss(func, weights, x, y ,np.array([0.33,0.33,0.33]))

re_uploading_pqc1 = pqc.PQC(n_inputs = n_inputs,n_layers = n_layers)
re_uploading_pqc1.compile(loss = loss1)
metric_train_history1, metric_test_history1 = re_uploading_pqc1.fit(
                            input_train,output_train,input_test,output_test,
                            epochs = epochs
                            )

# Test 2
loss2 = spvsd_loss

re_uploading_pqc2 = pqc.PQC(n_inputs = n_inputs,n_layers = n_layers)
re_uploading_pqc2.compile(loss = loss2)
metric_train_history2, metric_test_history2 = re_uploading_pqc2.fit(
                            input_train,output_train,input_test,output_test,
                            epochs = epochs
                            )
# Plot
plt.plot(metric_test_history1[:,0,0], label = "Gradient")
plt.plot(metric_test_history2[:,0,0], label = "No gradient")
plt.legend()
plt.yscale("log")
plt.grid()
plt.savefig('results/images/test10.png')
plt.show()

# Plot two surfaces
