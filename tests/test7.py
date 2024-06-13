################################################
# Training with gradient for the trigonometric
###############################################

import sys
sys.path.append("./")
sys.path.append("../")

#from pennylane import numpy as np
from jax import numpy as np
import agents.pqc as pqc
from agents.losses import spvsd_gradients_loss, spvsd_loss
import matplotlib.pyplot as plt

###########
# Data
##########

# Define inputs
size = int(1e3)
x = np.linspace(0,2*np.pi,size,)
input_train, input_test = pqc.mesh((x,), test_size = 0.2)

# Define outputs
output_train = np.sin(input_train[:,0])
output_test = np.sin(input_test[:,0])
output_train = output_train[:,None]
output_test = output_test[:,None]

output_train_grad0 = np.cos(input_train[:,0])[:,None]

output_test_grad0 = np.cos(input_test[:,0])[:,None]

output_train = np.hstack((output_train,output_train_grad0))
output_test = np.hstack((output_test,output_test_grad0))

# Epochs
epochs = 1000

# Test 1
n_inputs = 1
loss1 = lambda func, weights, x, y: spvsd_gradients_loss(func, weights, x, y ,np.array([0.5,0.5]))

re_uploading_pqc1 = pqc.PQC(n_inputs,n_layers = 1)
re_uploading_pqc1.compile(loss = loss1)
metric_train_history1, metric_test_history1 = re_uploading_pqc1.fit(
                            input_train,output_train,input_test,output_test,
                            epochs = epochs
                            )


# Test 2
n_inputs = 1
loss2 = spvsd_loss

re_uploading_pqc2 = pqc.PQC(n_inputs,n_layers = 1)
re_uploading_pqc2.compile(loss = loss2)
metric_train_history2, metric_test_history2 = re_uploading_pqc2.fit(
                            input_train,output_train,input_test,output_test,
                            epochs = epochs
                            )


# Plot test convergence
epoch_number = np.arange(epochs)
fig, ax = plt.subplots(nrows=1, ncols=2)
# Metric test values
ax[0].plot(epoch_number,metric_test_history1[1:,0,0], label = "Gradients")
ax[0].plot(epoch_number,metric_test_history2[1:,0,0], label = "Exact")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("MSE")
ax[0].set_yscale('log')
ax[0].set_title("Test")
ax[0].legend()
ax[0].grid()
# Metric train values
ax[1].plot(epoch_number,metric_train_history1[1:,0,0], label = "Gradients")
ax[1].plot(epoch_number,metric_train_history2[1:,0,0], label = "Exact")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("MSE")
ax[1].set_yscale('log')
ax[1].set_title("Train")
ax[1].legend()
ax[1].grid()
plt.savefig('images/test7.png')
plt.show()

# Plot results 
output_test1 = re_uploading_pqc1.predict(input_test)
output_test2 = re_uploading_pqc2.predict(input_test)
fig2, ax2 = plt.subplots(nrows=1, ncols=1)
ax2.scatter(input_test,output_test1[:,0],label = "Gradient")
ax2.scatter(input_test,output_test2[:,0],label = "No Gradient")
ax2.scatter(input_test,output_test[:,0],label = "Exact")
plt.legend()
plt.show()
