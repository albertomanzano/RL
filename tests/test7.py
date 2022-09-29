import sys
sys.path.append("./")
sys.path.append("../")

#from pennylane import numpy as np
from jax import numpy as np
import agents.pqc as pqc
from multiprocessing import Pool
import multiprocessing
import matplotlib.pyplot as plt

###########
# Data
##########

# Define inputs
size = int(10e3)
x = np.linspace(-np.pi/2,np.pi/2,size,)
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

# Test 1
loss_weights1 = np.array([0.5, 0.5])
metric_weights1 = np.array([1.0, 0.0])
n_inputs = 1
re_uploading_pqc1 = pqc.PQC(n_inputs,n_layers = 1)

re_uploading_pqc1.compile(loss_weights = loss_weights1, metric_weights = metric_weights1)

metric_train_history1, metric_test_history1 = re_uploading_pqc1.fit(
                            input_train/2,output_train,input_test/2,output_test,
                            epochs = 100
                            )


# Test 2
loss_weights2 = np.array([1.0, 0.0])
metric_weights2 = np.array([1.0, 0.0])
n_inputs = 1
re_uploading_pqc2 = pqc.PQC(n_inputs,n_layers = 1)

re_uploading_pqc2.compile(loss_weights = loss_weights2, metric_weights = metric_weights2)

metric_train_history2, metric_test_history2 = re_uploading_pqc2.fit(
                            input_train/2,output_train,input_test/2,output_test,
                            epochs = 100
                            )


# Plot
fig, ax = plt.subplots(nrows=1, ncols=2)
# Metric test values
ax[0].plot(metric_test_history1[1:,0,0], label = "Gradients")
ax[0].plot(metric_test_history2[1:,0,0], label = "Exact")
ax[0].set_yscale('log')
ax[0].set_title("Test")
ax[0].legend()
ax[0].grid()
# Metric train values
ax[1].plot(metric_train_history1[1:,0,0], label = "Gradients")
ax[1].plot(metric_train_history2[1:,0,0], label = "Exact")
ax[1].set_yscale('log')
ax[1].set_title("Train")
ax[1].legend()
ax[1].grid()
plt.savefig('images/test7.png')
plt.show()

# Plot two surfaces
