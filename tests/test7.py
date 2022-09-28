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
size = int(10e6)
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
loss_weights1 = [0.5, 0.5]
metric_weights1 = [1.0, 0.0]
n_inputs = 1
re_uploading_pqc1 = pqc.PQC(n_inputs,n_layers = 1)

re_uploading_pqc1.compile()

loss_train_history1, loss_test_history1, metric_train_history1, metric_test_history1 = re_uploading_pqc1.fit(
                            input_train/2,output_train,input_test/2,output_test,
                            epochs = 200,loss_weights = loss_weights1,metric_weights = metric_weights1
                            )


# Test 2
loss_weights2 = [1.0, 0.0]
metric_weights2 = [1.0, 0.0]
n_inputs = 1
re_uploading_pqc = pqc.PQC(n_inputs,n_layers = 1)

re_uploading_pqc.compile()

loss_train_history2, loss_test_history2, metric_train_history2, metric_test_history2 = re_uploading_pqc2.fit(
                            input_train/2,output_train,input_test/2,output_test,
                            epochs = 200,loss_weights = loss_weights2,metric_weights = metric_weights2
                            )


# Plot
plt.plot(metric_test_history1, label = "Gradients")
plt.plot(metric_test_history2, label = "Exact")
plt.legend()
plt.grid()
plt.savefig('images/test7.png')
plt.show()

# Plot two surfaces
