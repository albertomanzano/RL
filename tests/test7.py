import sys
sys.path.append("./")
sys.path.append("../")

from pennylane import numpy as np
from agents.pqc import PQC, mesh

import matplotlib.pyplot as plt


# Data
size = 10
x = np.linspace(-1,1,size,requires_grad = False)
input_train, input_test = mesh((x,), test_size = 0.2)

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
n_inputs = 1
re_uploading_pqc = PQC(n_inputs,n_layers = 1)
re_uploading_pqc.plot()
re_uploading_pqc.compile()

loss_weights = [0.5, 0.5]
metric_weights = [1.0, 0.0]
loss_train_history1, loss_test_history1, metric_train_history1, metric_test_history1 = re_uploading_pqc.fit(input_train,output_train,
                                                                                                            input_test,output_test,
                                                                                                            epochs = 50,
                                                                                                            loss_weights = loss_weights, 
                                                                                                            metric_weights = metric_weights)

print("\n\n#######################################################################################################\n\n")
# Test 2
n_inputs = 1
re_uploading_pqc = PQC(n_inputs, n_layers = 1)
re_uploading_pqc.compile()

loss_weights = [1.0,0.0]
metric_weights = [1.0,0.0]
loss_train_history2, loss_test_history2, metric_train_history2, metric_test_history2 = re_uploading_pqc.fit(input_train,output_train,
                                                                                                            input_test,output_test,
                                                                                                            epochs = 50,
                                                                                                            loss_weights = loss_weights, 
                                                                                                            metric_weights = metric_weights)

# Plot
plt.plot(metric_test_history1, label = "Gradients")
plt.plot(metric_test_history2, label = "No gradients")
plt.legend()
plt.grid()
plt.savefig('images/2d.png')
plt.show()

# Plot two surfaces
