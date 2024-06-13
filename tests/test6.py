##################################################
# Comparison RNN with standard 
##################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import numpy as np

from agents.losses import spvsd_gradients_loss, spvsd_loss
import agents.pqc as pqc

# Define domain
n = 10
t0 = -1
T = 1
t = np.linspace(t0,T,n)
dt = t[1]-t[0]
x = 0.1*t*t



###############################
# MRNN
###############################
base_frequency = 1.
n_inputs = 2
n_layers = 5
loss = spvsd_loss

re_uploading_pqc = pqc.PQC(n_inputs,n_layers, base_frequency)
re_uploading_pqc.compile(loss = loss)
re_uploading_pqc.plot()

# Define dataset
t_ = t.reshape((n,1))
x_ = x.reshape((n,1))

inputs = np.concatenate((t_[0:-1],x_[0:-1]), axis = 1)
outputs = x_[1:]
# Execution
metric_train_history, metric_test_history = re_uploading_pqc.fit(
                            inputs,outputs,inputs,outputs,
                            epochs = 200
                            )

temp = re_uploading_pqc.predict(inputs)
x_MRNN = np.zeros_like(t)
x_MRNN[0] = x[0]
x_MRNN[1:] = temp[:,0]

###############################
# NN
###############################
base_frequency = np.pi
n_inputs = 1
n_layers = 5
loss = spvsd_loss

re_uploading_pqc = pqc.PQC(n_inputs,n_layers, base_frequency)
re_uploading_pqc.compile(loss = loss)
re_uploading_pqc.plot()

# Define dataset
t_ = t.reshape((n,1))
x_ = x.reshape((n,1))

inputs = t_
outputs = x_
# Execution
metric_train_history, metric_test_history = re_uploading_pqc.fit(
                            inputs,outputs,inputs,outputs,
                            epochs = 200
                            )

temp = re_uploading_pqc.predict(inputs)
x_NN = temp[:,0]

###############################
# RNN
###############################
base_frequency = 1.
n_inputs = 1
n_layers = 1
loss = spvsd_loss

re_uploading_pqc = pqc.PQC(n_inputs,n_layers, base_frequency)
re_uploading_pqc.compile(loss = loss)
re_uploading_pqc.plot()

# Define dataset
t_ = t.reshape((n,1))
x_ = x.reshape((n,1))

inputs = x_[0:-1,0][:,None]
outputs = x_[1:,0][:,None]
# Execution
metric_train_history, metric_test_history = re_uploading_pqc.fit(
                            inputs,outputs,inputs,outputs,
                            epochs = 200
                            )

temp = re_uploading_pqc.predict(inputs)
x_RNN = np.zeros_like(t)
x_RNN[0] = x[0]
x_RNN[1:] = temp[:,0]

# Plot
plt.plot(t,x, label = "Exact")
plt.plot(t,x_RNN,label = "RNN")
plt.plot(t,x_MRNN,label = "MRNN")
plt.plot(t,x_NN,label = "NN")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.grid()
plt.show()
