########################################################
# Test of loss function with random samples from a distribution
########################################################

import sys
sys.path.append("./")
sys.path.append("../")
import time

from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ncx2, norm
import src.pqc as pqc
from src.arquitectures import HardwareEfficientAnsatz
from src.losses import spvsd_gradients_loss, spvsd_loss, max_loss, sample_density_loss
from src.metrics import MAE_metric, MSE_metric, one_metric



epochs = 100
a=-np.pi
b=np.pi
########################################################
# Gradients
########################################################

y_pred = []
y_pred_ = []

N_points_train = 1000
N_points_test = 1000

# Define market space
x_train = np.random.uniform(a,b,N_points_train) 
x_test = np.linspace(a,b,N_points_test) 

y_train = norm.cdf(x_train) 
y_train_ = norm.pdf(x_train)

y_test = norm.cdf(x_test) 
y_test_ = norm.pdf(x_test)

# Produce shape
x_train = jnp.array(x_train.reshape((N_points_train,1)))
y_train = np.concatenate((y_train[:,None],y_train_[:,None]),axis = 1)
y_train = jnp.array(y_train.reshape((N_points_train,2)))

x_test = jnp.array(x_test.reshape((N_points_test,1)))
y_test = np.concatenate((y_test[:,None],y_test_[:,None]),axis = 1)
y_test = jnp.array(y_test.reshape((N_points_test,2)))


# Test 1
n_inputs = 1
n_qubits = 3
n_layers = 3
base_frequency = np.pi/(b-a)
level = 2.0


arquitecture = HardwareEfficientAnsatz(n_inputs = n_inputs,n_qubits = n_qubits, n_layers = n_layers,base_frequency = base_frequency,level = level)
re_uploading_pqc = pqc.PQC(arquitecture = arquitecture)

# Compilation
metrics = {"MAE": MAE_metric}
loss_weights = jnp.array([1.0, 4.0])
loss = lambda func,weights,x,y: spvsd_gradients_loss(func,weights,x,y,loss_weights)
#loss = spvsd_gradients_loss
re_uploading_pqc.compile(loss = loss,metrics = metrics)
re_uploading_pqc.plot()

# Train history
metric_train_history, metric_test_history = re_uploading_pqc.fit(
                            x_train,y_train,x_test,y_test,
                            epochs = epochs,
                            )



y_predict = re_uploading_pqc.predict(x_test)
y_pred = y_predict[:,0]
y_pred_ = y_predict[:,1]

plt.plot(x_test,y_test[:,0],c = "g",label = "Target")
plt.plot(x_test,y_pred,c = "r",label = "Test")
plt.legend()
plt.show()

plt.plot(x_test,y_test[:,1],c = "g",label = "Target_")
plt.plot(x_test,y_pred_,c = "r",label = "Test_")
plt.legend()
plt.show()
