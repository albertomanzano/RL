################################################
# Training with gradient for the BS
###############################################

import sys
sys.path.append("./")
sys.path.append("../")

#from pennylane import numpy as np
from jax import numpy as np
import agents.pqc as pqc
from agents.losses import spvsd_gradients_loss, spvsd_loss
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

#######################
# Auxiliary functions
#######################
def bs_call_price(
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    strike: float,
):
    first = np.log(s_0 / strike)
    positive = (risk_free_rate + volatility * volatility / 2) * maturity
    negative = (risk_free_rate - volatility * volatility / 2) * maturity
    d_1 = (first + positive) / (volatility * np.sqrt(maturity))
    d_2 = (first + negative) / (volatility * np.sqrt(maturity))
    price = s_0 * norm.cdf(d_1) - strike * np.exp(
        -risk_free_rate * maturity
    ) * norm.cdf(d_2)
    return price

def delta(s_0,risk_free_rate,volatility,maturity,strike):
    d1 = 1/(volatility*np.sqrt(maturity))*(np.log(s_0/strike)+(risk_free_rate+volatility*volatility/2)*maturity)
    delta = norm.cdf(d1)
    return delta

###########
# Data
##########

# Define inputs
size = int(10e3)
s_0 = np.linspace(0.5,1.5, size)
r = 0.01
volatility = 0.5
maturity = 1.
strike = 1.5
input_train, input_test = pqc.mesh((s_0,), test_size = 0.2)


# Define outputs
output_train = bs_call_price(input_train[:,0],r,volatility,maturity,strike)
output_test = bs_call_price(input_test[:,0],r,volatility,maturity,strike)
output_train = output_train[:,None]
output_test = output_test[:,None]

output_train_grad0 = delta(input_train[:,0],r,volatility,maturity,strike)[:,None]
output_test_grad0 = delta(input_test[:,0],r,volatility,maturity,strike)[:,None]


output_train = np.hstack((output_train,output_train_grad0))
output_test = np.hstack((output_test,output_test_grad0))


# Circuit definition
n_inputs = 1
n_layers = 5
epochs = 5000 

###################################
# Test 1
###################################
# Data
index1 = int(len(input_train)/2)
input_train1 = input_train[:index1,:]
output_train1 = output_train[:index1,:]

# Loss
loss1 = lambda func, weights, x, y: spvsd_gradients_loss(func, weights, x, y ,np.array([0.5,0.5]))

re_uploading_pqc1 = pqc.PQC(n_inputs = n_inputs,n_layers = n_layers)
re_uploading_pqc1.compile(loss = loss1)
metric_train_history1, metric_test_history1 = re_uploading_pqc1.fit(
                            input_train1,output_train1,input_test,output_test,
                            epochs = epochs
                            )


###################################
# Test 2
###################################
# Data
index2 = int(len(input_train))
input_train2 = input_train[:index2,:]
output_train2 = output_train[:index2,:]

# Loss
loss2 = spvsd_loss

re_uploading_pqc2 = pqc.PQC(n_inputs = n_inputs,n_layers = n_layers)
re_uploading_pqc2.compile(loss = loss2)
metric_train_history2, metric_test_history2 = re_uploading_pqc2.fit(
                            input_train2,output_train2,input_test,output_test,
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
plt.savefig('results/images/test7.png')
#plt.show()

df = pd.DataFrame(columns = ["epoch", "mse_value_gradient","mse_value_no_gradient","mse_gradient_gradient","mse_gradient_no_gradient"] )
df["epoch"] = [i for i in range(epochs+1)]
df["mse_value_gradient"] = metric_test_history1[:,0,0]
df["mse_value_no_gradient"] = metric_test_history2[:,0,0]
df["mse_gradient_gradient"] = metric_test_history1[:,0,1]
df["mse_gradient_no_gradient"] = metric_test_history2[:,0,1]
df.to_csv("results/dat/test11.dat", sep= " ", index = False)

# Plot results 
output_test1 = re_uploading_pqc1.predict(input_test)
output_test2 = re_uploading_pqc2.predict(input_test)
fig2, ax2 = plt.subplots(nrows=1, ncols=1)
ax2.scatter(input_test,output_test1[:,0],label = "Gradient")
ax2.scatter(input_test,output_test2[:,0],label = "No Gradient")
ax2.scatter(input_test,output_test[:,0],label = "Exact")
plt.legend()
plt.show()
