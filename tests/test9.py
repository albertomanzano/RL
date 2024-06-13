######################################
# Data scarcity for BS
######################################
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
size = int(2*10e3)
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
n_layers = 2
epochs = 2000

# Data scarcity
training_size = [i for i in range(20,1010,20) ]
error1 = []
error2 = []

for i in range(len(training_size)):
    ###################################
    # Test 1
    ###################################
    # Data
    index1 = int(training_size[i]/2)
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

    error1.append(metric_test_history1[-1,0,0])


    ###################################
    # Test 2
    ###################################
    # Data
    index2 = int(training_size[i])
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
    
    error2.append(metric_test_history2[-1,0,0])

# Save into txt
df = pd.DataFrame(columns=["n_points","error1","error2"])
df["n_points"] = training_size
df["error1"] = error1
df["error2"] = error2
print(df)
df.to_csv("results/dat/test9.dat",sep = " ",index = False)

plt.plot(training_size,error1,label = "Gradients")
plt.plot(training_size,error2,label = "No Gradients")
plt.legend()
plt.xlabel("Number of points")
plt.ylabel("MSE")
plt.grid()
plt.savefig("results/images/test9.png")
plt.show()
