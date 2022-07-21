import sys
sys.path.append("/home/alberto/RL/RL")
import matplotlib
matplotlib.use('TkAgg')

from agents.re_uploading_pqc import ReUploadingPQC
from scipy.stats import qmc
from scipy.special import ndtr
import numpy as np
import tensorflow_quantum as tfq
import tensorflow as tf
import matplotlib.pyplot as plt

def delta(tau,K,S,sigma,r):
    d1 = 1/(sigma*np.sqrt(tau))*(np.log(S/K)+(r+sigma*sigma/2)*tau)
    delta = ndtr(d1)
    return delta

# Define market space
strike = 1
r = 0.05
volatility = 0.5

# Load dataset
dataset = tf.data.experimental.load("dataset2")
x = []
y = []
for elements in dataset:
    x.append(elements[0])
    y.append(elements[1])
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

# Shuffle indices
indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)

shuffled_x = tf.gather(x, shuffled_indices)
shuffled_y = tf.gather(y, shuffled_indices)

# Train and test split
size = shuffled_x.shape[0]
train_size = int(0.8*size)
x_train = shuffled_x[0:train_size]
y_train = shuffled_y[0:train_size]
x_test = shuffled_x[train_size:]
y_test = shuffled_y[train_size:]

print("Data Loaded")

# Define the associated circuit
input_space = 2
output_space = 1
n_layers = 6
model = ReUploadingPQC(input_space,output_space,n_layers)
model.print_circuit()
model.fit(x_train,y_train,x_test,y_test,epochs = 60)

# Predict values 
n = 10
tau = np.linspace(1/252,1,n)
s = np.linspace(0.1,3,n)

# Compute exact
inputs = np.array(np.meshgrid(tau,s)).T.reshape(-1,2)
prediction = model(inputs)
exact = np.array(delta(inputs[:,0],strike,inputs[:,1],volatility,r))
prediction = prediction.numpy().reshape(len(tau),len(s))
exact = exact.reshape(len(tau),len(s))
absolute_error = np.abs(prediction-exact)


# Line plots
minimum = min([np.min(exact),np.min(prediction),np.min(absolute_error)])
maximum = min([np.max(exact),np.max(prediction),np.max(absolute_error)])
extent = [np.min(tau),np.max(tau),np.min(s),np.max(s)]
fig, ax = plt.subplots(1,3)


ax[0].set_title('Prediction')
im0 = ax[0].imshow(prediction,cmap = 'plasma',vmin = minimum,vmax = maximum,extent = extent)
ax[0].set_xlabel("tau")
ax[0].set_ylabel("s")
ax[1].set_title('Exact')
im1 = ax[1].imshow(exact,cmap = 'plasma',vmin = minimum, vmax = maximum,extent = extent)
ax[1].set_xlabel("tau")
ax[1].set_ylabel("s")
ax[2].set_title('Absolute Error')
im2 = ax[2].imshow(absolute_error,cmap = 'plasma',vmin = minimum, vmax = maximum, extent = extent)
ax[2].set_xlabel("tau")
ax[2].set_ylabel("s")

fig.subplots_adjust(right=0.8)


plt.colorbar(im2)
plt.show()
