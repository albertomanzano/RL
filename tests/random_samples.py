########################################################
# Test of loss function with random samples from a distribution
########################################################

import sys
sys.path.append("./")
sys.path.append("../")
import time

from jax import numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ncx2, norm
import src.pqc as pqc
from src.arquitectures import HardwareEfficientAnsatz, ConstantArquitecture
from src.losses import spvsd_gradients_loss, spvsd_loss, max_loss, sample_density_loss, SLC_loss, SLC_gradient_loss, SIC_gradient_loss
from src.metrics import MAE_metric, MSE_metric, one_metric
from src.utils import empirical_distribution_function, bs_samples, bs_pdf, bs_cdf




def test_dml(N_points_train: int, N_points_test: int, epochs: int, weight_spvsd: float, weight_differential: float, distribution: str):
    if (distribution == "normal"):
        a=-np.pi
        b=np.pi
        
        x_train = np.random.randn(N_points_train) 
        x_test = np.linspace(a,b,N_points_test) 
        
        y_train = empirical_distribution_function(np.reshape(x_train,(N_points_train,1))) - 0.5
        y_train_ = norm.pdf(x_train)
        
        y_test = norm.cdf(x_test) - 0.5
        y_test_ = norm.pdf(x_test)
    
    # Define market space
    if (distribution == "bs"):
        a = 0.1
        b = 3.0
        
        x_train = bs_samples(N_points_train) 
        x_test = np.linspace(a,b,N_points_test) 
        
        y_train = empirical_distribution_function(np.reshape(x_train,(N_points_train,1))) - 0.5
        y_train_ = bs_pdf(x_train)
        
        y_test = bs_cdf(x_test) - 0.5
        y_test_ = bs_pdf(x_test)
    
    # Produce shape
    x_train = jnp.array(x_train.reshape((N_points_train,1)))
    y_train = np.concatenate((y_train[:,None],y_train_[:,None]),axis = 1)
    y_train = jnp.array(y_train.reshape((N_points_train,2)))
    
    x_test = jnp.array(x_test.reshape((N_points_test,1)))
    y_test = np.concatenate((y_test[:,None],y_test_[:,None]),axis = 1)
    y_test = jnp.array(y_test.reshape((N_points_test,2)))
    
    
    # Test 1
    n_inputs = 1
    n_qubits = 2
    n_layers = 3
    base_frequency = np.pi/(b-a)
    level = 1.0
    
    
    arquitecture = HardwareEfficientAnsatz(n_inputs = n_inputs,n_qubits = n_qubits, n_layers = n_layers,base_frequency = base_frequency,level = level)
    re_uploading_pqc = pqc.PQC(arquitecture = arquitecture)
    
    # Compilation
    metrics = {"MAE": MAE_metric}
    key = jax.random.PRNGKey(0)
    shape = (N_points_test,1)
    minval = jnp.array([a])
    maxval = jnp.array([b])
    
    loss_weights = jnp.array([weight_spvsd, weight_differential])
    loss = lambda func,weights,x,y: SIC_gradient_loss(func,weights,x,y,loss_weights,key,shape,minval,maxval)
    re_uploading_pqc.compile(loss = loss,metrics = metrics)
    
    
    # Train history
    metric_train_history, metric_test_history = re_uploading_pqc.fit(
                                x_train,y_train,x_test,y_test,
                                epochs = epochs,
                                )
    
    
    y_pred = re_uploading_pqc.predict(x_test)

    return x_test, y_test, y_pred, metric_train_history, metric_test_history


def save(x_test, y_test, y_pred, y_pred_, mae_train_history, mae_train_history_, mae_test_history, mae_test_history_, loss, distribution):
    y_pred = np.array(y_pred)
    y_pred_ = np.array(y_pred_)

    mae_test_history = np.array(mae_test_history)
    mae_test_history_ = np.array(mae_test_history_)
    mae_train_history = np.array(mae_train_history)
    mae_train_history_ = np.array(mae_train_history_)
    
    # Test
    mae_test_mean = np.percentile(mae_test_history,50,axis = 0)
    mae_test_upper = np.percentile(mae_test_history,75,axis = 0)
    mae_test_lower = np.percentile(mae_test_history,25,axis = 0)
    
    save = np.concatenate((np.arange(epochs+1)[:,None],mae_test_mean[:,None],mae_test_upper[:,None],mae_test_lower[:,None]),axis = 1)
    np.savetxt("dat/test_{0}_{1}.dat".format(loss,distribution),save,delimiter = ' ',header = "x mean upper lower",comments='')
    
    # Test
    mae_test_mean_ = np.percentile(mae_test_history_,50,axis = 0)
    mae_test_upper_ = np.percentile(mae_test_history_,75,axis = 0)
    mae_test_lower_ = np.percentile(mae_test_history_,25,axis = 0)
    
    save = np.concatenate((np.arange(epochs+1)[:,None],mae_test_mean_[:,None],mae_test_upper_[:,None],mae_test_lower_[:,None]),axis = 1)
    np.savetxt("dat/test_{0}_{1}_.dat".format(loss,distribution),save,delimiter = ' ',header = "x mean upper lower",comments='')
    
    # Train
    mae_train_mean = np.percentile(mae_train_history,50,axis = 0)
    mae_train_upper = np.percentile(mae_train_history,75,axis = 0)
    mae_train_lower = np.percentile(mae_train_history,25,axis = 0)
    
    save = np.concatenate((np.arange(epochs+1)[:,None],mae_train_mean[:,None],mae_train_upper[:,None],mae_train_lower[:,None]),axis = 1)
    np.savetxt("dat/train_{0}_{1}.dat".format(loss,distribution),save,delimiter = ' ',header = "x mean upper lower",comments='')
    
    # Train
    mae_train_mean_ = np.percentile(mae_train_history_,50,axis = 0)
    mae_train_upper_ = np.percentile(mae_train_history_,75,axis = 0)
    mae_train_lower_ = np.percentile(mae_train_history_,25,axis = 0)
    
    save = np.concatenate((np.arange(epochs+1)[:,None],mae_train_mean_[:,None],mae_train_upper_[:,None],mae_train_lower_[:,None]),axis = 1)
    np.savetxt("dat/train_{0}_{1}_.dat".format(loss,distribution),save,delimiter = ' ',header = "x mean upper lower",comments='')
    
    # Evaluate  f
    y_pred_mean = np.percentile(y_pred,50,axis = 0)[:,None]
    y_pred_upper = np.percentile(y_pred,75,axis = 0)[:,None]
    y_pred_lower = np.percentile(y_pred,25,axis = 0)[:,None]
    
    save = np.concatenate((x_test,y_test[:,0][:,None],y_pred_mean,y_pred_upper,y_pred_lower),axis = 1)
    np.savetxt("dat/f_{0}_{1}.dat".format(loss,distribution),save,delimiter = ' ',header = "x y y_pred y_pred_upper y_pred_lower",comments='')
    
    # Evaluate  f_
    y_pred_mean_ = np.percentile(y_pred_,50,axis = 0)[:,None]
    y_pred_upper_ = np.percentile(y_pred_,75,axis = 0)[:,None]
    y_pred_lower_ = np.percentile(y_pred_,25,axis = 0)[:,None]
    
    save = np.concatenate((x_test,y_test[:,1][:,None],y_pred_mean_,y_pred_upper_,y_pred_lower_),axis = 1)
    np.savetxt("dat/f_{0}_{1}_.dat".format(loss,distribution),save,delimiter = ' ',header = "x y y_pred y_pred_upper y_pred_lower",comments='')

    return x_test, y_pred_mean, y_pred_upper, y_pred_lower, y_pred_mean_, y_pred_upper_, y_pred_lower_


iterations = 100
epochs = 100
N_points_train = 10
N_points_test = 100
distribution = "normal"

########################################################
# Experiment gradient
########################################################
y_pred = []
y_pred_ = []
mae_test_history = []
mae_test_history_ = []
mae_train_history = []
mae_train_history_ = []

weight_spvsd = 1.0
weight_differential = 5.0
loss = "{}_{}".format(int(weight_spvsd), int(weight_differential))
for i in range(iterations):
    x_test, y_test, y_prediction, metric_train_history, metric_test_history = test_dml(N_points_train, N_points_test, epochs, weight_spvsd, weight_differential, distribution)
    
    y_pred.append(y_prediction[:,0])
    y_pred_.append(y_prediction[:,1])
    
    mae_test_history.append(metric_test_history["MAE"][:,0])
    mae_test_history_.append(metric_test_history["MAE"][:,1])
    mae_train_history.append(metric_train_history["MAE"][:,0])
    mae_train_history_.append(metric_train_history["MAE"][:,1])

x_test, y_pred_mean, y_pred_upper, y_pred_lower, y_pred_mean_, y_pred_upper_, y_pred_lower_ = \
        save(x_test, y_test, y_pred, y_pred_, mae_train_history, mae_train_history_, mae_test_history, mae_test_history_, loss, distribution)


########################################################
# Gradients
########################################################
#x_test = x_test.flatten()
#
#plt.plot(x_test,y_test[:,0],c = "g",label = "Target")
#plt.plot(x_test,y_pred_mean.flatten(),c = "b",label = "Differential Loss")
#plt.fill_between(x_test, y_pred_lower.flatten(), y_pred_upper.flatten(), alpha=0.2)
#plt.title("CDF")
#plt.legend()
#plt.show()
#
#plt.plot(x_test,y_test[:,1],c = "g",label = "Target_")
#plt.plot(x_test,y_pred_mean_.flatten(),c = "b",label = "Differential Loss")
#plt.fill_between(x_test, y_pred_lower_.flatten(), y_pred_upper_.flatten(), alpha=0.2)
#plt.title("PDF")
#plt.legend()
#plt.show()
