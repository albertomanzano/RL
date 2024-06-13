import jax.numpy as np
import jax

@jax.jit
def MSE_metric(y, y_pred): 
    return np.sum(np.power(y-y_pred,2.0), axis = 0)/y.shape[0]

@jax.jit
def MAE_metric(y, y_pred): 
    return np.sum(np.abs(y-y_pred), axis = 0)/y.shape[0]

@jax.jit
def MAX_metric(y, y_pred): 
    return np.max(np.abs(y-y_pred), axis = 0)

@jax.jit
def one_metric(y, y_pred): 
    return 1.
