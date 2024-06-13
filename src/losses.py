from jax import numpy as np
import jax

from src.metrics import MSE_metric, MAE_metric, MAX_metric
from src.utils import trapezoidal_rule


def spvsd_gradients_loss(func,weights,x,y, loss_weights):
    """Cost function to be minimized.

    Args:
        x (array[float]): array of input vectors
        y (array[float]): array of targets

    Returns:
        float: loss value to be minimized
    """
    # Compute prediction for each input in data batch
    y_pred = func(weights,x)
    loss_i = MSE_metric(y,y_pred)
    return np.dot(loss_weights,loss_i)

def spvsd_loss(func,weights,x,y):
    """Cost function to be minimized.

    Args:
        x (array[float]): array of input vectors
        y (array[float]): array of targets

    Returns:
        float: loss value to be minimized
    """
    # Compute prediction for each input in data batch
    y_pred = func(weights,x)
    loss = MSE_metric(y,y_pred)
    return loss[0]

def max_loss(func,weights,x,y):
    """Cost function to be minimized.

    Args:
        x (array[float]): array of input vectors
        y (array[float]): array of targets

    Returns:
        float: loss value to be minimized
    """
    # Compute prediction for each input in data batch
    y_pred = func(weights,x)
    loss = MAX_metric(y,y_pred)
    return loss[0]

def sample_density_loss(func,weights,x,y,key,shape,minval,maxval):
    """Cost function to be minimized.

    Args:
        x (array[float]): array of input vectors
        y (array[float]): array of targets

    Returns:
        float: loss value to be minimized
    """

    # Compute part from samples
    y_pred_random = func(weights,x)
    mean = -2*np.mean(y_pred_random[:,1])

    # Compute integral
    uniform = jax.random.uniform(key = key,shape = shape,minval = minval,maxval = maxval)
    f_integral = func(weights,uniform)
    area = np.prod(maxval-minval)
    integral = area*np.mean(f_integral[:,1]*f_integral[:,1])

    return mean+integral

def SLC_loss(func,weights,x,y,key,shape,minval,maxval):
    """Cost function to be minimized.

    Args:
        x (array[float]): array of input vectors
        y (array[float]): array of targets

    Returns:
        float: loss value to be minimized
    """

    # Compute part from samples
    y_pred_random = func(weights,x)
    uniform = jax.random.uniform(key,shape = (np.size(x),1))
    loss = MSE_metric(y_pred_random[:,0],uniform[:,0])
    
    return loss

def SLC_gradient_loss(func,weights,x,y,loss_weights,key,shape,minval,maxval):
    """Cost function to be minimized.

    Args:
        x (array[float]): array of input vectors
        y (array[float]): array of targets

    Returns:
        float: loss value to be minimized
    """
    size = np.size(x)

    # Compute part from samples
    y_pred_random = func(weights,x)
    uniform = jax.random.uniform(key,shape = (size,1))
    loss_1 = MSE_metric(y_pred_random[:,0],uniform[:,0])
    
    # Compute part from samples
    mean = -2*np.mean(y_pred_random[:,1])

    # Compute integral
    uniform = jax.random.uniform(key = key,shape = shape,minval = minval,maxval = maxval)
    f_integral = func(weights,uniform)
    area = np.prod(maxval-minval)
    integral = area*np.mean(f_integral[:,1]*f_integral[:,1])
    loss_2 = integral+mean

    #
    loss = loss_1*loss_weights[0]+loss_2*loss_weights[1]
    
    return loss


def SIC_gradient_loss(func,weights,x,y,loss_weights,key,shape,minval,maxval):
    """Cost function to be minimized.

    Args:
        x (array[float]): array of input vectors
        y (array[float]): array of targets

    Returns:
        float: loss value to be minimized
    """
    size = np.size(x)

    # Compute SIC
    y_pred_random = func(weights,x)
    loss_1 = MSE_metric(y_pred_random[:,0],y[:,0])
    
    # Compute density loss
    # Compute part from samples
    mean = -2*np.mean(y_pred_random[:,1])

    # Compute integral
    #uniform = jax.random.uniform(key = key,shape = shape,minval = minval,maxval = maxval)
    #f_integral = func(weights,uniform)
    #area = np.prod(maxval-minval)
    #integral = area*np.mean(f_integral[:,1]*f_integral[:,1])
    x_integral = np.linspace(minval,maxval,shape[0])
    y_integral = func(weights,x_integral)
    integral = trapezoidal_rule(x_integral[:,0],y_integral[:,1] * y_integral[:,1]) 

    # Compute second loss
    loss_2 = integral+mean

    #
    loss = loss_1*loss_weights[0]+loss_2*loss_weights[1]
    
    return loss
