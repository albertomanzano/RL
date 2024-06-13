import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
from scipy.special import erf
import time

@jax.jit
def rectangle_rule(x,y):
    dx = jnp.diff(x)
    integral = jnp.dot(y[:-1],dx)
    return integral

@jax.jit
def trapezoidal_rule(x,y):
    dx = jnp.diff(x)
    integral = jnp.dot((y[:-1]+y[1:])/2,dx)
    return integral

def empirical_distribution_function(data_points: np.array):
    N = data_points.shape[0]
    distribution = np.zeros(N)
    for m in range(N):
        count = 0
        for n in list(range(0,m))+list(range(m+1,N)):
            check = np.all(data_points[m]>=data_points[n])
            if check: 
                count = count+1

        distribution[m] = count/(N-1)

    return distribution

def lower_std(x: np.array,axis = 0):
    mean = np.mean(x)
    index = x<mean
    std = np.sqrt(np.mean(np.square(x-mean),axis = axis,where = index))
    return std

def mesh(foo, test_size = 0.0):
    # Define shuffled mesh
    coordinates = jnp.array(jnp.meshgrid(*foo)).T.reshape(-1, len(foo))
    generator = jax.random.PRNGKey(int(time.time()))
    coordinates = jax.random.permutation(generator,coordinates, independent = True)
    # Divide in validation and training
    train_index = int(len(coordinates)*(1.-test_size))
    input_train = coordinates[:train_index]
    input_test = coordinates[train_index:]
    return input_train, input_test

def fwht_sequency(x_input: np.array):
    """Fast Walsh-Hadamard Transform of array x_input in sequence ordering
    The result is not normalised
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3
    algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications
    of Walsh and Related Functions.
    Parameters
    ----------
    x_input : numpy array
    Returns
    ----------
    x_output : numpy array
        Fast Walsh Hadamard transform of array x_input.
    """
    n_ = x_input.size
    n_groups = int(n_ / 2)  # Number of Groups
    m_in_g = 2  # Number of Members in Each Group

    # First stage
    y_ = np.zeros((int(n_ / 2), 2))
    y_[:, 0] = x_input[0::2] + x_input[1::2]
    y_[:, 1] = x_input[0::2] - x_input[1::2]
    x_output = y_.copy()
    # Second and further stage
    for n_stage in range(2, int(np.log2(n_)) + 1):
        y_ = np.zeros((int(n_groups / 2), m_in_g * 2))
        y_[0 : int(n_groups / 2), 0 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] + x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 1 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] - x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 2 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] - x_output[1:n_groups:2, 1:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 3 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] + x_output[1:n_groups:2, 1:m_in_g:2]
        )
        x_output = y_.copy()
        n_groups = int(n_groups / 2)
        m_in_g = m_in_g * 2
    x_output = y_[0, :]
    return x_output



def fwht(x_input: np.array, ordering: str = "sequency"):
    """Fast Walsh Hadamard transform of array x_input
    Works as a wrapper for the different orderings
    of the Walsh-Hadamard transforms.
    Parameters
    ----------
    x_input : numpy array
    ordering: string
        desired ordering of the transform
    Returns
    ----------
    y_output : numpy array
        Fast Walsh Hadamard transform of array x_input
        in the corresponding ordering
    """

    if len(x_input) == 1:
        return x_input
    else:
        y_output = fwht_sequency(x_input)
        return y_output




def left_conditional_probability(initial_bins, probability):
    # Initial domain division
    domain_divisions = 2 ** (initial_bins)
    if domain_divisions >= len(probability):
        raise ValueError(
            "The number of Initial Regions (2**initial_bins)\
        must be lower than len(probability)"
        )
    # Original number of bins of the probability distribution
    nbins = len(probability)
    # Number of Original bins in each one of the bins of Initial
    # domain division
    bins_by_dd = nbins // domain_divisions
    # probability for x located in each one of the bins of Initial
    # domain division
    prob4dd = [
        np.sum(probability[j * bins_by_dd : j * bins_by_dd + bins_by_dd])
        for j in range(domain_divisions)
    ]
    # Each bin of Initial domain division is splatted in 2 equal parts
    bins4_left_dd = nbins // (2 ** (initial_bins + 1))
    # probability for x located in the left bin of the new splits
    left_probabilities = [
        np.sum(probability[j * bins_by_dd : j * bins_by_dd + bins4_left_dd])
        for j in range(domain_divisions)
    ]
    # Conditional probability of x located in the left bin when x is located
    # in the bin of the initial domain division that contains the split
    # Basically this is the f(j) function of the article with
    # j=0,1,2,...2^(i-1)-1 and i the number of qubits of the initial
    # domain division
    with np.errstate(divide="ignore", invalid="ignore"):
        left_cond_prob = np.array(left_probabilities) / np.array(prob4dd)
    left_cond_prob[np.isnan(left_cond_prob)] = 0
    return left_cond_prob

#############################################
# PQC expansion transform
#############################################

def pqc_observable_diagonalization(vector):
    
    n = int(np.log2(len(vector)))
    b = -float(np.real(vector[0]))
    c = -np.abs(np.vdot(vector[1:],vector[1:]))
    autovalue1 = (-b+np.sqrt(b*b-4*c))/2.
    autovalue2 = (-b-np.sqrt(b*b-4*c))/2.
   
    a00 = np.sqrt(1/(1-c/(autovalue1*autovalue1)))
    a01 = np.sqrt(1/(1-c/(autovalue2*autovalue2)))

    autovector1 = np.ones(len(vector),dtype = np.complex128)*a00
    autovector2 = np.ones(len(vector),dtype = np.complex128)*a01

    autovector1[1:] = autovector1[0]*vector[1:]/autovalue1
    autovector2[1:] = autovector2[0]*vector[1:]/autovalue2


    return autovalue1, autovalue2, autovector1, autovector2



def expansion_to_pqc(expansion):
        expansion = np.conj(expansion)
        n = int(np.log2(len(expansion)))

        # Diagonalization of the observable
        autovalue1, autovalue2, autovector1, autovector2 = pqc_observable_diagonalization(expansion)

        # Transformation to compute the angles
        theta = np.arctan(np.real(autovector2[0]/autovector1[0]))
        column1 = autovector1*np.cos(theta)+autovector2*np.sin(theta)
        column2 = -autovector1*np.sin(theta)+autovector2*np.cos(theta)
        column2[0] = 0

        # Map to probabilities and phases
        probability = np.square(np.abs(column2))
        phase = np.angle(column2)

        # Initialize probabilities
        size = int(2**n+2**(n-1)-2)
        probability_angles = np.zeros(size)
        
        # Initialize phases
        phase_angles = np.zeros_like(phase)
        phase_angles[0] = -2*np.mean(phase)

        for i in range(n-1):
            # Probabilities
            position = int(2**(i+1)-2)
            conditional_probability = left_conditional_probability(i, probability)
            array = np.concatenate((np.ones_like(conditional_probability),np.sqrt(conditional_probability)))
            probability_angles[position:position+2**(i+1)] = fwht(2.0 * (np.arccos(array)), ordering = "sequency")/ 2**(i+1)
            
            # Phases
            split = np.array(np.split(phase,2**(i+1)))
            suma = np.sum(split,axis = 1)
            left = suma[0::2] 
            right = suma[1::2] 
            shifts = (right-left)/2**(n-i)
            phase_angles[2**i:2**(i+1)] = fwht(2.0 * shifts, ordering = "sequency")/ 2**i

        # Last position (sin!)
        position = int(2**n-2)
        conditional_probability = left_conditional_probability(n-1, probability)
        probability_angles[position:] = fwht(2.0 * (-np.arcsin(np.sqrt(conditional_probability))), ordering = "sequency")/ 2**(n-1)
        
        # Phases
        split = np.split(phase,2**n)
        suma = np.sum(split,axis = 1)
        left = suma[0::2] 
        right = suma[1::2] 
        shifts = (right-left)/2
        phase_angles[2**(n-1):2**n] = fwht(2.0 * shifts, ordering = "sequency")/ 2**(n-1)

        
        angles = np.concatenate(([-2*theta],probability_angles,phase_angles))
        return angles, autovalue1, autovalue2
        
#################################
# Circuits
#################################
def multiplexor_ry_t(angles,n,control_register,target_register):
        
    control = [i for i in range(2**n) ]
    for i in range(n):
        for j in range(2**i - 1, 2**n, 2**i):
            control[j] = n - i - 1

    for i in range(2**n-1,-1,-1):
        qml.CNOT(wires = [control_register[control[i]],target_register])
        qml.RY(-angles[i], wires = target_register)

def multiplexor_rz_t(angles,n,control_register,target_register):
        
    control = [ i for i in range(2**n) ]
    for i in range(n):
        for j in range(2**i - 1, 2**n, 2**i):
            control[j] = n - i - 1

    for i in range(2**n-1,-1,-1):
        qml.CNOT(wires = [control_register[control[i]],target_register])
        qml.RZ(-angles[i], wires = target_register)


##################################
# Black-Scholes
##################################
def bs_pdf(s_t: float,s_0: float = 1.0,risk_free_rate: float = 0.0,volatility: float = 0.5,maturity: float = 0.5):

    mean = (risk_free_rate - 0.5 * volatility * volatility) * maturity + np.log(s_0)
    factor = s_t * volatility * np.sqrt(2 * np.pi * maturity)
    exponent = -((np.log(s_t) - mean) ** 2) / (2 * volatility * volatility * maturity)
    density = np.exp(exponent) / factor
    return density

def bs_cdf(s_t: float,s_0: float = 1.0,risk_free_rate: float = 0.0,volatility: float = 0.5,maturity: float = 0.5):

    mean = (risk_free_rate - 0.5 * volatility * volatility) * maturity + np.log(s_0)
    variance = volatility * volatility * maturity

    return 0.5 * (1 + erf((np.log(s_t) - mean) / (np.sqrt(2 * variance))))

def bs_samples(number_samples: int,s_0: float = 1.0,risk_free_rate: float = 0.0,volatility: float = 0.5,maturity: float = 0.5):

    dW = np.random.randn(number_samples)
    s_t = s_0 * np.exp((risk_free_rate - 0.5 * volatility * volatility) * maturity + volatility * dW * np.sqrt(maturity))

    return s_t


