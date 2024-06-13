import time
import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
from src.utils import expansion_to_pqc, multiplexor_rz_t, multiplexor_ry_t

class HardwareEfficientAnsatz:

    def __init__(self,n_inputs: int = 1,n_qubits: int = 1,n_layers: int = 1,base_frequency: float = 1.,level: float = 1.):
        self.n_inputs = n_inputs
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.base_frequency = base_frequency
        self.level = level
        self.weights = None

    def init_weights(self,generator = None):
        if generator is None:
            generator = jax.random.PRNGKey(int(time.time()))
            self.weights = jax.random.uniform(generator,(self.n_inputs,self.n_qubits,self.n_layers))

        #elif (type(generator) == type(jax.random.PRNGKey(0))):
        #    self.weights = jax.random.uniform(generator,(self.n_inputs,self.n_qubits,self.n_layers))

        else:
            self.weights = generator

    def __call__(self,weights,x):
        """A variational quantum circuit representing the Universal classifier.

        Args:
            params (array[float]): array of parameters
            x (array[float]): single input vector

        Returns:
            float: fidelity between output state and input
        """

        for l in range(self.n_layers):
            
            for i in range(self.n_inputs):

                # Variational layer
                for j in range(self.n_qubits):
                    qml.RX(weights[i,j,l],wires=i*self.n_qubits+j)
                
                # Encoding layer
                for j in range(self.n_qubits):
                    qml.RY(self.base_frequency*x[i],wires=i*self.n_qubits+j)
                    
            # Entangling layer
            for i in range(self.n_qubits*self.n_inputs-1):
                    qml.CNOT(wires = [i,i+1])
            qml.CNOT(wires = [self.n_qubits*self.n_inputs-1,0])

        # Operator
        if (self.n_inputs == 1) and (self.n_qubits == 1):
            op = qml.PauliZ(wires = 0)
        else:
            op = qml.prod(*[ qml.PauliZ(wires = i) for i in range(self.n_qubits*self.n_inputs) ])
        return qml.expval(self.level*op)

class ConstantPureSin:

    def __init__(self,n_inputs: int = 1,n_layers: int = 1,base_frequency: float = 1.,level: float = 1.):
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.base_frequency = base_frequency
        self.level = level
        self.weights = None

    def init_weights(self,generator = None):
        if generator is None:
            generator = jax.random.PRNGKey(int(time.time()))
        self.weights = jax.random.uniform(generator,(self.n_inputs,self.n_layers))

    def __call__(self,weights,x):
        """A variational quantum circuit representing the Universal classifier.

        Args:
            params (array[float]): array of parameters
            x (array[float]): single input vector

        Returns:
            float: fidelity between output state and input
        """

        for l in range(self.n_layers):
            
            # Variational layer
            for i in range(self.n_inputs):
                qml.RX(weights[i,l],wires=i)
            
            # Encoding layer
            for i in range(self.n_inputs):
                qml.RY(self.base_frequency*x[i]+3*np.pi/2,wires=i)
                
            # Entangling layer
            for i in range(self.n_inputs-1):
                qml.SWAP(wires = [i,i+1])

        # Operator
        if self.n_inputs == 1:
            op = qml.PauliZ(wires = 0)
        else:
            op = qml.prod(*[ qml.PauliZ(wires = i) for i in range(self.n_inputs) ])
        return qml.expval(self.level*op)

class ConstantGeneral:

    def __init__(self,n_inputs: int = 1,n_layers: int = 1,base_frequency: float = 1.,level: float = 1.):
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.base_frequency = base_frequency
        self.level = level
        self.weights = None

    def init_weights(self,generator = None):
        if generator is None:
            generator = jax.random.PRNGKey(int(time.time()))
        self.weights = jax.random.uniform(generator,(self.n_inputs,2,self.n_layers))

    def __call__(self,weights,x):
        """A variational quantum circuit representing the Universal classifier.

        Args:
            params (array[float]): array of parameters
            x (array[float]): single input vector

        Returns:
            float: fidelity between output state and input
        """

        for l in range(self.n_layers):
            
            # Variational layer
            for i in range(self.n_inputs):
                qml.RY(weights[i,0,l],wires=i)
            
            # Encoding layer
            for i in range(self.n_inputs):
                qml.RX(weights[i,1,l],wires=i)
                qml.RX(self.base_frequency*x[i],wires=i)
                
            # Entangling layer
            for i in range(self.n_inputs-1):
                qml.SWAP(wires = [i,i+1])

        # Operator
        if self.n_inputs == 1:
            op = qml.PauliZ(wires = 0)
        else:
            op = qml.prod(*[ qml.PauliZ(wires = i) for i in range(self.n_inputs) ])
        return qml.expval(self.level*op)

class ConstantArquitecture:

    def __init__(self,n_qubits: int = 1,n_inputs: int = 1, n_layers: int = 1,base_frequency: float = 1.,level: float = 1.):
        self.n_qubits = n_qubits
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.base_frequency = base_frequency
        self.level = level
        self.weights = None

    def init_weights(self,generator = None):
        if generator is None:
            generator = jax.random.PRNGKey(int(time.time()))
        self.weights = jax.random.uniform(generator,(self.n_inputs,3,self.n_layers))
        #self.weights = jax.random.uniform(generator,(self.n_inputs,self.n_layers))

    def __call__(self,weights,x):
        """A variational quantum circuit representing the Universal classifier.

        Args:
            params (array[float]): array of parameters
            x (array[float]): single input vector

        Returns:
            float: fidelity between output state and input
        """

        for l in range(self.n_layers):
            
            # Variational layer
            for i in range(self.n_inputs):
                qml.Hadamard(wires = i)
                #qml.RY(weights[i,l],wires=i)
                qml.RZ(weights[i,0,l],wires=i)
                qml.RY(weights[i,1,l],wires=i)
                qml.RZ(weights[i,2,l],wires=i)
            
            # Encoding layer
            for i in range(self.n_inputs):
                qml.Hadamard(wires = i)
                qml.RX(self.base_frequency*x[i],wires=i)
                
            # Entangling layer
            for i in range(self.n_inputs-1):
                qml.SWAP(wires = [i,i+1])

        # Operator
        #op = qml.prod(*[ qml.PauliY(wires = i) for i in range(self.n_inputs) ])
        op = qml.PauliY(wires = 0)
        return qml.expval(self.level*op)

class FourierAnsatz:
    def __init__(self,n_qubits: int = 1,n_inputs: int = 1,n_layers: int = 1, base_frequency = 1., simulator: str = "default.qubit.jax",level: float = 1.):
        # Structure of the network
        self.n_qubits = n_qubits
        self.n_inputs = n_inputs
        self.base_frequency = base_frequency
        self.level = level
        self.weights = None
        self.ob = None
        
        
    def init_weights(self,generator = None):
        N = 2**self.n_qubits

        if generator is None:
            size1 = int(2**self.n_qubits+2**(self.n_qubits-1)-2)
            size2 = int(2**self.n_qubits)
            size = size1+size2+1
            generator = jax.random.PRNGKey(int(time.time()))
            self.weights = jax.random.uniform(generator,(size,))
            matrix = np.zeros((N,N))
            matrix[0,0] = 1.*2**(self.n_qubits-1)
            matrix[1,1] = -1.*2**(self.n_qubits-1)
            self.ob = qml.Hermitian(matrix, wires = [i for i in range(self.n_qubits)])
        else:
            angles, autovalue1, autovalue2 = expansion_to_pqc(generator)
            hermitian = np.zeros((N,N))
            hermitian[0,0] = autovalue1*2**(self.n_qubits-1)
            hermitian[1,1] = autovalue2*2**(self.n_qubits-1)
            self.ob = qml.Hermitian(hermitian, wires = [i for i in range(self.n_qubits)])
            self.weights = jnp.array(angles)


    def __call__(self,angles,x):
            n = self.n_qubits
            theta = angles[0]
            angles_amplitude = angles[1:-2**n]
            angles_phase = angles[-2**n:]

            for q in range(n):
                qml.Hadamard(wires = q)
        
            for q in range(n):
                qml.RZ((2**(n-1-q))*x[0],wires = q)

            control_register = [j for j in range(n-2,-1,-1)]
            target_register = n-1
            position = int(2**n-2)
            multiplexor_rz_t(angles_phase[2**(n-1):2**n],n-1,control_register = control_register,target_register = n-1)
            multiplexor_ry_t(angles_amplitude[position:position+2**(n-1)],n-1,control_register = control_register,target_register = target_register)
            ##########################################################
            ##########################################################
            for i in range(n-2,-1,-1):
                if i == 0:
                    qml.RZ(-angles_phase[1], wires = 0)
                else:
                    control_register = [j for j in range(i-1,-1,-1)]
                    multiplexor_rz_t(angles_phase[2**i:2**(i+1)],i,control_register = control_register,target_register = i)
                
                control_register = [j for j in range(i-1,-1,-1)]
                control_register = control_register+[n-1]
                target_register = i
                position = int(2**(i+1)-2)
                
                multiplexor_ry_t(angles_amplitude[position:position+2**(i+1)],i+1,control_register = control_register,target_register = target_register)
                

            ##########################################################
            qml.RZ(-angles_phase[0],wires = 0)
            qml.ctrl(qml.RY, [i for i in range(n-1)], control_values = [0 for i in range(n-1)])(-theta, wires=n-1)
            return qml.expval(self.level*self.ob)
    



