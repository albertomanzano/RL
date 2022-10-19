import pennylane as qml
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import optax
from agents.losses import spvsd_loss
from agents.metrics import MSE_metric
import time
import copy

def mesh(foo, test_size = 0.0):
    # Define shuffled mesh
    coordinates = np.array(np.meshgrid(*foo)).T.reshape(-1, len(foo))
    generator = jax.random.PRNGKey(int(time.time()))
    coordinates = jax.random.permutation(generator,coordinates, independent = True)
    # Divide in validation and training
    train_index = int(len(coordinates)*(1.-test_size))
    input_train = coordinates[:train_index]
    input_test = coordinates[train_index:]
    return input_train, input_test


class PQC:

    def __init__(self,n_inputs: int = 1,n_layers: int = 1, base_frequency = 1., simulator: str = "default.qubit.jax", arquitecture = "constant"):
        # Structure of the network
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.base_frequency = base_frequency
        
        # Arquitecture
        if arquitecture=="linear_arquitecture":
            print("Linear arquitecture")
            self.arquitecture = self.linear_arquitecture
        else:
            self.arquitecture = self.constant_arquitecture
        
        # Main functions
        self.dev = qml.device(simulator, wires=self.n_inputs)
        self.circuit = qml.QNode(self.arquitecture,self.dev,interface = "jax")
        self.circuit_plot = copy.deepcopy(self.circuit)
        self.gradient_circuit = jax.grad(self.circuit,argnums = 1)
        self.circuit = jax.jit(self.circuit)
        self.gradient_circuit = jax.jit(self.gradient_circuit)

        # Batched functions
        self.call_map = jax.vmap(self.call,in_axes = (None,0), out_axes = 0)

        # Compiling time
        self.loss = None
        self.metrics = None
        self.optimizer = None
        self.opt_state = None
        self.weights = None
        self.metric_train_history = None
        self.metric_test_history = None


    def constant_arquitecture(self,weights,x):
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
                qml.RY(weights[i,l],wires=i)
                #qml.RZ(weights[i,2,l],wires=i)
            
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
        return qml.expval(op)
    
    def linear_arquitecture(self,weights,x):
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
                qml.RY(weights[i,l],wires=i)
            
            # Encoding layer
            for i in range(self.n_inputs):
                qml.Hadamard(wires = i)
                qml.RX(self.base_frequency*(l+1)*x[i],wires=i)
                
            # Entangling layer
            for i in range(self.n_inputs-1):
                qml.SWAP(wires = [i,i+1])

        # Operator
        #op = qml.prod(*[ qml.PauliY(wires = i) for i in range(self.n_inputs) ])
        op = qml.PauliY(wires = 0)
        return qml.expval(op)

    def call(self,weights,x):
        ''' Processes a single input'''
        y = self.circuit(weights,x)
        y_grad = self.gradient_circuit(weights,x)
        solution = np.hstack([y,y_grad])
        return solution
    

    def compile(self,optimizer = None, loss = None, metrics = None,mask = None):
        
        # Set weights
        generator = jax.random.PRNGKey(int(time.time()))
        #self.weights = jax.random.uniform(generator,(self.n_inputs,3,self.n_layers))
        self.weights = jax.random.uniform(generator,(self.n_inputs,self.n_layers))
        
        # Optimizer
        if optimizer is None:
            optimizer = optax.adam(0.1)
        self.optimizer = optimizer
        self.opt_state = optimizer.init(self.weights)

        # Metrics
        if metrics is None:
            metrics = {"MSE": MSE_metric}
        self.metrics = metrics

        if loss is None:
            loss = losses.spvsd_loss
        self.loss = loss

        
        # History
        self.metric_train_history = []
        self.metric_test_history = []
        
        return None

    

    def compute_metrics(self,x,y,lista):
        y_pred = self.call_map(self.weights,x)

        title = [ '{0: <25}'.format(str(i)) for i in range(y.shape[1]) ]
        underline = [ '{0: <25}'.format("-"*25) for i in range(y.shape[1]) ]
        title = ''.join(title)
        underline = ''.join(underline)
        print('{0: <25}'.format(" ")+title)
        print('{0: <25}'.format(" ")+underline)

        epoch_list = []
        for name, metric in self.metrics.items():
            metric_list = []
            metric_string = []
            for j in range(y_pred.shape[1]):
                result = metric(y[:,j],y_pred[:,j])
                metric_list.append(result)
                metric_string.append('{0: <25}'.format(str(result)))

            print('{0: <25}'.format(name)+''.join(metric_string))
            epoch_list.append(metric_list)

        print("\n")
        lista.append(epoch_list)

        return None
     

    def minibatch(self,inputs,targets,batch_size):
        """
        A generator for batches of the input data

        Args:
            inputs (array[float]): input data
            targets (array[float]): targets

        Returns:
            inputs (array[float]): one batch of input data of length `batch_size`
            targets (array[float]): one batch of targets of length `batch_size`
        """
        for start_idx in range(0, inputs.shape[0], batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]

   
    def predict(self,inputs):
        return self.call_map(self.weights,inputs)
        
    def fit(self,x_train,y_train,x_test = None,y_test = None, epochs: int = 30, validation_split = None):

        energy = lambda x: self.loss(self.call_map,x,x_train,y_train)
        energy = jax.jit(energy)

        # Metrics
        print("Initial value: ")
        print("#######################################################################")
        print("Training: ")
        self.compute_metrics(x_train,y_train,self.metric_train_history)
        print("Test: ")
        self.compute_metrics(x_test,y_test,self.metric_test_history)
        print("#######################################################################")


        for it in range(epochs):
            # Optimize
            grads = jax.grad(energy)(self.weights)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.weights = optax.apply_updates(self.weights, updates)
            
            # Metrics
            print("Epoch: ", it)
            print("#######################################################################")
            print("Training: ")
            self.compute_metrics(x_train,y_train,self.metric_train_history)
            print("Test: ")
            self.compute_metrics(x_test,y_test,self.metric_test_history)
            print("#######################################################################")
            
        return np.array(self.metric_train_history), np.array(self.metric_test_history)


    def plot(self):
        print(qml.draw(self.circuit_plot)(self.weights,[i for i in range(self.n_inputs)]))
        return None

