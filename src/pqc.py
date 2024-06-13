import pennylane as qml
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import optax
from src.losses import spvsd_loss
from src.metrics import MSE_metric
from src.arquitectures import ConstantArquitecture
import time
import copy



class PQC:

    def __init__(self,arquitecture = None):
        
        if arquitecture is None:
            arquitecture = ConstantArquitecture(1,1,1,1)
        self.arquitecture = arquitecture
        
        # Structure of the network
        self.n_qubits = arquitecture.n_qubits
        

    def call(self,weights,x):
        ''' Processes a single input'''
        y = self.circuit(weights,x)
        y_grad = self.gradient_circuit(weights,x)
        solution = np.hstack([y,y_grad])
        return solution
    


    def compile(self,optimizer = None, loss = None, metrics = None,mask = None, generator = None,simulator: str = "default.qubit.jax"):
        # Set weights
        self.arquitecture.init_weights(generator)
        
        # Arquitecture
        
        # Main functions
        self.dev = qml.device(simulator, wires=self.n_qubits)
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
        self.metric_train_history = None
        self.metric_test_history = None
        
        
        # Optimizer
        if optimizer is None:
            optimizer = optax.adam(0.1)
        self.optimizer = optimizer
        self.opt_state = optimizer.init(self.arquitecture.weights)

        # Metrics
        if metrics is None:
            metrics = {"MSE": MSE_metric}
        self.metrics = metrics

        if loss is None:
            loss = losses.spvsd_loss
        self.loss = loss

        
        # History
        self.metric_train_history = {}
        self.metric_test_history = {}
        for key in self.metrics:
            self.metric_train_history[key] = []
            self.metric_test_history[key] = []
        
        return None

    

    def compute_metrics(self,x,y,dictionary):
        y_pred = self.call_map(self.arquitecture.weights,x)

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
            for j in range(y.shape[1]):
                result = metric(y[:,j],y_pred[:,j])
                metric_list.append(result)
                metric_string.append('{0: <25}'.format(str(result)))

            print('{0: <25}'.format(name)+''.join(metric_string))
            dictionary[name].append(metric_list)
            #epoch_list.append(metric_list)

        print("\n")
        #lista.append(epoch_list)

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
        return self.call_map(self.arquitecture.weights,inputs)
        
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
            grads = jax.grad(energy)(self.arquitecture.weights)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.arquitecture.weights = optax.apply_updates(self.arquitecture.weights, updates)
            
            # Metrics
            print("Epoch: ", it)
            print("#######################################################################")
            print("Training: ")
            self.compute_metrics(x_train,y_train,self.metric_train_history)
            print("Test: ")
            self.compute_metrics(x_test,y_test,self.metric_test_history)
            print("#######################################################################")

        for key in self.metrics:
            self.metric_train_history[key] = np.array(self.metric_train_history[key])
            self.metric_test_history[key] = np.array(self.metric_test_history[key])
            
        return self.metric_train_history, self.metric_test_history


    def plot(self):
        print(qml.draw(self.circuit_plot)(self.arquitecture.weights,[i for i in range(self.arquitecture.n_inputs)]))
        return None

