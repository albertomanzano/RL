import pennylane as qml
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import optax

def mesh(foo, test_size = 0.0):
    # Define shuffled mesh
    coordinates = np.array(np.meshgrid(*foo)).T.reshape(-1, len(foo))
    generator = jax.random.PRNGKey(0)
    jax.random.permutation(generator,coordinates, independent = True)
    # Divide in validation and training
    train_index = int(len(coordinates)*(1.-test_size))
    input_train = coordinates[:train_index]
    input_test = coordinates[train_index:]
    return input_train, input_test

@jax.jit
def MSE(y, y_pred): 
    return np.sum(np.power(y-y_pred,2.0), axis = 0)/y.shape[0]

@jax.jit
def MAE(y, y_pred): 
    return np.sum(np.abs(y-y_pred), axis = 0)/y.shape[0]

class PQC:

    def __init__(self,n_inputs: int = 1,n_outputs: int = 1,n_layers: int = 1, simulator: str = "default.qubit.jax"):
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        #self.dev = qml.device("forest.qvm", device="{}q-pyqvm".format(n_inputs+n_outputs), shots = 1024)
        self.dev = qml.device(simulator, wires=self.n_inputs+self.n_outputs)
        self.circuit = qml.QNode(self.arquitecture,self.dev,interface = "jax")
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


    def arquitecture(self,weights,x):
        """A variational quantum circuit representing the Universal classifier.

        Args:
            params (array[float]): array of parameters
            x (array[float]): single input vector

        Returns:
            float: fidelity between output state and input
        """
        # Initialization
        for i in range(self.n_inputs+self.n_outputs):
            qml.Hadamard(wires = i)

        for l in range(self.n_layers):
            
            # Variational layer
            for i in range(self.n_inputs+self.n_outputs):
                qml.Rot(weights[i,l],weights[i,l],weights[i,l],wires=i)
            
            # Encoding layer
            for i in range(self.n_inputs):
                qml.Rot(x[i],x[i],x[i],wires=i)
                
            # Entangling layer
            for i in range(self.n_inputs+self.n_outputs-1):
                qml.CNOT(wires = [i,i+1])
            qml.CNOT(wires = [self.n_inputs+self.n_outputs-1,0])

        # Operator
        #op = qml.prod(*[ qml.PauliX(wires = i) for i in range(self.n_inputs) ])
        op = qml.PauliX(wires = self.n_inputs+self.n_outputs-1)
        return qml.expval(op)

    def call(self,weights,x):
        ''' Processes a single input'''
        y = self.circuit(weights,x)
        y_grad = self.gradient_circuit(weights,x)
        solution = np.hstack([y,y_grad])
        return solution
    

    def compile(self,optimizer = None, loss = None, metrics = None,mask = None, loss_weights = None, metric_weights = None):
        
        # Set weights
        generator = jax.random.PRNGKey(0)
        self.weights = jax.random.uniform(generator,(self.n_inputs+self.n_outputs,self.n_layers))
        
        # Optimizer
        if optimizer is None:
            optimizer = optax.adam(0.01)
        self.optimizer = optimizer
        self.opt_state = optimizer.init(self.weights)

        # Loss
        if loss is None:
            loss = MSE
        self.loss = loss

        # Metrics
        if metrics is None:
            metrics = {"MSE": MSE}
        self.metrics = metrics

        # Loss and metric weights
        if loss_weights is None:
            loss_weights = np.array([i for i in range(self.n_inputs+1)])
        self.loss_weights = loss_weights

        if metric_weights is None:
            metric_weights = np.array([i for i in range(self.n_inputs+1)])
        self.metric_weights = metric_weights

        
        # History
        self.metric_train_history = []
        self.metric_test_history = []
        
        return None

    
    def cost(self,weights,inputs, outputs,loss_weights):
        """Cost function to be minimized.

        Args:
            x (array[float]): array of input vectors
            y (array[float]): array of targets

        Returns:
            float: loss value to be minimized
        """
        # Compute prediction for each input in data batch
        pred   = self.call_map(weights,inputs)
        loss_i = self.loss(outputs,pred)
        return np.dot(loss_weights,loss_i)

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

        energy = lambda x: self.cost(x,x_train,y_train,self.loss_weights) 

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
        print(qml.draw(self.circuit)(self.weights,[i for i in range(self.n_inputs+self.n_outputs)]))
        return None

