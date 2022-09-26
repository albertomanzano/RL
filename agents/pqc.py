import pennylane as qml
import matplotlib.pyplot as plt
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
from pennylane import numpy as np

class PQC:

    def __init__(self,n_inputs: int = 1,n_outputs: int = 1,n_layers: int = 1, simulator: str = "forest.numpy_wavefunction"):
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.uniform(size = (self.n_inputs+self.n_outputs,self.n_layers),requires_grad = True)
        
        #self.dev = qml.device("forest.qvm", device="{}q-pyqvm".format(n_inputs+n_outputs), shots = 1024)
        self.dev = qml.device(simulator, wires=self.n_inputs+self.n_outputs)
        self.circuit = qml.QNode(self.arquitecture,self.dev)

        # Training
        self.optimizer = None


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
                qml.Rot(0,0,weights[i,l],wires=i)
            
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
        x = np.array(x,requires_grad = True)
        y = self.circuit(weights,x)
        y_grad = qml.gradients.param_shift(self.circuit)(weights,x)
        solution = np.hstack([y,y_grad[1]])
        return solution
    
    def predict(self,x):
        ''' Processes an array of inputs'''
        y = np.zeros((len(x),self.n_inputs+self.n_outputs),requires_grad = False)
        for i in range(len(x)):
            y[i] = self.call(self.weights,x[i])
        
        return y

    def compile(self,optimizer = None, loss = None, metrics = None, loss_weights = None):
        if optimizer is None:
            optimizer = AdamOptimizer(0.06, beta1=0.9, beta2=0.999) 
        self.optimizer = optimizer

    def L2(self,y1,y2):
        return np.power(y1-y2,2.0)
    
    def cost(self,weights,inputs, outputs,loss_weights):
        """Cost function to be minimized.

        Args:
            x (array[float]): array of input vectors
            y (array[float]): array of targets

        Returns:
            float: loss value to be minimized
        """
        # Compute prediction for each input in data batch
        loss = 0.0
        for i, x in enumerate(inputs):
            pred   = self.call(weights,x)
            loss_i = self.L2(outputs[i],pred)
            for j in range(self.n_inputs+1):
                loss = loss + loss_weights[j]*loss_i[j]
        return loss
     

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

    
    def fit(self,x_train,y_train,x_test = None,y_test = None, batch_size: int = 32, epochs: int = 30, validation_split = None, loss_weights = None, metric_weights = None):

        if loss_weights is None:
            loss_weights = np.array([i for i in range(self.n_inputs+1)],requires_grad = False)

        if metric_weights is None:
            metric_weights = np.array([i for i in range(self.n_inputs+1)],requires_grad = False)

        
        # Save training data 
        loss_train_history = np.zeros(epochs+1,requires_grad = False)
        metric_train_history = np.zeros(epochs+1,requires_grad = False)
        loss_test_history = np.zeros(epochs+1,requires_grad = False)
        metric_test_history = np.zeros(epochs+1,requires_grad = False)

        # Define optimizer
        learning_rate = 0.01
        
        # Train
        y_train_pred  = self.predict(x_train) 
        l2_train = np.sum(self.L2(y_train,y_train_pred),axis = 0)
        
        loss_train = np.dot(loss_weights,l2_train)
        metric_train = np.dot(metric_weights,l2_train)
        
        loss_train_history[0] = loss_train
        metric_train_history[0] = metric_train
        
        # Test
        y_test_pred = self.predict(x_test)
        l2_test = np.sum(self.L2(y_test,y_test_pred),axis = 0)
            
        loss_test = np.dot(loss_weights,l2_test) 
        metric_test = np.dot(metric_weights,l2_test) 
            
        loss_test_history[0] = loss_test
        metric_test_history[0] = metric_test
            

        print(f"Initial | L2 train: {l2_train} | Loss train: {loss_train} | Metric train: {metric_train} | L2 test: {l2_test} | Loss test: {loss_test} | Metric test: {metric_test}")

        for it in range(epochs):
            for x_batch, y_batch in self.minibatch(x_train, y_train, batch_size=batch_size):
                #grad = self.optimizer.compute_grad(self.cost,(self.weights,x_batch, y_batch, loss_weights),{})
                self.weights, _, _, _ = self.optimizer.step(self.cost,self.weights,x_batch, y_batch, loss_weights)
            
            # Train
            y_train_pred  = self.predict(x_train) 
            l2_train = np.sum(self.L2(y_train,y_train_pred),axis = 0)
        
            loss_train = np.dot(loss_weights,l2_train)
            metric_train = np.dot(metric_weights,l2_train)
            
            loss_train_history[it+1] = loss_train
            metric_train_history[it+1] = metric_train
            
            # Test
            y_test_pred = self.predict(x_test)
            l2_test = np.sum(self.L2(y_test,y_test_pred),axis = 0)
            
            loss_test = np.dot(loss_weights,l2_test) 
            metric_test = np.dot(metric_weights,l2_test) 
            
            loss_test_history[it+1] = loss_test
            metric_test_history[it+1] = metric_test
        
            print(f"Epoch: {it} | L2 train: {l2_train} | Loss train: {loss_train} | Metric train: {metric_train} | L2 test: {l2_test} | Loss test: {loss_test} | Metric test: {metric_test}")

        return loss_train_history, loss_test_history, metric_train_history, metric_test_history


    def plot(self):
        print(qml.draw(self.circuit)(self.weights,[i for i in range(self.n_inputs+self.n_outputs)]))
        return None

def mesh(foo, test_size = 0.0):
    # Define shuffled mesh
    coordinates = np.array(np.meshgrid(*foo)).T.reshape(-1, len(foo))
    np.random.shuffle(coordinates)
    # Divide in validation and training
    train_index = int(len(coordinates)*(1.-test_size))
    input_train = coordinates[:train_index]
    input_test = coordinates[train_index:]
    return input_train, input_test
