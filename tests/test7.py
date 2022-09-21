import pennylane as qml
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
from pennylane import numpy as np

class ReUploadingPqc:

    def __init__(self,n_inputs: int = 1,n_layers: int = 1, simulator: str = "lightning.gpu"):
        self.dev = qml.device(simulator, wires=n_inputs)
        self.n_layers = n_layers
        self.weights = np.random.uniform(size = (3,n_layers),requires_grad = True)
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
        for l in range(self.n_layers):
            qml.Rot(x,x,x,wires=0)
            qml.Rot(weights[0,l],weights[1,l],weights[2,l],wires=0)
        return qml.expval(qml.PauliZ(0))

    def call(self,inputs,training = False):
            return self.circuit(weights,x[i])

    def compile(self,optimizer = None, loss = None, metrics = None, loss_weights = None):
        if optimizer is None:
            optimizer = AdamOptimizer(0.06, beta1=0.9, beta2=0.999) 
        self.optimizer = optimizer
        

    def evaluate(self,x: np.array):
        y_pred = np.zeros(len(x),requires_grad = False)
        for i in range(len(x)):
            y_pred[i] = self.call(x[i])
        return y_pred

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

    def cost(self,weights,x, y):
        """Cost function to be minimized.

        Args:
            x (array[float]): array of input vectors
            y (array[float]): array of targets

        Returns:
            float: loss value to be minimized
        """
        # Compute prediction for each input in data batch
        loss = 0.0
        for i in range(len(x)):
            y_pred  = self.circuit(weights,x[i])
            loss = loss+(y[i]-y_pred)**2
        return loss
    
    def fit(self,x,y, batch_size: int = 32, epochs: int = 20, validation_split = None, validation_data = None):
        # Save training data 
        loss_train_history = np.zeros(epochs+1,requires_grad = False)
        loss_test_history = np.zeros(epochs+1,requires_grad = False)

        # Define train and test
        x_train = x
        y_train = y
        x_test = validation_data[0]
        y_test = validation_data[1]
        
        # Define optimizer
        learning_rate = 0.6

        # First step
        loss_train = self.cost(self.weights,x_train, y_train)
        loss_test = self.cost(self.weights,x_test, y_test)
        loss_train_history[0] = loss_train
        loss_test_history[0] = loss_test
        print(f"Initial loss | Loss train: {loss_train} | Loss test: {loss_test}")

        for it in range(epochs):
            for x_batch, y_batch in self.minibatch(x_train, y_train, batch_size=batch_size):
                self.weights, _, _ = self.optimizer.step(self.cost,self.weights,x_batch, y_batch)
        
            loss_train = self.cost(self.weights,x_train, y_train)
            loss_test = self.cost(self.weights,x_test, y_test)
            loss_train_history[it+1] = loss_train
            loss_test_history[it+1] = loss_test
        
            print(f"Epoch: {it} | Loss train: {loss_train} | Loss test: {loss_test}")

        return loss_train_history, loss_test_history


    def plot(self):
        print(qml.draw(self.circuit)(self.weights))
        return None



n_inputs = 1
re_uploading_pqc = ReUploadingPqc(n_inputs)

# Data
x = np.linspace(0,np.pi/2,64,requires_grad = False)
y = np.cos(x)

re_uploading_pqc.compile()
re_uploading_pqc.fit(x,y,validation_data = (x,y))
