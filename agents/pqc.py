# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

class PQC():
    def __init__(self,input_space = 2,output_space = 1,n_layers = 2, arquitecture: str = None):
        self.input_space = input_space
        self.output_space = output_space
        self.n_qubits = input_space+output_space
        self.n_layers = n_layers

        # Build the circuit
        self.circuit = cirq.Circuit()
        self.qubits = cirq.GridQubit.rect(1,self.n_qubits)
        if (arquitecture=="init1"):
            self.init_1()
        elif (arquitecture=="init2"):
            self.init_2()
        elif (arquitecture=="init3"):
            self.init_3()
        elif (arquitecture=="init4"):
            self.init_4()
        elif (arquitecture=="init5"):
            self.init_5()
        else:
            self.init_6()


        self.readout = cirq.Z(self.qubits[-1])
        
        
        # Build the Keras model.
        self.model = tf.keras.Sequential([
            # The input
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the readout gate, range [-1,1].
            tfq.layers.PQC(self.circuit, self.readout),
        ])
    
    def atan_mapping(self,x):
        return np.arctan(x)/np.pi*2

    def angle_encoding(self,x):
        """Encode truncated classical image into quantum datapoint."""
        qubits = cirq.GridQubit.rect(1, self.n_qubits)
        angle = np.arcsin(x)
        circuit = cirq.Circuit()
        for i in range(len(angle)):
            circuit.append(cirq.ry(angle[i]).on(qubits[i]))
        return circuit

    def convert_to_circuit(self,x):
        x_normalised = self.atan_mapping(x)
        circuit = self.angle_encoding(x_normalised)

        return circuit

    def layer_single(self,gate,layer_number):
        for i in range(self.input_space):
            symbol = sympy.Symbol(str(layer_number) + '-' + str(i))
            self.circuit.append(gate(symbol).on(self.qubits[i]))

    def output_entangling(self,gate):
        for i in range(self.input_space):
            for j in range(self.output_space):
                self.circuit.append(gate(self.qubits[i],self.qubits[self.input_space+j]))

    def cyclic_entangling(self,gate):
        self.circuit.append( [ gate(q0, q1) for q0, q1 in zip(self.qubits, self.qubits[1:])])
        self.circuit.append( gate(self.qubits[-1], self.qubits[0]) )

    def init_1(self):
        for i in range(self.n_layers):
            self.layer_single(cirq.ry,2*i)
            self.output_entangling(cirq.CNOT)
        self.number_parameters = self.n_layers*self.n_qubits
    
    def init_2(self):
        # Output entagling and same parameter for rx,ry,rz
        for i in range(self.n_layers):
            self.layer_single(cirq.ry,i)
            self.layer_single(cirq.rx,i)
            self.layer_single(cirq.rz,i)
            self.output_entangling(cirq.CNOT)
        self.number_parameters = self.n_layers*self.n_qubits
    
    def init_3(self):
        # Cyclic entagling and same parameter for rx,ry,rz
        for i in range(self.n_layers):
            self.layer_single(cirq.ry,i)
            self.layer_single(cirq.rx,i)
            self.layer_single(cirq.rz,i)
            self.cyclic_entangling(cirq.CNOT)
        self.number_parameters = self.n_layers*self.n_qubits
    
    def init_4(self):
        # Cyclic entagling and same parameter for rx,ry,rz
        for i in range(self.n_layers):
            self.layer_single(cirq.ry,3*i)
            self.layer_single(cirq.rx,3*i+1)
            self.layer_single(cirq.rz,3*i+2)
            self.cyclic_entangling(cirq.CNOT)
        self.number_parameters = self.n_layers*self.n_qubits
    
    def init_5(self):
        # Cyclic entagling and same parameter for rx,ry,rz
        self.circuit.append(cirq.H(self.qubits[-1]))
        for i in range(self.n_layers):
            self.layer_single(cirq.ry,3*i)
            self.layer_single(cirq.rx,3*i+1)
            self.layer_single(cirq.rz,3*i+2)
            self.cyclic_entangling(cirq.CNOT)
        self.number_parameters = self.n_layers*self.n_qubits
        self.circuit.append(cirq.H(self.qubits[-1]))
    
    def init_6(self):
        # Cyclic entagling and same parameter for rx,ry,rz
        for i in range(self.n_layers):
            self.layer_single(cirq.ry,3*i)
            self.layer_single(cirq.rx,3*i+1)
            self.layer_single(cirq.rz,3*i+2)
            for j in range(self.input_space):
                self.circuit.append(cirq.H(self.qubits[j]))
            self.cyclic_entangling(cirq.CNOT)
        self.number_parameters = self.n_layers*self.n_qubits

    def print_circuit(self):
        print(self.circuit)
        return None
    
 
    def fit(self,x_train,y_train,x_test,y_test,epochs = 2):
        self.model.compile(
            loss=tf.keras.losses.MeanAbsoluteError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])

        print(self.model.summary())

        qnn_history = self.model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

        return qnn_history

    def predict(self,x):
        d = len(x)
        d_i = [ len(element) for element in x ]
        grid = np.array(np.meshgrid(*x)).T.reshape(-1,d)
        grid_size = len(grid)
        circuits = [ self.convert_to_circuit(element) for element in grid ] 
        tensor = tfq.convert_to_tensor(circuits)
        prediction = self.model.predict(tensor)
        prediction = prediction.reshape(d_i).T

        return prediction

    def __call__(self,x):
        return self.predict(x)
       



        
