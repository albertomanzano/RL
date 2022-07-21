
# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq, sympy
from functools import reduce
import numpy as np
tf.get_logger().setLevel('ERROR')


class ReUploadingPQCLayer(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, n_inputs,n_outputs, n_layers, activation="linear", name="re-uploading_PQC_layer"):
        super(ReUploadingPQCLayer, self).__init__(name=name)
        # Dimensions of the circuit
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_qubits = n_inputs+n_outputs
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)

        # Arquitecture of the layer
        self.init_observables()
        self.generate_circuit()
        self.init_weights()
        
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(self.circuit, self.observables)

    def init_weights(self):
        # Initialization of the encoding and weights
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(len(self.theta_symbols),), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_inputs,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=False, name="lambdas"
        )
        return None

    def init_observables(self):
        ops = [ cirq.Z(self.qubits[i])
                for i in range(self.n_inputs,self.n_inputs+self.n_outputs)]
        #self.observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3
        self.observables = ops 
        return None


    def generate_circuit(self):
        """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""

        self.circuit = cirq.Circuit()
        self.input_symbols = sympy.symbols(f'x(0:{self.n_inputs})')
        self.theta_symbols = []

        # Define circuit
        for l in range(self.n_layers):
            # Variational layer
            self.encoding_layer()
            self.variational_layer(l)
            self.entangling_layer() 

        self.input_symbols = list(self.input_symbols)
        self.theta_symbols = list(self.theta_symbols)

        # Define explicit symbol order.
        symbols = [str(symb) for symb in self.theta_symbols + self.input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        
        return None
    
    def encoding_layer(self):
        for i in range(self.n_inputs):
            self.circuit.append(cirq.rx(self.input_symbols[i]).on(self.qubits[i]))
            self.circuit.append(cirq.ry(self.input_symbols[i]).on(self.qubits[i]))
            self.circuit.append(cirq.rz(self.input_symbols[i]).on(self.qubits[i]))
        return None
    
    def entangling_layer(self):
        """
        Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
        """
        for i in range(self.n_qubits-1):
            self.circuit.append(cirq.CNOT(self.qubits[i],self.qubits[i+1]))
        self.circuit.append(cirq.CNOT(self.qubits[-1],self.qubits[0]))
        return None
    
    
    def entangling_layer_input_output(self):
        """
        Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
        """
        #for i in range(len(self.qubits)-1):
        #    self.circuit.append(cirq.CNOT(self.qubits[i],self.qubits[i+1]))
        #self.circuit.append(cirq.CNOT(self.qubits[-1],self.qubits[0]))
        for i in range(self.n_inputs):
            for j in range(self.n_inputs,self.n_qubits):
                self.circuit.append(cirq.CNOT(self.qubits[i],self.qubits[j]))
        return None
    
    def entangling_layer_output_input(self):
        """
        Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
        """
        #for i in range(len(self.qubits)-1):
        #    self.circuit.append(cirq.CNOT(self.qubits[i],self.qubits[i+1]))
        #self.circuit.append(cirq.CNOT(self.qubits[-1],self.qubits[0]))
        for i in range(self.n_inputs):
            for j in range(self.n_inputs,self.n_qubits):
                self.circuit.append(cirq.CNOT(self.qubits[j],self.qubits[i]))
        return None

    
    def variational_layer(self, layer):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        for i in range(self.n_inputs):
            theta_x = sympy.symbols(f'{i}_'+f'{3*layer}')
            theta_y = sympy.symbols(f'{i}_'+f'{3*layer+1}')
            theta_z = sympy.symbols(f'{i}_'+f'{3*layer+2}')
            self.theta_symbols.append(theta_x)
            self.theta_symbols.append(theta_y)
            self.theta_symbols.append(theta_z)
            self.circuit.append(cirq.rx(theta_x).on(self.qubits[i]))
            self.circuit.append(cirq.ry(theta_y).on(self.qubits[i]))
            self.circuit.append(cirq.rz(theta_z).on(self.qubits[i]))

        return None
    
    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        #batch_dim = tf.gather(tf.shape(inputs),0)
        batch_dim = tf.shape(inputs)[0]
        
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile([self.theta], multiples=[batch_dim, 1])
        tiled_up_inputs = inputs

        joined_vars = tf.concat([tiled_up_thetas, tiled_up_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        output = self.computation_layer([tiled_up_circuits, joined_vars])
        return output
    
    

class ReUploadingPQC():
    """Generates a Keras model for a data re-uploading PQC policy."""
    def __init__(self,n_inputs,n_outputs, n_layers):
        pi = tf.constant(np.pi)

        # Define model
        input_tensor = tf.keras.Input(shape=(n_inputs, ), dtype=tf.dtypes.float32, name='input')
        preprocess = tf.keras.layers.Lambda(lambda x: self.encoding(x),output_shape=(n_inputs,))(input_tensor)
        re_uploading_pqc = ReUploadingPQCLayer(n_inputs,n_outputs, n_layers)
        postprocess = tf.keras.layers.Lambda(lambda x: self.decoding(x))     
        policy = postprocess(re_uploading_pqc(preprocess))
        
        self.model = tf.keras.Model(inputs=input_tensor, outputs=policy)

        # Define circuit
        self.circuit = re_uploading_pqc.circuit

    def encoding(self,x):
        tau = tf.gather(x,indices = [0], axis = 1)
        s = tf.gather(x,indices = [1], axis = 1)
        tau_transformed = tf.atan(tau)
        s_transformed = tf.atan(s-1.0)
        output = tf.concat([tau_transformed,s_transformed],axis = 1)
        return output

    def decoding(self,x):
        return tf.tan(x)

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
 
    
    def __call__(self,inputs):
        return self.model(inputs)
