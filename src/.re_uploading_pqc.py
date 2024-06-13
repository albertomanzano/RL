
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

    def __init__(self,  n_inputs: int = 1,
                        n_outputs: int = 1, 
                        n_layers: int = 1, 
                        schedule: str = 'exp',
                        entangling: str = 'cyclic',
                        arquitecture: str = 'rxryrz',
                        repetitions: int = 1, 
                        activation="linear", name="re-uploading_PQC_layer"):
        
        super(ReUploadingPQCLayer, self).__init__(name=name)
        # Dimensions of the circuit
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_qubits = n_inputs+n_outputs
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
       
        # Arquitecture
        self.schedule = schedule
        self.entangling = entangling
        self.arquitecture = arquitecture
        self.repetitions = repetitions

        # Initialize PQC
        self.init_observables()
        self.init_circuit()
        self.init_weights()
        
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(self.circuit, self.observables)

##################################################
# Initialization 
##################################################
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
        ops = [ cirq.Y(self.qubits[i])
                for i in range(self.n_inputs,self.n_inputs+self.n_outputs)]
        #self.observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3
        self.observables = ops 
        return None


    def init_circuit(self):
        """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""

        self.circuit = cirq.Circuit()
        self.input_symbols = sympy.symbols(f'x(0:{self.n_inputs})')
        self.theta_symbols = []

        # Define circuit
        for l in range(self.n_layers):
            # Variational layer
            self.variational_layer(l)
            self.encoding_layer(l)
            self.entangling_layer() 

        self.input_symbols = list(self.input_symbols)
        self.theta_symbols = list(self.theta_symbols)

        # Define explicit symbol order.
        symbols = [str(symb) for symb in self.theta_symbols + self.input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        
        return None
    
##################################################
# Encoding Layer 
##################################################

    def encoding_layer(self,n_layer):
            if self.schedule == 'linear':
                self.linear_encoding(n_layer)
            elif self.schedule == 'rx_constant':
                self.rx_constant_encoding(n_layer)
            elif self.schedule == 'rx_linear':
                self.rx_linear_encoding(n_layer)
            elif self.schedule == 'rx_exp':
                self.rx_exp_encoding(n_layer)
            else: 
                self.exponential_encoding(n_layer)

            return None
    
    def rx_constant_encoding(self,n_layer):
        for i in range(self.n_inputs):
            self.circuit.append(cirq.rx(self.input_symbols[i]).on(self.qubits[i]))
        return None
    
    def rx_linear_encoding(self,n_layer):
        for i in range(self.n_inputs):
            factor = n_layer+1
            self.circuit.append(cirq.rx(self.input_symbols[i]*factor).on(self.qubits[i]))
        return None
    
    def rx_exp_encoding(self,n_layer):
        for i in range(self.n_inputs):
            factor = 2**n_layer
            self.circuit.append(cirq.rx(self.input_symbols[i]*factor).on(self.qubits[i]))
        return None

    def linear_encoding(self,n_layer):
        for i in range(self.n_inputs):
            self.circuit.append(cirq.Z.on(self.qubits[i])**(self.input_symbols[i]/2))
            self.circuit.append(cirq.X.on(self.qubits[i]))
            self.circuit.append(cirq.Z.on(self.qubits[i])**(-self.input_symbols[i]/2))
            self.circuit.append(cirq.X.on(self.qubits[i]))
        return None

    def exponential_encoding(self,n_layer):
        for i in range(self.n_inputs):
            self.circuit.append(cirq.Z.on(self.qubits[i])**(2**(n_layer)*self.input_symbols[i]/2))
            self.circuit.append(cirq.X.on(self.qubits[i]))
            self.circuit.append(cirq.Z.on(self.qubits[i])**(-2**(n_layer)*self.input_symbols[i]/2))
            self.circuit.append(cirq.X.on(self.qubits[i]))
        return None

##################################################
# Entangling layer
##################################################
    
    def entangling_layer(self):
        """
        Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
        """
        if self.entangling == "input_output":
            self.entangling_layer_input_output()
        elif self.entangling == "output_input":
            self.entangling_layer_output_input()
        else:
            self.entangling_cyclic()

        return None
    
    def entangling_cyclic(self):
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

##################################################
# Variational Layer 
##################################################
    
    def variational_layer(self, layer):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        if self.arquitecture=='rot':
            self.variational_rot(layer)
        else:
            self.variational_rxryrz(layer)

        return None
    
    def variational_rxryrz(self, layer):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        for i in range(self.n_qubits):
            for j in range(self.repetitions):
                theta_x = sympy.symbols(f'{i}_'+f'{3*layer}_'+f'{j}')
                theta_y = sympy.symbols(f'{i}_'+f'{3*layer+1}_'+f'{j}')
                theta_z = sympy.symbols(f'{i}_'+f'{3*layer+2}_'+f'{j}')
                self.theta_symbols.append(theta_x)
                self.theta_symbols.append(theta_y)
                self.theta_symbols.append(theta_z)
                self.circuit.append(cirq.rx(theta_x).on(self.qubits[i]))
                self.circuit.append(cirq.ry(theta_y).on(self.qubits[i]))
                self.circuit.append(cirq.rz(theta_z).on(self.qubits[i]))

        return None
    
    def variational_rot(self, layer):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        for i in range(self.n_qubits):
            for j in range(self.repetitions):
                theta_x = sympy.symbols(f'{i}_'+f'{3*layer}_'+f'{j}')
                theta_y = sympy.symbols(f'{i}_'+f'{3*layer+1}_'+f'{j}')
                theta_z = sympy.symbols(f'{i}_'+f'{3*layer+2}_'+f'{j}')
                self.theta_symbols.append(theta_x)
                self.theta_symbols.append(theta_y)
                self.theta_symbols.append(theta_z)
                self.circuit.append(cirq.rz(theta_x).on(self.qubits[i]))
                self.circuit.append(cirq.ry(theta_y).on(self.qubits[i]))
                self.circuit.append(cirq.rz(theta_z).on(self.qubits[i]))

        return None

##################################################
# Other methods 
##################################################
    
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
    def __init__(self,  n_inputs: int = 1,
                        n_outputs: int = 1,
                        n_layers: int = 1,
                        schedule: str = 'exp',
                        entangling: str = 'cyclic',
                        arquitecture: str = 'rxryrz',
                        repetitions: int = 1,
                        base_frequencies = None):

        if base_frequencies is None:
            self.base_frequencies = tf.ones([n_inputs,1],dtype=tf.dtypes.float32)
        else:
            self.base_frequencies = tf.constant(base_frequencies,dtype=tf.dtypes.float32)

        # Define model
        input_tensor = tf.keras.Input(shape=(n_inputs, ), dtype=tf.dtypes.float32, name='input')
        preprocess = tf.keras.layers.Lambda(lambda x: self.encoding(x),output_shape=(n_inputs,))(input_tensor)
        re_uploading_pqc = ReUploadingPQCLayer( n_inputs = n_inputs,
                                                n_outputs = n_outputs, 
                                                n_layers = n_layers,
                                                schedule = schedule,
                                                entangling = entangling,
                                                arquitecture = arquitecture,
                                                repetitions = repetitions)

        postprocess = tf.keras.layers.Lambda(lambda x: self.decoding(x))     
        policy = postprocess(re_uploading_pqc(preprocess))
        
        self.model = tf.keras.Model(inputs=input_tensor, outputs=policy)

        # Define circuit
        self.circuit = re_uploading_pqc.circuit

    def encoding(self,x):
        output = self.base_frequencies*x
        return output

    def decoding(self,x):
        return x

    def print_circuit(self):
        print(self.circuit)
        return None

    def summary(self):
        re_uploading_pqc_layer = self.model.layers[2]
        if re_uploading_pqc_layer.schedule=="rx_linear":
            n_frequencies = int(re_uploading_pqc_layer.n_layers*(re_uploading_pqc_layer.n_layers+1)/2)
        elif re_uploading_pqc_layer.schedule=="rx_exp":
            n_frequencies = int(2**re_uploading_pqc_layer.n_layers-1)
        else:
            n_frequencies = re_uploading_pqc_layer.n_layers
        
        print("\n") 
        print("##################################################")
        print("N inputs: ",re_uploading_pqc_layer.n_inputs)
        print("N outputs: ",re_uploading_pqc_layer.n_outputs)
        print("-----------------------------------------")
        print("N layers: ",re_uploading_pqc_layer.n_layers)
        print("Entangling: ",re_uploading_pqc_layer.entangling)
        print("Arquitecture: ",re_uploading_pqc_layer.arquitecture)
        print("-----------------------------------------")
        print("Schedule: ",re_uploading_pqc_layer.schedule)
        print("Number of frequencies: ",n_frequencies)
        print("##################################################")
        print("\n") 
        return None
    
    def fit(self,x,y,epochs = 2,validation_split = 0.2):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=3e-2,
                decay_steps=5,
                decay_rate=0.1)
             
        self.model.compile(
            loss=tf.keras.losses.MeanAbsoluteError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])

        #print(self.model.summary())

        qnn_history = self.model.fit(
            x, y,
            batch_size=200,
            epochs=epochs,
            verbose=1,
            validation_split = validation_split) 

        return qnn_history
 
    
    def __call__(self,inputs):
        return self.model(inputs)
