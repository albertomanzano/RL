# Test for ReUploadingPQC building
import sys
sys.path.append("./")

from agents.re_uploading_pqc import ReUploadingPQC 
import tensorflow as tf

# Dimensions of the problem
n_inputs = 2 
n_outputs = 1
n_layers = 2 

model = ReUploadingPQC(n_inputs,n_outputs,n_layers)

constant = tf.constant([[2,3],[6,7]])
model.print_circuit()
model(constant)

