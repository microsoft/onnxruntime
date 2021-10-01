# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""

.. _l-orttraining-linreg-gpu:

Train a linear regression with onnxruntime-training on GPU
==========================================================

This example follows the same steps introduced in example
:ref:`l-orttraining-linreg-cpu` but on GPU.

**to be completed**

.. contents::
    :local:

A simple linear regression with scikit-learn
++++++++++++++++++++++++++++++++++++++++++++

This code begins like example :ref:`l-orttraining-linreg-gpu`.
It creates a graph to train a linear regression initialized
with random coefficients.
"""
import os
from pprint import pprint
import numpy as np
from pandas import DataFrame
from onnx import helper, numpy_helper, TensorProto, ModelProto
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
from onnxruntime import (
    InferenceSession, __version__ as ort_version, get_device, OrtValue,
    TrainingParameters, SessionOptions, TrainingSession)
import matplotlib.pyplot as plt
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

X, y = make_regression(n_features=2, bias=2)
X = X.astype(np.float32)
y = y.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)


def plot_dot(model):
    pydot_graph = GetPydotGraph(
        model.graph, name=model.graph.name, rankdir="TB",
        node_producer=GetOpNodeProducer("docstring"))
    return plot_graphviz(pydot_graph.to_string())
    

def onnx_linear_regression_training(coefs, intercept):
    if len(coefs.shape) == 1:
        coefs = coefs.reshape((1, -1))
    coefs = coefs.T

    X = helper.make_tensor_value_info(
        'X', TensorProto.FLOAT, [None, coefs.shape[0]])
    label = helper.make_tensor_value_info(
        'label', TensorProto.FLOAT, [None, coefs.shape[1]])
    Y = helper.make_tensor_value_info(
        'Y', TensorProto.FLOAT, [None, coefs.shape[1]])
    loss = helper.make_tensor_value_info('loss', TensorProto.FLOAT, [])

    # inference
    node_matmul = helper.make_node('MatMul', ['X', 'coefs'], ['y1'], name='N1')
    node_add = helper.make_node('Add', ['y1', 'intercept'], ['Y'], name='N2')

    # loss
    node_diff = helper.make_node('Sub', ['Y', 'label'], ['diff'], name='L1')
    node_square = helper.make_node('Mul', ['diff', 'diff'], ['diff2'], name='L2')
    node_square_sum = helper.make_node('ReduceSum', ['diff2'], ['loss'], name='L3')

    # initializer
    init_coefs = numpy_helper.from_array(coefs, name="coefs")
    init_intercept = numpy_helper.from_array(intercept, name="intercept")

    # graph
    graph_def = helper.make_graph(
        [node_matmul, node_add, node_diff, node_square, node_square_sum],
        'lrt',
        [X, label], [loss, Y],
        [init_coefs, init_intercept])
    model_def = helper.make_model(
        graph_def, producer_name='orttrainer', ir_version=7,
        producer_version=ort_version,
        opset_imports=[helper.make_operatorsetid('', 14)])
    return model_def

onx_train = onnx_linear_regression_training(
    np.random.randn(2).astype(np.float32),
    np.random.randn(1).astype(np.float32))

plot_dot(onx_train)

#########################################
# First iterations of training on GPU
# ++++++++++++++++++++++++++++++++++++
#
# Prediction needs an instance of class *InferenceSession*,
# the training needs an instance of class *TrainingSession*.
# Next function creates this one.

device = "cuda" if get_device() == 'GPU' else 'cpu'


def create_training_session(training_onnx, weights_to_train, loss_output_name='loss',
                            training_optimizer_name='SGDOptimizer',
                            device='cpu'):
    
    ort_parameters = TrainingParameters()
    ort_parameters.loss_output_name = loss_output_name
    ort_parameters.use_mixed_precision = False
    # ort_parameters.world_rank = -1
    # ort_parameters.world_size = 1
    ort_parameters.gradient_accumulation_steps = 1
    ort_parameters.allreduce_post_accumulation = False
    # ort_parameters.deepspeed_zero_stage = 0
    ort_parameters.enable_grad_norm_clip = False
    ort_parameters.set_gradients_as_graph_outputs = False
    # ort_parameters.use_memory_efficient_gradient = False
    # ort_parameters.enable_adasum = False

    output_types = {}
    for output in training_onnx.graph.output:
        output_types[output.name] = output.type.tensor_type

    ort_parameters.weights_to_train = set(weights_to_train)
    ort_parameters.training_optimizer_name = training_optimizer_name
    # ort_parameters.lr_params_feed_name = lr_params_feed_name

    ort_parameters.optimizer_attributes_map = {
        name: {} for name in weights_to_train}
    ort_parameters.optimizer_int_attributes_map = {
        name: {} for name in weights_to_train}

    session_options = SessionOptions()
    session_options.use_deterministic_compute = True

    if device == 'cpu':
        provider = ['CPUExecutionProvider']
    elif device.startswith("cuda"):
        provider = ['CUDAExecutionProvider']
    else:
        raise ValueError("Unexpected device %r." % device)
        

    session = TrainingSession(
        training_onnx.SerializeToString(), ort_parameters, session_options,
        providers=provider)
    return session


train_session = create_training_session(onx_train, ['coefs', 'intercept'])
print(train_session)

##########################################
# The coefficients.

state_tensors = train_session.get_state()
pprint(state_tensors)

######################################
# We can now check the coefficients are updated after one iteration.

bind_x = OrtValue.ortvalue_from_numpy(X_train[:1], device, 0)
bind_y = OrtValue.ortvalue_from_numpy(y_train[:1].reshape((-1, 1)), device, 0)
bind_lr = OrtValue.ortvalue_from_numpy(np.array([0.01], dtype=np.float32), device, 0)

bind = train_session.io_binding()

bind.bind_input(
    name='X', device_type=bind_x.device_name(), device_id=0,
    element_type=np.float32, shape=bind_x.shape(),
    buffer_ptr=bind_x.data_ptr())

bind.bind_input(
    name='label', device_type=bind_y.device_name(), device_id=0,
    element_type=np.float32, shape=bind_y.shape(),
    buffer_ptr=bind_y.data_ptr())

bind.bind_input(
    name='Learning_Rate', device_type=bind_lr.device_name(), device_id=0,
    element_type=np.float32, shape=bind_lr.shape(),
    buffer_ptr=bind_lr.data_ptr())

bind.bind_output('loss')
bind.bind_output('Y')

train_session.run_with_iobinding(bind)

outputs = bind.copy_outputs_to_cpu()
print(outputs)

##########################################
# We check the coefficients have changed.

state_tensors = train_session.get_state()
pprint(state_tensors)

##########################################
# Training on GPU
# +++++++++++++++
#
# We still need to implement a gradient descent.
# Let's wrap this into a class similar following scikit-learn's API.

class DataLoaderDevice:
    
    def __init__(self, X, y, batch_size=20, device='cpu'):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if X.shape[0] != y.shape[0]:
            raise VaueError(
                "Shape mismatch X.shape=%r, y.shape=%r." % (X.shape, y.shape))
        self.X = np.ascontiguousarray(X)
        self.y = np.ascontiguousarray(y)
        self.batch_size = batch_size
        self.device = device
    
    def __len__(self):
        return self.X.shape[0]
    
    def __iter__(self):
        N = 0
        b = len(self) - self.batch_size
        while N < len(self):
            i = np.random.randint(0, b)
            N += self.batch_size
            yield (
                OrtValue.ortvalue_from_numpy(
                    self.X[i:i+self.batch_size], self.device, 0),
                OrtValue.ortvalue_from_numpy(
                    self.y[i:i+self.batch_size], self.device, 0))

    @property
    def data(self):
        return self.X, self.y


data_loader = DataLoaderDevice(X_train, y_train, batch_size=2)


for i, batch in enumerate(data_loader):
    if i >= 2:
        break
    print("batch %r: %r" % (i, batch))

##########################################
# The training algorithm.

class CustomTraining:

    def __init__(self, model_onnx, weights_to_train, loss_output_name='loss',
                 max_iter=100, training_optimizer_name='SGDOptimizer', batch_size=10, 
                 eta0=0.01, alpha=0.0001, power_t=0.25, learning_rate='invscaling',
                 device='cpu', verbose=0):
        # See https://scikit-learn.org/stable/modules/generated/
        # sklearn.linear_model.SGDRegressor.html
        self.model_onnx = model_onnx
        self.batch_size = batch_size
        self.weights_to_train = weights_to_train
        self.loss_output_name = loss_output_name
        self.training_optimizer_name = training_optimizer_name
        self.verbose = verbose
        self.max_iter = max_iter
        self.eta0 = eta0
        self.alpha = alpha
        self.power_t = power_t
        self.learning_rate = learning_rate.lower()
        self.device = device

    def _init_learning_rate(self):
        eta0 = self.eta0
        if self.learning_rate == "optimal":
            typw = np.sqrt(1.0 / np.sqrt(self.alpha))
            self.eta0_ = typw / max(1.0, loss.dloss(-typw, 1.0))
            self.optimal_init_ = 1.0 / (self.eta0_ * self.alpha)
        else:
            self.eta0_ = self.eta0
        return self.eta0_

    def _update_learning_rate(self, t, eta):
        if self.learning_rate == "optimal":
            eta = 1.0 / (alpha * (self.optimal_init_ + t))
        elif self.learning_rate == "invscaling":
            eta = self.eta0_ / np.power(t + 1, self.power_t)
        return eta
        
    def fit(self, X, y):
        """
        Trains the model.
        :param X: features
        :param y: expected output
        """
        self.train_session_ = create_training_session(
            self.model_onnx, self.weights_to_train,
            loss_output_name=self.loss_output_name,
            training_optimizer_name=self.training_optimizer_name,
            device=self.device)
        
        data_loader = DataLoaderDevice(
            X, y, batch_size=self.batch_size, device=self.device)
        lr = self._init_learning_rate()
        self.input_names_ = [i.name for i in self.train_session_.get_inputs()]
        self.output_names_ = [o.name for o in self.train_session_.get_outputs()]
        self.loss_index_ = self.output_names_.index(self.loss_output_name)
        
        bind = self.train_session_.io_binding()
    
        loop = tqdm(range(self.max_iter)) if self.verbose else range(max_iter)
        train_losses = []
        for it in loop:
            bind_lr = OrtValue.ortvalue_from_numpy(
                np.array([lr], dtype=np.float32), device, 0)
            loss = self._iteration(data_loader, bind_lr, bind)
            lr = self._update_learning_rate(it, lr)
            if self.verbose > 1:
                loop.set_description("loss=%1.3g lr=%1.3g" % (loss, lr))
            train_losses.append(loss)
        self.train_losses_ = train_losses
        self.trained_coef_ = self.train_session_.get_state()
        return self
        
    def _iteration(self, data_loader, learning_rate, bind):
        actual_losses = []
        for batch_idx, (data, target) in enumerate(data_loader):
                
            bind.bind_input(
                name=self.input_names_[0], device_type=data.device_name(), device_id=0,
                element_type=np.float32, shape=data.shape(),
                buffer_ptr=data.data_ptr())

            bind.bind_input(
                name=self.input_names_[1], device_type=target.device_name(), device_id=0,
                element_type=np.float32, shape=target.shape(),
                buffer_ptr=target.data_ptr())

            bind.bind_input(
                name=self.input_names_[2],
                device_type=learning_rate.device_name(), device_id=0,
                element_type=np.float32, shape=learning_rate.shape(),
                buffer_ptr=learning_rate.data_ptr())

            bind.bind_output('loss')

            self.train_session_.run_with_iobinding(bind)
            outputs = bind.copy_outputs_to_cpu()
            actual_losses.append(outputs[self.loss_index_])
        return np.array(actual_losses).mean()

###########################################
# Let's now train the model in a very similar way
# that it would be done with *scikit-learn*.

trainer = CustomTraining(onx_train, ['coefs', 'intercept'], verbose=1,
                         max_iter=10)
trainer.fit(X, y)
print(trainer.train_losses_)

df = DataFrame({"iteration": np.arange(len(trainer.train_losses_)),
                "loss": trainer.train_losses_})
df.set_index('iteration').plot(title="Training loss", logy=True)

######################################################
# Let's compare scikit-learn trained coefficients and the coefficients
# obtained with onnxruntime and check they are very close.

print("onnxruntime", trainer.trained_coef_)

