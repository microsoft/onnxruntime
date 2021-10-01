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

"""
import os
from pprint import pprint
import numpy as np
from pandas import DataFrame
from onnx import helper, numpy_helper, TensorProto, ModelProto
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
from onnxruntime import (
    InferenceSession, __version__ as ort_version,
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


###################################
# An equivalent ONNX graph.
# +++++++++++++++++++++++++
#
# This graph can be obtained with *sklearn-onnx` as we need to
# modify it for training, it is easier to create an explicit one.


def onnx_linear_regression(coefs, intercept):
    if len(coefs.shape) == 1:
        coefs = coefs.reshape((1, -1))
    coefs = coefs.T

    X = helper.make_tensor_value_info(
        'X', TensorProto.FLOAT, [None, coefs.shape[0]])
    Y = helper.make_tensor_value_info(
        'Y', TensorProto.FLOAT, [None, coefs.shape[1]])

    # inference
    node_matmul = helper.make_node('MatMul', ['X', 'coefs'], ['y1'], name='N1')
    node_add = helper.make_node('Add', ['y1', 'intercept'], ['Y'], name='N2')

    # initializer
    init_coefs = numpy_helper.from_array(coefs, name="coefs")
    init_intercept = numpy_helper.from_array(intercept, name="intercept")

    # graph
    graph_def = helper.make_graph(
        [node_matmul, node_add], 'lr', [X], [Y],
        [init_coefs, init_intercept])
    model_def = helper.make_model(
        graph_def, producer_name='orttrainer', ir_version=7,
        producer_version=ort_version,
        opset_imports=[helper.make_operatorsetid('', 14)])
    return model_def


onx = onnx_linear_regression(
    np.random.randn(2).astype(np.float32),
    np.random.randn(1).astype(np.float32))

########################################
# Let's visualize it.

def plot_dot(model):
    pydot_graph = GetPydotGraph(
        model.graph, name=model.graph.name, rankdir="TB",
        node_producer=GetOpNodeProducer("docstring"))
    return plot_graphviz(pydot_graph.to_string())
    
plot_dot(onx)


#####################################
# Training with onnxruntime-training
# ++++++++++++++++++++++++++++++++++
#
# It is possible only if the graph to train has a gradient.
# Then the model can be trained with a gradient descent algorithm.
# The previous graph only predicts, a new graph needs to be created
# to compute the loss as well. In our case, it is a square loss.
# The new graph requires another input for the label
# and another output for the loss value.


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

#######################################
# We create a graph with random coefficients.

onx_train = onnx_linear_regression_training(
    np.random.randn(2).astype(np.float32),
    np.random.randn(1).astype(np.float32))

plot_dot(onx_train)

################################################
# DataLoader
# ++++++++++
#
# Next class draws consecutive random observations from a dataset
# by batch.


class DataLoader:
    
    def __init__(self, X, y, batch_size=20):
        self.X, self.y = X, y
        self.batch_size = batch_size
        if len(self.y.shape) == 1:
            self.y = self.y.reshape((-1, 1))
        if self.X.shape[0] != self.y.shape[0]:
            raise VaueError(
                "Shape mismatch X.shape=%r, y.shape=%r." % (self.X.shape, self.y.shape))
    
    def __len__(self):
        return self.X.shape[0]
    
    def __iter__(self):
        N = 0
        b = len(self) - self.batch_size
        while N < len(self):
            i = np.random.randint(0, b)
            N += self.batch_size
            yield (self.X[i:i+self.batch_size], self.y[i:i+self.batch_size])

    @property
    def data(self):
        return self.X, self.y


data_loader = DataLoader(X_train, y_train, batch_size=2)


for i, batch in enumerate(data_loader):
    if i >= 2:
        break
    print("batch %r: %r" % (i, batch))


#########################################
# First iterations of training on GPU
# ++++++++++++++++++++++++++++++++++++
#
# Prediction needs an instance of class *InferenceSession*,
# the training needs an instance of class *TrainingSession*.
# Next function creates this one.


def create_training_session(training_onnx, weights_to_train, loss_output_name='loss',
                            training_optimizer_name='SGDOptimizer'):
    
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

    session = TrainingSession(
        training_onnx.SerializeToString(), ort_parameters, session_options)
    train_io_binding = session.io_binding()
    eval_io_binding = session.io_binding()
    return session


train_session = create_training_session(onx_train, ['coefs', 'intercept'])
print(train_session)

######################################
# Let's look into the expected inputs and outputs.

for i in train_session.get_inputs():
    print("+input: %s (%s%s)" % (i.name, i.type, i.shape))
for o in train_session.get_outputs():
    print("output: %s (%s%s)" % (o.name, o.type, o.shape))

######################################
# A third parameter `Learning_Rate` was added.
# The training updates the weight with a gradient multiplied
# by this parameter. Let's see now how to 
# retrieve the trained coefficients.

state_tensors = train_session.get_state()
pprint(state_tensors)

######################################
# We can now check the coefficients are updated after one iteration.

inputs = {'X': X_train[:1],
          'label': y_train[:1].reshape((-1, 1)),
          'Learning_Rate': np.array([0.001], dtype=np.float32)}

train_session.run(None, inputs)
state_tensors = train_session.get_state()
pprint(state_tensors)

######################################
# They changed. Another iteration to be sure.

inputs = {'X': X_train[:1],
          'label': y_train[:1].reshape((-1, 1)),
          'Learning_Rate': np.array([0.001], dtype=np.float32)}
res = train_session.run(None, inputs)
state_tensors = train_session.get_state()
pprint(state_tensors)

#####################################
# It works. The training loss can be obtained by looking into the results.

pprint(res)

######################################
# Training
# ++++++++
#
# We need to implement a gradient descent.
# Let's wrap this into a class similar following scikit-learn's API.


class CustomTraining:

    def __init__(self, model_onnx, weights_to_train, loss_output_name='loss',
                 max_iter=100, training_optimizer_name='SGDOptimizer', batch_size=10, 
                 eta0=0.01, alpha=0.0001, power_t=0.25, learning_rate='invscaling',
                 verbose=0):
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
            training_optimizer_name=self.training_optimizer_name)
        
        data_loader = DataLoader(X, y, batch_size=self.batch_size)
        lr = self._init_learning_rate()
        self.input_names_ = [i.name for i in self.train_session_.get_inputs()]
        self.output_names_ = [o.name for o in self.train_session_.get_outputs()]
        self.loss_index_ = self.output_names_.index(self.loss_output_name)
    
        loop = tqdm(range(self.max_iter)) if self.verbose else range(max_iter)
        train_losses = []
        for it in loop:
            loss = self._iteration(data_loader, lr)
            lr = self._update_learning_rate(it, lr)
            if self.verbose > 1:
                loop.set_description("loss=%1.3g lr=%1.3g" % (loss, lr))
            train_losses.append(loss)
        self.train_losses_ = train_losses
        self.trained_coef_ = self.train_session_.get_state()
        return self
        
    def _iteration(self, data_loader, learning_rate):
        actual_losses = []
        lr = np.array([learning_rate], dtype=np.float32)
        for batch_idx, (data, target) in enumerate(data_loader):
            if len(target.shape) == 1:
                target = target.reshape((-1, 1))
                
            inputs = {self.input_names_[0]: data,
                      self.input_names_[1]: target,
                      self.input_names_[2]: lr}
            res = self.train_session_.run(None, inputs)
            actual_losses.append(res[self.loss_index_])
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

####################################################
# It works. We could stop here or we could update the weights
# in the training model or the first model. That requires to
# update the constants in an ONNX graph. We could also compares
# the algorithm processing time to *scikit-learn* or *pytorch*.
#
# Update weights in an ONNX graph
# +++++++++++++++++++++++++++++++
#
# Let's first check the output of the first model in ONNX.

sess = InferenceSession(onx.SerializeToString())
before = sess.run(None, {'X': X[:5]})[0]
print(before)

#################################
# Let's replace the initializer.

def update_onnx_graph(model_onnx, new_weights):
    replace_weights = []
    replace_indices = []
    for i, w in enumerate(model_onnx.graph.initializer):
        if w.name in new_weights:
            replace_weights.append(numpy_helper.from_array(new_weights[w.name], w.name))
            replace_indices.append(i)
    replace_indices.sort(reverse=True)
    for w_i in replace_indices:
        del model_onnx.graph.initializer[w_i]
    model_onnx.graph.initializer.extend(replace_weights)    


update_onnx_graph(onx, trainer.trained_coef_)

########################################
# Let's compare with the previous output.

sess = InferenceSession(onx.SerializeToString())
after = sess.run(None, {'X': X[:5]})[0]
print(after)

######################################
# It looks almost the same but slighly different.

print(after - before)

################################################
# Next example will show how to train a linear regression on GPU:
# :ref:`l-orttraining-linreg-cpu`.
