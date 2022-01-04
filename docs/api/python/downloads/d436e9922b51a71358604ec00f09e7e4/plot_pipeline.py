# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Draw a pipeline
===============

There is no other way to look into one model stored
in ONNX format than looking into its node with 
*onnx*. This example demonstrates
how to draw a model and to retrieve it in *json*
format.

.. contents::
    :local:

Retrieve a model in JSON format
+++++++++++++++++++++++++++++++

That's the most simple way.
"""

from onnxruntime.datasets import get_example
example1 = get_example("mul_1.onnx")

import onnx
model = onnx.load(example1)  # model is a ModelProto protobuf message

print(model) 


#################################
# Draw a model with ONNX
# ++++++++++++++++++++++
# We use `net_drawer.py <https://github.com/onnx/onnx/blob/master/onnx/tools/net_drawer.py>`_
# included in *onnx* package.
# We use *onnx* to load the model
# in a different way than before.


from onnx import ModelProto
model = ModelProto()
with open(example1, 'rb') as fid:
    content = fid.read()
    model.ParseFromString(content)

###################################
# We convert it into a graph.
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
pydot_graph = GetPydotGraph(model.graph, name=model.graph.name, rankdir="LR",
                            node_producer=GetOpNodeProducer("docstring"))
pydot_graph.write_dot("graph.dot")

#######################################
# Then into an image
import os
os.system('dot -O -Tpng graph.dot')

################################
# Which we display...
import matplotlib.pyplot as plt
image = plt.imread("graph.dot.png")
plt.imshow(image)






