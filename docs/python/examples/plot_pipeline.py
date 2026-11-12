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

Retrieve a model in JSON format
+++++++++++++++++++++++++++++++

That's the most simple way.
"""

from onnxruntime.datasets import get_example

example1 = get_example("mul_1.onnx")

import onnx  # noqa: E402

model = onnx.load(example1)  # model is a ModelProto protobuf message

print(model)


#################################
# Draw a model with pydot
# ++++++++++++++++++++++
# We use *onnx* to load the model in a different way than before, then
# construct a pydot graph from its nodes and values.


from onnx import ModelProto  # noqa: E402

model = ModelProto()
with open(example1, "rb") as fid:
    content = fid.read()
    model.ParseFromString(content)

###################################
# We convert it into a graph.
import pydot  # noqa: E402

pydot_graph = pydot.Dot(model.graph.name, rankdir="LR")
value_nodes = {}
for index, node in enumerate(model.graph.node):
    node_name = node.name or f"{node.op_type}_{index}"
    op_node = pydot.Node(node_name, label=node.op_type, shape="box")
    pydot_graph.add_node(op_node)
    for value_name in node.input:
        value_node = value_nodes.setdefault(value_name, pydot.Node(value_name, shape="ellipse"))
        pydot_graph.add_node(value_node)
        pydot_graph.add_edge(pydot.Edge(value_node, op_node))
    for value_name in node.output:
        value_node = pydot.Node(value_name, shape="ellipse")
        value_nodes[value_name] = value_node
        pydot_graph.add_node(value_node)
        pydot_graph.add_edge(pydot.Edge(op_node, value_node))

pydot_graph.write_dot("graph.dot")

#######################################
# Then into an image
pydot_graph.write_png("graph.dot.png")

################################
# Which we display...
import matplotlib.pyplot as plt  # noqa: E402

image = plt.imread("graph.dot.png")
plt.imshow(image)
