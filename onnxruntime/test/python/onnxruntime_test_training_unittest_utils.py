import sys
import numpy as np
from onnx import numpy_helper

def get_node_index(model, node):
    i = 0
    while i < len(model.graph.node):
        if model.graph.node[i] == node:
            break
        i += 1
    return i if i < len(model.graph.node) else None

def add_const(model, name, output, t_value = None, f_value = None):
    const_node = model.graph.node.add()
    const_node.op_type = 'Constant'
    const_node.name = name
    const_node.output.extend([output])
    attr = const_node.attribute.add()
    attr.name = 'value'
    if t_value is not None:
        attr.type = 4
        attr.t.CopyFrom(t_value)
    else:
        attr.type = 1
        attr.f = f_value
    return const_node

def process_dropout(model):
    dropouts = []
    index = 0
    for node in model.graph.node:
        if node.op_type == 'Dropout':
            new_dropout = model.graph.node.add()
            new_dropout.op_type = 'TrainableDropout'
            new_dropout.name = 'TrainableDropout_%d' % index
            #make ratio node
            ratio = np.asarray([node.attribute[0].f], dtype=np.float32)
            print(ratio.shape)
            ratio_value = numpy_helper.from_array(ratio)
            ratio_node = add_const(model, 'dropout_node_ratio_%d' % index, 'dropout_node_ratio_%d' % index, t_value=ratio_value)
            print (ratio_node)
            new_dropout.input.extend([node.input[0], ratio_node.output[0]])
            new_dropout.output.extend(node.output)
            dropouts.append(get_node_index(model, node))
            index += 1
    dropouts.sort(reverse=True)
    for d in dropouts:
        del model.graph.node[d]
    model.opset_import[0].version = 10
