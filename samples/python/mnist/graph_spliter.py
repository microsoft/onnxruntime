import onnx
import copy
from onnx import shape_inference
from onnxruntime.capi import _pybind_state as C

def add_input_from_initializer(model, initializer, docstring=None):
    new_input = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims, docstring)
    model.graph.input.append(new_input)

def add_input(model, name, data_type = None, dims = None, docstring = None):
    new_input = onnx.helper.make_tensor_value_info(name, data_type, dims, docstring)
    model.graph.input.append(new_input)

def add_output(model, name, data_type = None, docstring = None):
    new_output = model.graph.value_info.add()
    new_output.name = name
    if data_type:
        new_output.type.CopyFrom(data_type)
    if docstring:
        new_output.doc_string = docstring
    model.graph.output.append(new_output)

def remove_nodes(onnx_model, nodes_to_remove):
    all_nodes = []
    for node in onnx_model.graph.node:
        if node not in nodes_to_remove:
            all_nodes.append(node)
    
    onnx_model.graph.ClearField('node')
    onnx_model.graph.node.extend(all_nodes)

def split_graph(onnx_model):
    forward_graph_outputs = set()
    backward_graph_inputs = set()
    backward_graph_outputs = set()
    # Get forward graph
    forward_model = copy.deepcopy(onnx_model)
    nodes_to_remove_from_forward_graph = []
    initializers = {}
    for initializer in forward_model.graph.initializer:
        initializers[initializer.name] = initializer
    forward_graph_initializer_names = set()
    for node in forward_model.graph.node:
        if node.doc_string == 'Backward pass':
            # nodes belongs to backward graph
            nodes_to_remove_from_forward_graph.append(node)
            for input in node.input:
                backward_graph_inputs.add(input)
            for output in node.output:
                backward_graph_outputs.add(output)
        else:
            # nodes belogs to forward graph
            for input in node.input:
                if input in initializers:
                    forward_graph_initializer_names.add(input)
            for output in node.output:
                forward_graph_outputs.add(output)

    forward_model.graph.ClearField('initializer')
    for initializer_name in forward_graph_initializer_names:
        forward_model.graph.initializer.append(initializers[initializer_name])

    # outputs from forward graph that are also inputs of backwoard graph need to be added as graph output.
    for output in forward_graph_outputs:
        if output in backward_graph_inputs:
            add_output(forward_model, output)

    remove_nodes(forward_model, nodes_to_remove_from_forward_graph)

    # Get backward graph
    tensor_elem_types = {}
    infered_model = shape_inference.infer_shapes(onnx_model)
    for value_info in infered_model.graph.value_info:
        tensor_elem_types[value_info.name] = value_info.type.tensor_type.elem_type

    backward_model = copy.deepcopy(onnx_model)
    initializers = {}
    for initializer in backward_model.graph.initializer:
        initializers[initializer.name] = initializer

    nodes_to_remove_from_backward_graph = []
    for node in backward_model.graph.node:
        if node.doc_string != 'Backward pass':
            nodes_to_remove_from_backward_graph.append(node)

    # gradient of forward graph output will be the input of backward graph
    for output in backward_model.graph.output:
        if output.name + '_grad' in backward_graph_inputs:
            add_input(backward_model, output.name + '_grad', output.type.tensor_type.elem_type)

    backward_graph_initializer_names = set()
    for input in backward_graph_inputs:
        if input in forward_graph_outputs:
            # inputs of backward graph that are also outputs from forward graph need to be added to backward graph input
            add_input(backward_model, input, tensor_elem_types[input] if input in tensor_elem_types else 1)
        elif input in forward_graph_initializer_names:
            # inputs from forward graph initializers need to be added to backward graph input
            add_input_from_initializer(backward_model, initializers[input])
        elif input in initializers:
            backward_graph_initializer_names.add(input)

    backward_model.graph.ClearField('initializer')
    for initializer_name in backward_graph_initializer_names:
        backward_model.graph.initializer.append(initializers[initializer_name])

    # add gradient output to backward graph output
    # TODO: need to add gradient of graph input to backward graph output
    new_backward_graph_outputs = set()
    for output in backward_graph_outputs:
        if output.endswith('_grad') and output[:-5] in forward_graph_initializer_names:
            new_backward_graph_outputs.add(output)
    
    backward_model.graph.ClearField('output')
    for output in new_backward_graph_outputs:
        add_output(backward_model, output)

    remove_nodes(backward_model, nodes_to_remove_from_backward_graph)

    return forward_model, backward_model


# MNIST
original_model = onnx.load('mnist_original.onnx')
config = C.ModuleGradientGraphBuilderConfiguration()
weight_names_to_train = set()
for initializer in original_model.graph.initializer:
    weight_names_to_train.add(initializer.name)
config.weight_names_to_train = weight_names_to_train
output_names = set()
for output in original_model.graph.output:
    output_names.add(output.name)
config.output_names = output_names

models = [onnx.load_model_from_string(model_as_string) for model_as_string in C.ModuleGradientGraphBuilder().build_and_split(original_model.SerializeToString(), config)]
onnx.save(models[0], 'minst_gradient_graph.onnx')
onnx.save(models[1], 'mnist_forward.onnx')
onnx.save(models[2], 'mnist_backward.onnx')


"""
#BERT
original_model = onnx.load('BertForSequenceClassification_full_training.onnx')
config = C.ModuleGradientGraphBuilderConfiguration()
weight_names_to_train = set()
for initializer in original_model.graph.initializer:
    weight_names_to_train.add(initializer.name)
config.weight_names_to_train = weight_names_to_train
output_names = set()
for output in original_model.graph.output:
    output_names.add(output.name)
config.output_names = output_names

models = [onnx.load_model_from_string(model_as_string) for model_as_string in C.ModuleGradientGraphBuilder().build_and_split(original_model.SerializeToString(), config)]
onnx.save(models[0], 'bert_gradient_graph.onnx')
onnx.save(models[1], 'bert_forward.onnx')
onnx.save(models[2], 'bert_backward.onnx')


#BERT with loss
original_model = onnx.load('bert-tiny-loss.onnx')
config = C.ModuleGradientGraphBuilderConfiguration()
weight_names_to_train = set()
for initializer in original_model.graph.initializer:
    if initializer.name.startswith('bert.') or initializer.name.startswith('cls.'):
        weight_names_to_train.add(initializer.name)
config.weight_names_to_train = weight_names_to_train
output_names = set()
output_names.add('total_loss')
#for output in original_model.graph.output:
#    output_names.add(output.name)
config.output_names = output_names

models = [onnx.load_model_from_string(model_as_string) for model_as_string in C.ModuleGradientGraphBuilder().build_and_split(original_model.SerializeToString(), config)]
onnx.save(models[0], 'bert_gradient_graph.onnx')
onnx.save(models[1], 'bert_forward.onnx')
onnx.save(models[2], 'bert_backward.onnx')
"""
