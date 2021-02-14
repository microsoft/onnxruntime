import onnx
import copy
from onnx import shape_inference
from onnxruntime.capi import _pybind_state as C


def print_list(name, value):
    print(name + ':', ', '.join(value))


def dim_str(dim):
    if dim.HasField('dim_value'):
        return str(dim.dim_value)
    elif dim.HasField('dim_param'):
        return dim.dim_param
    return 'n/a'

def print_type(name, type):
    print('[' + name + ']', 'type:', type.tensor_type.elem_type, '| size:', '[' + ','.join([dim_str(d) for d in type.tensor_type.shape.dim]) + ']')


"""
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
"""

#BERT with loss
original_model = onnx.load('bert-tiny-loss.onnx')
config = C.ModuleGradientGraphBuilderConfiguration()
initializer_names_to_train = []
for initializer in original_model.graph.initializer:
    if initializer.name.startswith('bert.') or initializer.name.startswith('cls.'):
        initializer_names_to_train.append(initializer.name)
config.initializer_names_to_train = initializer_names_to_train
input_names_require_grad = []
input_names_require_grad.append('input3')
config.input_names_require_grad = input_names_require_grad

module_gradient_graph_builder = C.ModuleGradientGraphBuilder()
module_gradient_graph_builder.build_and_split(original_model.SerializeToString(), config)

forward_model = onnx.load_model_from_string(module_gradient_graph_builder.get_forward_model())
backward_model = onnx.load_model_from_string(module_gradient_graph_builder.get_backward_model())
onnx.save(onnx.load_model_from_string(module_gradient_graph_builder.get_gradient_model()), 'bert_gradient_graph.onnx')
onnx.save(forward_model, 'bert_forward.onnx')
onnx.save(backward_model, 'bert_backward.onnx')

split_graphs_info = module_gradient_graph_builder.get_split_graphs_info()
print_list('user_input_names', split_graphs_info.user_input_names)
print_list('initializer_names_to_train', split_graphs_info.initializer_names_to_train)
print_list('user_output_names', split_graphs_info.user_output_names)
print_list('backward_user_input_names', split_graphs_info.backward_user_input_names)
print_list('backward_intializer_names_as_input', split_graphs_info.backward_intializer_names_as_input)
print_list('intermediate_tensor_names', split_graphs_info.intermediate_tensor_names)
print_list('user_output_grad_names', split_graphs_info.user_output_grad_names)
print_list('backward_output_grad_names', split_graphs_info.backward_output_grad_names)

type_map = {}
for name in split_graphs_info.user_input_names:
    type_map[name] = None
for name in split_graphs_info.initializer_names_to_train:
    type_map[name] = None
for name in split_graphs_info.user_output_names:
    type_map[name] = None
for name in split_graphs_info.backward_user_input_names:
    type_map[name] = None
for name in split_graphs_info.backward_intializer_names_as_input:
    type_map[name] = None
for name in split_graphs_info.intermediate_tensor_names:
    type_map[name] = None
for name in split_graphs_info.user_output_grad_names:
    type_map[name] = None
for name in split_graphs_info.backward_output_grad_names:
    type_map[name] = None

for input in forward_model.graph.input:
    if input.name in type_map and type_map[input.name] is None:
        type_map[input.name] = input.type

for output in forward_model.graph.output:
    if output.name in type_map and type_map[output.name] is None:
        type_map[output.name] = output.type
    output_grad_name = output.name + '_grad'
    if output_grad_name in type_map and type_map[output_grad_name] is None:
        type_map[output_grad_name] = output.type

for key, value in type_map.items():
    print_type(key, value)
