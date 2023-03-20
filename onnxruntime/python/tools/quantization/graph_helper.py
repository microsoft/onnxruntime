from onnx import GraphProto, NodeProto

# These are helper functions that refactored from the ONNXModel class in quantization/onnx_model.py
# The ONNXModel class is not used in the fp16_converter.py script because it cannot be used for sub-graphs
# so we need to refactor these functions


def add_initializer(graph: GraphProto, tensor):
    if find_by_name(tensor.name, graph.initializer) is None:
        graph.initializer.extend([tensor])


def find_by_name(item_name, item_list):
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None


def remove_initializer(graph: GraphProto, initializer):
    if initializer in graph.initializer:
        graph.initializer.remove(initializer)
        for input in graph.input:
            if input.name == initializer.name:
                graph.input.remove(input)
                break


def get_initializer(graph: GraphProto, name):
    for tensor in graph.initializer:
        if tensor.name == name:
            return tensor
    return None


def is_graph_input(graph: GraphProto, tensor_name):
    for input in graph.input:
        if input.name == tensor_name:
            return True
    return False


def is_graph_output(graph: GraphProto, tensor_name):
    for output in graph.output:
        if output.name == tensor_name:
            return True
    return False


def output_name_to_node(graph: GraphProto):
    output_name_to_nodes = {}
    for node in graph.node:
        for output_name in node.output:
            output_name_to_nodes[output_name] = node
    return output_name_to_nodes


def add_node(graph: GraphProto, node):
    graph.node.extend([node])


def get_initializer_name_set(graph: GraphProto):
    return set(initializer.name for initializer in graph.initializer)


def get_non_initializer_inputs(graph: GraphProto):
    initializer_names = get_initializer_name_set(graph)
    non_initializer_inputs = set()
    for input in graph.input:
        if input.name not in initializer_names:
            non_initializer_inputs.add(input.name)
    return non_initializer_inputs
