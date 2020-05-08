from onnx import numpy_helper

def add_name(model):
    i = 0
    for node in model.graph.node:
       node.name = '%s_%d' %(node.op_type, i)
       i += 1

def find_output_node(model, arg):
    result = []
    for node in model.graph.node:
        for input in node.input:
            if input == arg:
                result.append(node)
    return result[0] if len(result) == 1 else None

def find_input_as_initializer(model, arg):
    for initializer in model.graph.initializer:
        if initializer.name == arg:
            return initializer
    return None

def get_node_index(model, node):
    i = 0
    while i < len(model.graph.node):
        if model.graph.node[i] == node:
            break;
        i += 1
    return i if i < len(model.graph.node) else None;

def replace_input_arg(model, arg, new_arg):
    for node in model.graph.node:
        i = 0
        while i < len(node.input):
            if node.input[i] == arg:
                node.input[i] = new_arg
            i += 1

def find_weight_index(model, name):
    index = 0
    for w in model.graph.initializer:
        if w.name == name:
            return index
        index += 1
    return None

def fix_transpose(model):
    transpose = []
    for node in model.graph.node:
        if node.op_type == 'Transpose':
            weight = find_input_as_initializer(model, node.input[0])
            if weight is not None:
                result = []
                for n in model.graph.node:
                    for input in n.input:
                        if input == weight.name:
                            result.append(n)
                if len(result) > 1:
                    continue
                perm = node.attribute[0]
                assert perm.name == 'perm'
                perm = perm.ints
                assert len(perm) == 2 and perm[0] == 1 and perm[1] == 0
                transpose.append((get_node_index(model, node), weight))
    for t in transpose:
        node = model.graph.node[t[0]]
        weight = numpy_helper.to_array(t[1])
        assert len(weight.shape) == 2
        weight = weight.transpose(perm)
        new_weight = numpy_helper.from_array(weight, "%s_transposed" % t[1].name)
        model.graph.initializer.extend([new_weight])
        replace_input_arg(model, node.output[0], new_weight.name)

    transpose.sort(reverse=True)
    for t in transpose:
        del model.graph.node[t[0]]

    old_ws = []
    for t in transpose:
        if find_output_node(model, t[1].name) is None:
            old_ws.append(find_weight_index(model, t[1].name))
    old_ws.sort(reverse=True)
    for w_i in old_ws:
        del model.graph.initializer[w_i]

def add_expand_shape(model):
    expand_node = [n for n in model.graph.node if n.op_type == 'Expand']
    if len(expand_node) != 1:
        raise "cannot find the single expand node in the BERT model."
        return
    expand_out = model.graph.value_info.add()
    expand_out.name = expand_node[0].output[0] # base: '421' # tiny: '85'
    expand_out.type.CopyFrom(model.graph.input[0].type)