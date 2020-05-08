import sys
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import numpy as np
from onnx import numpy_helper

def add_name(model):
    i = 0
    for node in model.graph.node:
       node.name = '%s_%d' %(node.op_type, i)
       i += 1

def find_input_node(model, arg):
    result = []
    for node in model.graph.node:
        for output in node.output:
            if output == arg:
                result.append(node)
    return result[0] if len(result)== 1 else None

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

def find_all_fused_nodes(model, concat_node, check_for_singleton_input=False):
    result = []
    singleton_input = None
    no_singleton_input = False
    candidate = [concat_node]
    while len(candidate) > 0:
        node = candidate[0]
        candidate.pop(0)
        result.append(node)
        if node.op_type == 'Shape':
            if check_for_singleton_input:
                if not singleton_input:
                    singleton_input = node.input[0]
                elif singleton_input != node.input[0]:
                    no_singleton_input = True
            continue
        for input in node.input:
            input_node = find_input_node(model, input)
            if check_for_singleton_input:
                # the input node shall only have one downstream node
                input_node_s_output_node = find_output_node(model, input_node.output[0])
                if not input_node_s_output_node:
                    continue
            if input_node is not None:
                candidate.append(input_node)

    if check_for_singleton_input:
        if no_singleton_input:
            return result, None
        else:
            return result, singleton_input
    else:
        return result

def get_node_index(model, node):
    i = 0
    while i < len(model.graph.node):
        if model.graph.node[i] == node:
            break;
        i += 1
    return i if i < len(model.graph.node) else None;

def add_shape(model, name, input, output):
    shape_node = model.graph.node.add()
    shape_node.op_type = 'Shape'
    shape_node.name = name
    shape_node.output.extend([output])
    shape_node.input.extend([input])
    return shape_node

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

# this step is to fuse node composition generated for:
#     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#     x = x.view(*new_x_shape)
def process_concat(model):
    new_reshape_nodes = {}
    new_expand_nodes = {}
    delete_nodes = []
    for node in model.graph.node:
        if node.op_type == 'Concat':
            reshape_or_expand_node = find_output_node(model, node.output[0])
            if reshape_or_expand_node.op_type == 'Reshape' or reshape_or_expand_node.op_type == 'Expand':
                input_nodes = []
                for input in node.input:
                    input_nodes.append(find_input_node(model, input))
                #figure out target shape
                shape = []
                for input_node in input_nodes:
                    assert input_node.op_type == 'Unsqueeze'
                    const_input = find_input_node(model, input_node.input[0])
                    if const_input.op_type != 'Constant':
                        shape.append(0)
                    else:
                        attr = const_input.attribute
                        assert len(attr) == 1
                        assert attr[0].name == 'value'
                        assert attr[0].type == 4
                        data = numpy_helper.to_array(attr[0].t)
                        shape.append(np.asscalar(data))
                # print('concat node: %s, new_shape is: %s' % (node.name, shape))
                #find out the nodes need to be deleted.
                if reshape_or_expand_node.op_type == 'Reshape':
                    fuse_nodes = find_all_fused_nodes(model, node, False)
                    new_reshape_nodes[get_node_index(model, reshape_or_expand_node)] = shape
                elif reshape_or_expand_node.op_type == 'Expand':
                    fuse_nodes, singleton_input = find_all_fused_nodes(model, node, True)
                    new_expand_nodes[get_node_index(model, reshape_or_expand_node)] = singleton_input

                for n in fuse_nodes:
                    delete_nodes.append(get_node_index(model, n))
    #insert new shape to reshape
    index = 0
    for reshape_node_index in new_reshape_nodes:
        shape_tensor = numpy_helper.from_array(np.asarray(new_reshape_nodes[reshape_node_index], dtype=np.int64))
        const_node = add_const(model, 'concat_shape_node_%d' % index, 'concat_shape_%d' % index, shape_tensor)
        index+=1
        reshape_node = model.graph.node[reshape_node_index]
        reshape_node.input[1] = const_node.output[0]

    for expand_node_index in new_expand_nodes:
        expand_node = model.graph.node[expand_node_index]
        singleton_input = new_expand_nodes[expand_node_index]
        # the shape node connects singleton_input to the second (index=1) input of the expand_node.
        shape_node = add_shape(model, 'concat_shape_node_%d' % index, singleton_input, expand_node.input[1])
        index+=1
        expand_node.input[1] = shape_node.output[0]

    #delete nodes
    delete_nodes.sort(reverse=True)
    for delete_node in delete_nodes:
        del model.graph.node[delete_node]

def add_cast(model, name, input, output, type):
    cast_node = model.graph.node.add()
    cast_node.name = name
    cast_node.op_type = 'Cast'
    attr = cast_node.attribute.add()
    attr.name = 'to'
    attr.type = 2
    attr.i = type
    cast_node.input.extend([input])
    cast_node.output.extend([output])
    return cast_node

# handle_expand_input_is_not_constant_case, fix_expand and add_expand_shape are
# to convert node composite create for pytorch arange op.
# Now Range is added to ONNX and pytorch will export arange using Range op.
# these 3 steps will be longer needed after Range is supported in ORT.
def handle_expand_input_is_not_constant_case(model):
    expand_node = None
    for node in model.graph.node:
        if node.op_type == 'Expand':
            expand_node = node
            break
    assert expand_node is not None
    expand_input = find_input_node(model, expand_node.input[0])
    if expand_input.op_type == 'Constant':
        return

    def trace_back_node(model, op):
        for node in model.graph.node:
            if node.output[0] == op.input[0]:
                return node

    def del_value_info(model, input):
        for i in range(len(model.graph.input)):
            if model.graph.input[i].name == input:
                del model.graph.input[i]
                break

        for i in range(len(model.graph.output)):
            if model.graph.output[i].name == input:
                del model.graph.output[i]
                break

    def del_value_node(model, prev_op):
        for i in range(len(model.graph.node)):
            if model.graph.node[i].name == prev_op.name:
                del model.graph.node[i]
                break

    # we assumes input to expand is arange data with range being dimension of a fixed axis of input
    prev_op = expand_input
    range_max = None
    while True:
        if prev_op.op_type == 'Shape':
            range_max = 512
            break

        del_value_info(model, prev_op.input[0])
        del_value_node(model, prev_op)

        prev_op = trace_back_node(model, prev_op)

    new_constant_node = model.graph.node.add()
    new_constant_node.CopyFrom(onnx.helper.make_node("Constant", [],
            [expand_input.output[0]], "contant_to_expand_", value=np.arange(0, 256)))


    # onnx.save(model, "/bert_ort/liqun/test_out/handle_expand_input_is_not_constant_case.onnx")
    # import pdb; pdb.set_trace()

# will be longer needed after Range is supported in ORT.
def fix_expand(model):
    #find expand node
    expand_node = None
    for node in model.graph.node:
        if node.op_type == 'Expand':
            expand_node = node
            break
    assert expand_node is not None
    const_expand_input = find_input_node(model, expand_node.input[0])
    assert const_expand_input.op_type == 'Constant'
    shape_node = find_input_node(model, expand_node.input[1])
    assert shape_node.op_type == 'Shape'
    #insert cast --> min --> cast
    cast_1 = add_cast(model, 'new_cast_01', shape_node.output[0], 'to_min_01', 1)
    min_target = numpy_helper.from_array(np.asarray([1, 9999], dtype=np.float32))
    min_target_node = add_const(model, 'op_min_node_10', 'op_min_ends_expand_10', min_target)
    min_node = model.graph.node.add()
    min_node.name = 'new_min_01'
    min_node.op_type = 'Min'
    min_node.input.extend([cast_1.output[0], min_target_node.output[0]])
    min_node.output.extend(['from_min_01'])
    cast_2 = add_cast(model, 'new_cast_02', min_node.output[0], 'to_slice_01', 7)
    #insert slice
    position = numpy_helper.from_array(np.expand_dims(np.arange(512, dtype=np.int64), axis=0))
    position_node = add_const(model, 'position_01_node', 'position_01', position)
    start_extend = numpy_helper.from_array(np.asarray([0, 0], dtype=np.int64), 'start_expand_10')
    start_extend_node = add_const(model, 'start_expand_10_node', 'start_expand_10', start_extend)
    axes = numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), 'axes_expand_10')
    axes_node = add_const(model, 'axes_expand_10_node', 'axes_expand_10', axes)
    slice_node = model.graph.node.add()
    slice_node.name = 'new_slice_01'
    slice_node.op_type = 'Slice'
    slice_node.input.extend([position_node.output[0], start_extend_node.output[0], cast_2.output[0], axes_node.output[0]])
    slice_node.output.extend(['from_slice_01'])
    #connect to expand
    expand_node.input[0] = slice_node.output[0]
    #delete the const input
    del model.graph.node[get_node_index(model, const_expand_input)]

def fix_dim(model):
    # TODO: bert ort_model has 5 real inputs, bert ort_trainer has 7 real inputs.
    # fix_dim makes the model graph cleaner. Not calling fix_dim does not break anything. 
    # So calling fix_dim in ort_model case works fine even it keeps 7 instead of 5 inputs.
    del model.graph.input[7:]

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
            # print(ratio.shape)
            ratio_value = numpy_helper.from_array(ratio)
            ratio_node = add_const(model, 'dropout_node_ratio_%d' % index, 'dropout_node_ratio_%d' % index, t_value=ratio_value)
            # print (ratio_node)
            new_dropout.input.extend([node.input[0], ratio_node.output[0]])
            new_dropout.output.extend(node.output)
            dropouts.append(get_node_index(model, node))
            index += 1
    dropouts.sort(reverse=True)
    for d in dropouts:
        del model.graph.node[d]

# will be longer needed after Range is supported in ORT.
# Also need to set following line differently for differnt verison of bert
# expand_out.name = '412'
def add_expand_shape(model):
    expand_node = [n for n in model.graph.node if n.op_type == 'Expand']
    if len(expand_node) != 1:
        raise "cannot find the single expand node in the BERT model."
        return
    expand_out = model.graph.value_info.add()
    expand_out.name = expand_node[0].output[0] # base: '421' # tiny: '85'
    expand_out.type.CopyFrom(model.graph.input[0].type)