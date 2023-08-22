import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference  # noqa: F401

if len(sys.argv) < 2:
    print("Please give model path...")
    exit(1)

input_model_name = sys.argv[1]
output_model_name = input_model_name[:-5] + "_optimized.onnx"

model = onnx.load(input_model_name)


def add_name(model):
    i = 0
    for node in model.graph.node:
        node.name = "%s_%d" % (node.op_type, i)
        i += 1


def find_input_node(model, arg):
    result = []
    for node in model.graph.node:
        for output in node.output:
            if output == arg:
                result.append(node)
    return result[0] if len(result) == 1 else None


def find_output_node(model, arg):
    result = []
    for node in model.graph.node:
        for input in node.input:
            if input == arg:
                result.append(node)
    return result[0] if len(result) == 1 else None


def find_input(model, arg):
    for initializer in model.graph.initializer:
        if initializer.name == arg:
            return initializer
    return None


def find_all_fused_nodes(model, concat_node):
    result = []
    candidate = [concat_node]
    while len(candidate) > 0:
        node = candidate[0]
        candidate.pop(0)
        result.append(node)
        if node.op_type == "Shape":
            continue
        for input in node.input:
            input_node = find_input_node(model, input)
            if input_node is not None:
                candidate.append(input_node)
    return result


def get_node_index(model, node):
    i = 0
    while i < len(model.graph.node):
        if model.graph.node[i] == node:
            break
        i += 1
    return i if i < len(model.graph.node) else None


def add_const(model, name, output, t_value=None, f_value=None):
    const_node = model.graph.node.add()
    const_node.op_type = "Constant"
    const_node.name = name
    const_node.output.extend([output])
    attr = const_node.attribute.add()
    attr.name = "value"
    if t_value is not None:
        attr.type = 4
        attr.t.CopyFrom(t_value)
    else:
        attr.type = 1
        attr.f = f_value
    return const_node


def process_concat(model):
    new_nodes = {}
    delete_nodes = []
    for node in model.graph.node:
        if node.op_type == "Concat":
            input_nodes = []
            for input in node.input:
                input_nodes.append(find_input_node(model, input))
            # figure out target shape
            shape = []
            for input_node in input_nodes:
                assert input_node.op_type == "Unsqueeze"
                const_input = find_input_node(model, input_node.input[0])
                if const_input.op_type != "Constant":
                    shape.append(0)
                else:
                    attr = const_input.attribute
                    assert len(attr) == 1
                    assert attr[0].name == "value"
                    assert attr[0].type == 4
                    data = numpy_helper.to_array(attr[0].t)
                    shape.append(np.asscalar(data))
            print(f"concat node: {node.name}, new_shape is: {shape}")
            # find out the nodes need to be deleted.
            fuse_nodes = find_all_fused_nodes(model, node)
            reshape_node = find_output_node(model, node.output[0])
            assert reshape_node.op_type == "Reshape"
            new_nodes[get_node_index(model, reshape_node)] = shape
            for n in fuse_nodes:
                delete_nodes.append(get_node_index(model, n))
    # insert new shape to reshape
    index = 0
    for reshape_node_index in new_nodes:
        shape_tensor = numpy_helper.from_array(np.asarray(new_nodes[reshape_node_index], dtype=np.int64))
        const_node = add_const(model, "concat_shape_node_%d" % index, "concat_shape_%d" % index, shape_tensor)
        index += 1
        reshape_node = model.graph.node[reshape_node_index]
        reshape_node.input[1] = const_node.output[0]
    # delete nodes
    delete_nodes.sort(reverse=True)
    for delete_node in delete_nodes:
        del model.graph.node[delete_node]


def add_cast(model, name, input, output, type):
    cast_node = model.graph.node.add()
    cast_node.name = name
    cast_node.op_type = "Cast"
    attr = cast_node.attribute.add()
    attr.name = "to"
    attr.type = 2
    attr.i = type
    cast_node.input.extend([input])
    cast_node.output.extend([output])
    return cast_node


def fix_expand(model):
    # find expand node
    expand_node = None
    for node in model.graph.node:
        if node.op_type == "Expand":
            expand_node = node
            break
    assert expand_node is not None
    const_expand_input = find_input_node(model, expand_node.input[0])
    assert const_expand_input.op_type == "Constant"
    shape_node = find_input_node(model, expand_node.input[1])
    assert shape_node.op_type == "Shape"
    # insert cast --> min --> cast
    cast_1 = add_cast(model, "new_cast_01", shape_node.output[0], "to_min_01", 1)
    min_target = numpy_helper.from_array(np.asarray([1, 9999], dtype=np.float32))
    min_target_node = add_const(model, "op_min_node_10", "op_min_ends_expand_10", min_target)
    min_node = model.graph.node.add()
    min_node.name = "new_min_01"
    min_node.op_type = "Min"
    min_node.input.extend([cast_1.output[0], min_target_node.output[0]])
    min_node.output.extend(["from_min_01"])
    cast_2 = add_cast(model, "new_cast_02", min_node.output[0], "to_slice_01", 7)
    # insert slice
    position = numpy_helper.from_array(np.expand_dims(np.arange(512, dtype=np.int64), axis=0))
    position_node = add_const(model, "position_01_node", "position_01", position)
    start_extend = numpy_helper.from_array(np.asarray([0, 0], dtype=np.int64), "start_expand_10")
    start_extend_node = add_const(model, "start_expand_10_node", "start_expand_10", start_extend)
    axes = numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), "axes_expand_10")
    axes_node = add_const(model, "axes_expand_10_node", "axes_expand_10", axes)
    slice_node = model.graph.node.add()
    slice_node.name = "new_slice_01"
    slice_node.op_type = "Slice"
    slice_node.input.extend(
        [position_node.output[0], start_extend_node.output[0], cast_2.output[0], axes_node.output[0]]
    )
    slice_node.output.extend(["from_slice_01"])
    # connect to expand
    expand_node.input[0] = slice_node.output[0]
    # delete the const input
    del model.graph.node[get_node_index(model, const_expand_input)]


def fix_dim(model):
    del model.graph.input[3:]


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
        if node.op_type == "Transpose":
            weight = find_input(model, node.input[0])
            if weight is not None:
                result = []
                for n in model.graph.node:
                    for input in n.input:
                        if input == weight.name:
                            result.append(n)
                if len(result) > 1:
                    continue
                perm = node.attribute[0]
                assert perm.name == "perm"
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
        if node.op_type == "Dropout":
            new_dropout = model.graph.node.add()
            new_dropout.op_type = "TrainableDropout"
            new_dropout.name = "TrainableDropout_%d" % index
            # make ratio node
            ratio = np.asarray([node.attribute[0].f], dtype=np.float32)
            print(ratio.shape)
            ratio_value = numpy_helper.from_array(ratio)
            ratio_node = add_const(
                model, "dropout_node_ratio_%d" % index, "dropout_node_ratio_%d" % index, t_value=ratio_value
            )
            print(ratio_node)
            new_dropout.input.extend([node.input[0], ratio_node.output[0]])
            new_dropout.output.extend(node.output)
            dropouts.append(get_node_index(model, node))
            index += 1
    dropouts.sort(reverse=True)
    for d in dropouts:
        del model.graph.node[d]


# Also need to set following line differently for differnt verison of bert
# expand_out.name = '412'
def add_expand_shape(model):
    expand_out = model.graph.value_info.add()
    expand_out.name = "74"  #'410' # 74 for base model
    expand_out.type.CopyFrom(model.graph.input[0].type)


# add name to nodes
add_name(model)
# replace garther&concat to reshape
process_concat(model)
# fix the expand with dynamic shape
fix_expand(model)
# use dynamic batch/sequence
fix_dim(model)
# constant fold transpose
fix_transpose(model)
# replace dropout with trainable dropout
process_dropout(model)
# add output shape of expand
add_expand_shape(model)
# set opset version to 10
model.opset_import[0].version = 10

f = open(output_model_name, "wb")  # noqa: SIM115
f.write(model.SerializeToString())
f.close()

# Use ORT to verify the converted model. Notice that you must use python package from the
# training branch because training requires some extra ops.
import onnxruntime as ort  # noqa: E402

# We convert model to accept variable-length batch size, so it can be any positive integer.
batch = 3
# This should match --max_seq_length when calling nv_run_pretraining.py.
sq_length = 512
# This should match vocab_size in bert_config.json in DeepLearningExamples/PyTorch/LanguageModeling/BERT.
vocab_size = 30528

# Create a fake data point.
input_ids = np.random.randint(low=0, high=vocab_size, size=(batch, sq_length), dtype=np.int64)
segment_ids = np.random.randint(low=0, high=2, size=(batch, sq_length), dtype=np.int64)
input_mask = np.ones((batch, sq_length), dtype=np.int64)

# Do forward using the original model.
sess = ort.InferenceSession(input_model_name, providers=ort.get_available_providers())
result = sess.run(None, {"input1": input_ids, "input2": segment_ids, "input3": input_mask})

# Do forward using the new model.
new_sess = ort.InferenceSession(output_model_name, providers=ort.get_available_providers())
new_result = new_sess.run(None, {"input1": input_ids, "input2": segment_ids, "input3": input_mask})

# Compare the outcomes from the two models.
print(np.linalg.norm(result[0] - new_result[0]))
print(np.linalg.norm(result[1] - new_result[1]))
