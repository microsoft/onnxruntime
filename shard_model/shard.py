import onnx
from enum import Enum
from onnx import helper
from onnx import numpy_helper
import numpy as np

class ShardSpec:
    def __init__(self, shard_spec, is_partial):
        self.spec = shard_spec
        self.is_partial = is_partial

    def is_valid_sharding(self, shape):
        if self.is_partial:
            return True
        if len(self.spec) != len(shape):
            return False
        
        for shard, dim in zip(self.spec, shape):
            if dim % shard != 0:
                return False
        return True
    
    def num_shard(self):
        return np.prod(self.spec)
    
    def get_shard_dims(self, shape):
        return [int(dim / shard) for shard, dim in zip(self.spec, shape)]

class Actions(Enum):
    NoAction = 0
    AllGather = 1
    AllReduce = 2

class ShardedOnnxOp:
    def __init__(self, op_type, input_shapes, output_shapes):
        self.op_type = op_type
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
    
    def infer_sharding(self, input_shard_specs):
        actions = []
        for shard, shape in zip(input_shard_specs, self.input_shapes):
            if not shard.is_valid_sharding(shape):
                raise Exception(f'invalid sharding on shape {shape}')
            if shard.is_partial:
                actions.append(Actions.AllReduce)
            elif shard.num_shard() > 1:
                actions.append(Actions.AllGather)
            else:
                actions.append(Actions.NoAction)
        return actions, [ShardSpec([int(1)] * len(s), False) for s in self.output_shapes]

class ReductionOnnxOp(ShardedOnnxOp):
    def __init__(self, op_type, input_shapes, output_shapes, reduction_dims):
        super(ReductionOnnxOp, self).__init__(op_type, input_shapes, output_shapes)
        self.reduction_dims = reduction_dims
    
    def infer_sharding(self, input_shard_specs):
        actions, input_sharded_shape, partial_result = self.infer_actions(input_shard_specs)
        output_shard_spec = self.infer_output_shard_spec(input_sharded_shape, partial_result)
        return actions, output_shard_spec
    
    def same_shard(self, sharded_inputs, input_shard_shape):
        if len(sharded_inputs) == len(self.input_shapes):
            if len(sharded_inputs) == 0:
                return True
            x = input_shard_shape[sharded_inputs[0][0]][sharded_inputs[0][1]]
            for _ in sharded_inputs:
                if input_shard_shape[_[0]][_[1]] != x:
                    return False
            return True
        return False
    
    def infer_actions(self, input_shard_specs):
        actions = []
        input_sharded_shape = []
        for shard, shape in zip(input_shard_specs, self.input_shapes):
            if not shard.is_valid_sharding(shape):
                raise Exception(f'invalid sharding on shape {shape}')
            if shard.is_partial:
                actions.append(Actions.AllReduce)
                input_sharded_shape.append(shape)
            else:
                actions.append(Actions.NoAction)
                input_sharded_shape.append(shard.get_shard_dims(shape))
        
        partial_result = False
        for reduce_dim in self.reduction_dims:
            sharded_inputs = []
            for i, dim in enumerate(reduce_dim):
                if dim >= 0 and input_sharded_shape[i][dim] != self.input_shapes[i][dim]:
                    sharded_inputs.append((i, dim))
            if not self.same_shard(sharded_inputs, input_sharded_shape):
                for (i, dim) in sharded_inputs:
                    actions[i] = Actions.AllGather
                    input_sharded_shape[i] = self.input_shapes[i]
            elif len(sharded_inputs) > 0:
                partial_result = True

        return actions, input_sharded_shape, partial_result
    
    def infer_output_shard_spec(self, input_sharded_shape, partial_result):
        raise Exception("Unexpected")
    
class GemmOp(ReductionOnnxOp):
    def infer_output_shard_spec(self, input_sharded_shape, partial_result):
        # todo: transpose
        assert len(self.output_shapes) == 1
        spec = [int(self.output_shapes[0][0] / input_sharded_shape[0][0]),
                int(self.output_shapes[0][1] / input_sharded_shape[1][1])]
        return [ShardSpec(spec, partial_result)]

class MatmulOp(ReductionOnnxOp):
    def infer_output_shard_spec(self, input_sharded_shape, partial_result):
        assert len(self.output_shapes) == 1
        rank = len(self.output_shapes[0])
        i = 0
        spec = []
        while i < (rank - 1):
            spec.append(int(self.output_shapes[0][i] / input_sharded_shape[0][i]))
            i += 1
        spec.append(int(self.output_shapes[0][i] / input_sharded_shape[1][1]))
        return [ShardSpec(spec, partial_result)]

class AttentionOp(ReductionOnnxOp):
    def infer_output_shard_spec(self, input_sharded_shape, partial_result):
        assert len(self.output_shapes) == 1
        assert input_sharded_shape[1][1] == input_sharded_shape[2][0]
        spec = [int(self.output_shapes[0][0] / input_sharded_shape[0][0]),
                int(self.output_shapes[0][1] / input_sharded_shape[0][1]),
                int(self.output_shapes[0][2] / int(input_sharded_shape[1][1] / 3))]
        return [ShardSpec(spec, partial_result)]

class ElementwiseOp(ShardedOnnxOp):
    def __init__(self, op_type, input_shapes, output_shapes, broad_cast_dims):
        super(ElementwiseOp, self).__init__(op_type, input_shapes, output_shapes)
        self.broad_cast_dims = broad_cast_dims

    def infer_sharding(self, input_shard_specs):
        rank = len(self.output_shapes)
        assert rank == 1
        actions = [Actions.NoAction] * len(self.input_shapes)
        has_partial_input = False
        has_sharded_input = False
        for i, input_spec in enumerate(input_shard_specs):
            if input_spec.is_partial:
                # TODO: support partial tensor
                action[i] = Actions.AllReduce
                has_partial_input = True
            elif input_spec.num_shard() > 1:
                has_sharded_input = True
        if has_partial_input and has_sharded_input:
            raise Exception('Has both sharded tensor and partial tensor in element-wise op is not supported yet.')
        if has_partial_input:
            return actions, [ShardSpec([int(1)] * len(s), False) for s in self.output_shapes]
        # todo: improve it later
        output_shards = [1] * len(self.output_shapes[0])
        valid_shard = True
        for (i, axis) in self.broad_cast_dims:
            input_shards_on_axis = [s.spec[_] for s, _ in zip(input_shard_specs, axis)]
            if input_shards_on_axis.count(input_shards_on_axis[0]) != len(input_shards_on_axis):
                valid_shard = False
        if valid_shard:
            return actions, [ShardSpec(input_shard_specs[0].spec, False)]
        else:
            for i, input_spec in enumerate(input_shard_specs):
                if input_spec.num_shard() > 1:
                    action[i] = Actions.AllGather
            return actions, [ShardSpec([int(1)] * len(s), False) for s in self.output_shapes]

def lookup_arg(graph, arg_name):
    for value_info in model.graph.value_info:
        if value_info.name == arg_name:
            return value_info
    for value_info in model.graph.input:
        if value_info.name == arg_name:
            return value_info
    for value_info in model.graph.output:
        if value_info.name == arg_name:
            return value_info
    return None

def lookup_arg_shape(graph, arg_name):
    arg = lookup_arg(graph, arg_name)
    if arg is None:
        #look in the initializers
        w = [_ for _ in graph.initializer if _.name == arg_name]
        assert len(w) == 1
        return [_ for _ in w[0].dims]
    return [_.dim_value for _ in arg.type.tensor_type.shape.dim]

def get_shard_op(node, graph):
    if node.op_type == "Gemm":
        return GemmOp(node.op_type, 
                      [lookup_arg_shape(graph, input) for input in node.input],
                      [lookup_arg_shape(graph, output) for output in node.output],
                      ((1, 0),))
    elif node.op_type == "MatMul":
        return MatmulOp(node.op_type, 
                      [lookup_arg_shape(graph, input) for input in node.input],
                      [lookup_arg_shape(graph, output) for output in node.output],
                      ((len(lookup_arg_shape(graph, node.input[0])) - 1, 0),))
    elif node.op_type == "Attention":
        return AttentionOp(node.op_type,
                      [lookup_arg_shape(graph, input) for input in node.input],
                      [lookup_arg_shape(graph, output) for output in node.output],
                      ((2, 0, -1, -1),))
    elif node.op_type == "FastGelu":
        return ElementwiseOp(node.op_type,
                      [lookup_arg_shape(graph, input) for input in node.input],
                      [lookup_arg_shape(graph, output) for output in node.output],
                      ((2, (2, 0),),))
    return ShardedOnnxOp(node.op_type, 
                      [lookup_arg_shape(graph, input) for input in node.input],
                      [lookup_arg_shape(graph, output) for output in node.output])

def shard_model(model, shard_specs, output_name, rank):
    graph = model.graph
    shard_dict = shard_specs.copy()
    for graph_input in graph.input:
        shard_dict[graph_input.name] = ShardSpec([int(1)] * len(lookup_arg_shape(graph, graph_input.name)), False)
    
    for w in graph.initializer:
        if w.name not in shard_dict:
            shard_dict[w.name] = ShardSpec([int(1)] * len(lookup_arg_shape(graph, w.name)), False)
    
    collectives_actions = {}

    for node in graph.node:
        shard_op = get_shard_op(node, graph)
        input_shard_spec = [shard_dict[_] for _ in node.input]
        actions, output_shard_spec = shard_op.infer_sharding(input_shard_spec)
        for name, spec in zip (node.output, output_shard_spec,):
            shard_dict[name] = spec
        for name, action in zip(node.input, actions):
            if action is not Actions.NoAction:
                collectives_actions[name] = action
    
    # insert collective ops:
    collective_args_map = {}
    coolective_nodes = []
    for arg_name in collectives_actions:
        action = collectives_actions[arg_name]
        if action == Actions.AllReduce:
            arg = lookup_arg(graph, arg_name)
            assert arg
            new_arg = helper.make_tensor_value_info(f'{arg_name}_all_reduce', arg.type.tensor_type.elem_type, 
                                                    [_.dim_value for _ in arg.type.tensor_type.shape.dim])
            allreduce_node = helper.make_node(
                            "NcclAllReduce",
                            [arg_name],
                            [new_arg.name],
                            domain = "com.microsoft",
                            name=""
                            )
            coolective_nodes.append(allreduce_node)
            collective_args_map[arg_name] = new_arg.name
        elif action == Actions.AllGather:
            arg = lookup_arg(graph, arg_name)
            assert arg
            # this is not correct, temporary use it to demostrate.
            new_arg = helper.make_tensor_value_info(f'{arg_name}_all_reduce', arg.type.tensor_type.elem_type, 
                                                    [_.dim_value for _ in arg.type.tensor_type.shape.dim])
            allreduce_node = helper.make_node(
                            "NcclAllGather",
                            [arg_name],
                            [new_arg.name],
                            domain = "com.microsoft",
                            name=""
                            )
            coolective_nodes.append(allreduce_node)
            collective_args_map[arg_name] = new_arg.name
        else:
            raise Exception("Not Implemented.")
    
    for node in graph.node:
        new_inputs = [collective_args_map[_] if _ in collective_args_map else _ for _ in node.input]
        del node.input[:]
        node.input.extend(new_inputs)
    graph.node.extend(coolective_nodes)
    # clean all the shapes
    for value_info in graph.value_info:
        if value_info.name not in [_.name for _ in graph.input] and value_info.name not in [_.name for _ in graph.initializer]:
            del value_info.type.tensor_type.shape.dim[:]

    # shard tensors
    original_initializers = []
    original_initializers.extend(graph.initializer)
    del graph.initializer[:]

    opset=model.opset_import.add()
    opset.domain="com.microsoft"
    opset.version=1

    for _ in range(rank):
        for i, w in enumerate(original_initializers):
            shard_spec = shard_dict[w.name]
            if shard_spec.is_partial:
                raise Exception("Unexpected.")
            if shard_spec.num_shard() > 1:
                assert shard_spec.num_shard() == rank
                w_array = numpy_helper.to_array(w)
                slc = [slice(None)] * len(shard_spec.spec)
                for j, s in enumerate(shard_spec.spec):
                    if s > 1:
                        shard_size = int(w.dims[j] / s)
                        slc[j] = slice(_ * shard_size, _ * shard_size + shard_size)
                w_shard = numpy_helper.from_array(w_array[tuple(slc)])
                w_shard.name = w.name
                graph.initializer.append(w_shard)
            else:
                graph.initializer.append(w)

        onnx.save(model, f'{output_name}_{_}.onnx')
        del graph.initializer[:]

model = onnx.load('bert_base_cased_1_fp16_gpu_shaped.onnx')
shard = 4
megatron_shard_spec = {# layer 0
              'Attention_0_qkv_weight': ShardSpec([1, shard], False),
              'Attention_0_qkv_bias': ShardSpec([shard], False),
              '1654': ShardSpec([shard, 1,], False),
              '1655': ShardSpec([1, shard], False),
              'encoder.layer.0.intermediate.dense.bias': ShardSpec([shard], False),
              '1656': ShardSpec([shard, 1], False),
              # layer 1
              'Attention_1_qkv_weight': ShardSpec([1, shard], False),
              'Attention_1_qkv_bias': ShardSpec([shard], False),
              '1667': ShardSpec([shard, 1,], False),
              '1668': ShardSpec([1, shard], False),
              'encoder.layer.1.intermediate.dense.bias': ShardSpec([shard], False),
              '1669': ShardSpec([shard, 1], False),
              # layer 2
              'Attention_2_qkv_weight': ShardSpec([1, shard], False),
              'Attention_2_qkv_bias': ShardSpec([shard], False),
              '1680': ShardSpec([shard, 1,], False),
              '1681': ShardSpec([1, shard], False),
              'encoder.layer.2.intermediate.dense.bias': ShardSpec([shard], False),
              '1682': ShardSpec([shard, 1], False),
              # layer 3
              'Attention_3_qkv_weight': ShardSpec([1, shard], False),
              'Attention_3_qkv_bias': ShardSpec([shard], False),
              '1693': ShardSpec([shard, 1,], False),
              '1694': ShardSpec([1, shard], False),
              'encoder.layer.3.intermediate.dense.bias': ShardSpec([shard], False),
              '1695': ShardSpec([shard, 1], False),
              # layer 4
              'Attention_4_qkv_weight': ShardSpec([1, shard], False),
              'Attention_4_qkv_bias': ShardSpec([shard], False),
              '1706': ShardSpec([shard, 1,], False),
              '1707': ShardSpec([1, shard], False),
              'encoder.layer.4.intermediate.dense.bias': ShardSpec([shard], False),
              '1708': ShardSpec([shard, 1], False),
              # layer 5
              'Attention_5_qkv_weight': ShardSpec([1, shard], False),
              'Attention_5_qkv_bias': ShardSpec([shard], False),
              '1719': ShardSpec([shard, 1,], False),
              '1720': ShardSpec([1, shard], False),
              'encoder.layer.5.intermediate.dense.bias': ShardSpec([shard], False),
              '1721': ShardSpec([shard, 1], False),
              # layer 6
              'Attention_6_qkv_weight': ShardSpec([1, shard], False),
              'Attention_6_qkv_bias': ShardSpec([shard], False),
              '1732': ShardSpec([shard, 1,], False),
              '1733': ShardSpec([1, shard], False),
              'encoder.layer.6.intermediate.dense.bias': ShardSpec([shard], False),
              '1734': ShardSpec([shard, 1], False),
              # layer 7
              'Attention_7_qkv_weight': ShardSpec([1, shard], False),
              'Attention_7_qkv_bias': ShardSpec([shard], False),
              '1745': ShardSpec([shard, 1,], False),
              '1746': ShardSpec([1, shard], False),
              'encoder.layer.7.intermediate.dense.bias': ShardSpec([shard], False),
              '1747': ShardSpec([shard, 1], False),
              }

allgather_gemm_shard_spec = {# layer 0
              'Attention_0_qkv_weight': ShardSpec([1, shard], False),
              'Attention_0_qkv_bias': ShardSpec([shard], False),
              '1654': ShardSpec([1, shard], False),
              '1655': ShardSpec([1, shard], False),
              'encoder.layer.0.intermediate.dense.bias': ShardSpec([shard], False),
              '1656': ShardSpec([1, shard], False),
              # layer 1
              'Attention_1_qkv_weight': ShardSpec([1, shard], False),
              'Attention_1_qkv_bias': ShardSpec([shard], False),
              '1667': ShardSpec([1, shard], False),
              '1668': ShardSpec([1, shard], False),
              'encoder.layer.1.intermediate.dense.bias': ShardSpec([shard], False),
              '1669': ShardSpec([1, shard], False),
              # layer 2
              'Attention_2_qkv_weight': ShardSpec([1, shard], False),
              'Attention_2_qkv_bias': ShardSpec([shard], False),
              '1680': ShardSpec([1, shard], False),
              '1681': ShardSpec([1, shard], False),
              'encoder.layer.2.intermediate.dense.bias': ShardSpec([shard], False),
              '1682': ShardSpec([1, shard], False),
              # layer 3
              'Attention_3_qkv_weight': ShardSpec([1, shard], False),
              'Attention_3_qkv_bias': ShardSpec([shard], False),
              '1693': ShardSpec([1, shard], False),
              '1694': ShardSpec([1, shard], False),
              'encoder.layer.3.intermediate.dense.bias': ShardSpec([shard], False),
              '1695': ShardSpec([1, shard], False),
              # layer 4
              'Attention_4_qkv_weight': ShardSpec([1, shard], False),
              'Attention_4_qkv_bias': ShardSpec([shard], False),
              '1706': ShardSpec([1, shard], False),
              '1707': ShardSpec([1, shard], False),
              'encoder.layer.4.intermediate.dense.bias': ShardSpec([shard], False),
              '1708': ShardSpec([1, shard], False),
              # layer 5
              'Attention_5_qkv_weight': ShardSpec([1, shard], False),
              'Attention_5_qkv_bias': ShardSpec([shard], False),
              '1719': ShardSpec([1, shard], False),
              '1720': ShardSpec([1, shard], False),
              'encoder.layer.5.intermediate.dense.bias': ShardSpec([shard], False),
              '1721': ShardSpec([1, shard], False),
              # layer 6
              'Attention_6_qkv_weight': ShardSpec([1, shard], False),
              'Attention_6_qkv_bias': ShardSpec([shard], False),
              '1732': ShardSpec([1, shard], False),
              '1733': ShardSpec([1, shard], False),
              'encoder.layer.6.intermediate.dense.bias': ShardSpec([shard], False),
              '1734': ShardSpec([1, shard], False),
              # layer 7
              'Attention_7_qkv_weight': ShardSpec([1, shard], False),
              'Attention_7_qkv_bias': ShardSpec([shard], False),
              '1745': ShardSpec([1, shard], False),
              '1746': ShardSpec([1, shard], False),
              'encoder.layer.7.intermediate.dense.bias': ShardSpec([shard], False),
              '1747': ShardSpec([1, shard], False),
              }
shard_model(model, allgather_gemm_shard_spec, 'bert_shard_all_gather', shard)
print('Done')


    
