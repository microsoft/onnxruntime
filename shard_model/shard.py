import onnx
from enum import Enum
from onnx import helper
from onnx import numpy_helper
import numpy as np
import argparse

def gen_allgather(arg_name, out_name, out_shape, ranks):
    assert out_shape[-1] % ranks == 0

    ag_node = helper.make_node(
                    "NcclAllGatherV2",
                    [arg_name],
                    [f'{arg_name}_out'],
                    domain = "com.microsoft",
                    name="",
                    world_size=ranks,
                    )

    if len(out_shape) == 1:
        return [ag_node], []

    r1_shape = [ranks,] + out_shape
    r1_shape[-1] = int(r1_shape[-1] / ranks)

    # generate reshape->transpose->reshape
    r1 = np.array(r1_shape, dtype=np.int64)
    r1 = numpy_helper.from_array(r1, f'{arg_name}_reshape_1')
    r1_node = helper.make_node(
            'Reshape',
            [f'{arg_name}_out', f'{arg_name}_reshape_1'],
            [f'{arg_name}_r1_out']
            )
    perm = list(range(len(r1_shape)))[1:]
    perm.append(0)
    perm[-1], perm[-2] = perm[-2], perm[-1]
    
    t_node = helper.make_node(
            'Transpose',
            [f'{arg_name}_r1_out'],
            [f'{arg_name}_t1_out'],
            perm=perm
            )
    r2 = np.array(out_shape, dtype=np.int64)
    r2 = numpy_helper.from_array(r2, f'{arg_name}_r2')
    r2_node = helper.make_node(
            'Reshape',
            [f'{arg_name}_t1_out', f'{arg_name}_r2'],
            [out_name]
            )
    return (ag_node, r1_node, t_node, r2_node), (r1, r2)

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
                    actions[i] = Actions.AllGather
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

def lookup_arg_type_shape(graph, arg_name):
    arg = lookup_arg(graph, arg_name)
    if arg is not None:
        return arg.type.tensor_type.elem_type, [_.dim_value for _ in arg.type.tensor_type.shape.dim]
    for init in graph.initializer:
        if init.name == arg_name:
            return init.data_type, [_ for _ in init.dims]
    return None, None

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
    elif node.op_type == "FastGelu" or node.op_type == 'BiasGelu':
        return ElementwiseOp(node.op_type,
                      [lookup_arg_shape(graph, input) for input in node.input],
                      [lookup_arg_shape(graph, output) for output in node.output],
                      ((2, (2, 0),),))
    return ShardedOnnxOp(node.op_type, 
                      [lookup_arg_shape(graph, input) for input in node.input],
                      [lookup_arg_shape(graph, output) for output in node.output])

def shard_model(model, shard_specs, output_name, rank, num_layers):
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
    collective_initializer = []
    for arg_name in collectives_actions:
        action = collectives_actions[arg_name]
        if action == Actions.AllReduce:
            arg_type, arg_shape = lookup_arg_type_shape(graph, arg_name)
            assert arg_type is not None, f'can not find arg {arg_name} in the graph'
            new_arg = helper.make_tensor_value_info(f'{arg_name}_all_reduce', arg_type, arg_shape)
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
            arg_type, arg_shape = lookup_arg_type_shape(graph, arg_name)
            assert arg_type is not None, f'can not find arg {arg_name} in the graph'
            # this is not correct, temporary use it to demostrate.
            new_arg = helper.make_tensor_value_info(f'{arg_name}_all_gather', arg_type, arg_shape)
            ag_nodes, ag_inits = gen_allgather(arg_name, new_arg.name, arg_shape, rank)
            #allgather_node = helper.make_node(
            #                "NcclAllGather",
            #                [arg_name],
            #                [new_arg.name],
            #                domain = "com.microsoft",
            #                name=""
            #                )
            coolective_nodes.extend(ag_nodes)
            collective_initializer.extend(ag_inits)
            collective_args_map[arg_name] = new_arg.name
        else:
            raise Exception("Not Implemented.")
    
    atten_name = [f'Attention_{i}' for i in range(num_layers)]
    for node in graph.node:
        new_inputs = [collective_args_map[_] if _ in collective_args_map else _ for _ in node.input]
        del node.input[:]
        node.input.extend(new_inputs)
        # change attention's head to shard
        if node.name in atten_name:
            for a in node.attribute:
                if a.name == 'num_heads':
                    a.i = int(a.i / rank)
    graph.node.extend(coolective_nodes)

    #added_outputs = ['EmbedLayerNormalization_0_output', 'onnx::MatMul_336', 'onnx::Add_338']
    #for v in graph.value_info:
    #    if v.name in added_outputs:
    #        graph.output.append(v)


    # clean all the shapes
    #for value_info in graph.value_info:
    #    #if value_info.name != 'onnx::Expand_229' and value_info.name not in [_.name for _ in graph.input] and value_info.name not in [_.name for _ in graph.initializer]:
    #    if value_info.name not in [_.name for _ in graph.initializer]:
    #        del value_info.type.tensor_type.shape.dim[:]
    del graph.value_info[:]


    # shard tensors
    original_initializers = []
    original_initializers.extend(graph.initializer)
    del graph.initializer[:]

    opset=model.opset_import.add()
    opset.domain="com.microsoft"
    opset.version=1
    
    qkv_weight = [f'{i}_qkv_weight' for i in atten_name]
    qkv_bias = [f'{i}_qkv_bias' for i in atten_name]

    for _ in range(rank):
        for i, w in enumerate(original_initializers):
            shard_spec = shard_dict[w.name]
            if w.name in qkv_weight:
                w_array = numpy_helper.to_array(w)
                w_array = w_array.reshape((w_array.shape[0], 3, int(w_array.shape[1]/3)))
                spec = [1,1,rank]
                slc = [slice(None)] * len(spec)
                for j, s in enumerate(spec):
                    if s > 1:
                        shard_size = int(w_array.shape[j] / s)
                        slc[j] = slice(_ * shard_size, _ * shard_size + shard_size)
                w_array = w_array[tuple(slc)].reshape((w_array.shape[0], -1))
                w_shard = numpy_helper.from_array(w_array)
                w_shard.name = w.name
                graph.initializer.append(w_shard)
                continue
            if w.name in qkv_bias:
                w_array = numpy_helper.to_array(w)
                w_array = w_array.reshape((3, int(w_array.shape[0]/3)))
                spec = [1,rank]
                slc = [slice(None)] * len(spec)
                for j, s in enumerate(spec):
                    if s > 1:
                        shard_size = int(w_array.shape[j] / s)
                        slc[j] = slice(_ * shard_size, _ * shard_size + shard_size)
                w_array = w_array[tuple(slc)].reshape(-1)
                w_shard = numpy_helper.from_array(w_array)
                w_shard.name = w.name
                graph.initializer.append(w_shard)
                continue
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

        graph.initializer.extend(collective_initializer)
        onnx.save(model, f'{output_name}_{_}.onnx')
        del graph.initializer[:]

shard = 4
megatron_shard_spec = {# layer 0
              'Attention_0_qkv_weight': ShardSpec([1, shard], False),
              'Attention_0_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1650': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1651': ShardSpec([1, shard], False),
              'encoder.layer.0.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1652': ShardSpec([shard, 1], False),
              # layer 1
              'Attention_1_qkv_weight': ShardSpec([1, shard], False),
              'Attention_1_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1663': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1664': ShardSpec([1, shard], False),
              'encoder.layer.1.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1665': ShardSpec([shard, 1], False),
              ## layer 2
              'Attention_2_qkv_weight': ShardSpec([1, shard], False),
              'Attention_2_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1676': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1677': ShardSpec([1, shard], False),
              'encoder.layer.2.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1678': ShardSpec([shard, 1], False),
              # layer 3
              'Attention_3_qkv_weight': ShardSpec([1, shard], False),
              'Attention_3_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1689': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1690': ShardSpec([1, shard], False),
              'encoder.layer.3.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1691': ShardSpec([shard, 1], False),
              # layer 4
              'Attention_4_qkv_weight': ShardSpec([1, shard], False),
              'Attention_4_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1702': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1703': ShardSpec([1, shard], False),
              'encoder.layer.4.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1704': ShardSpec([shard, 1], False),
              # layer 5
              'Attention_5_qkv_weight': ShardSpec([1, shard], False),
              'Attention_5_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1715': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1716': ShardSpec([1, shard], False),
              'encoder.layer.5.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1717': ShardSpec([shard, 1], False),
              # layer 6
              'Attention_6_qkv_weight': ShardSpec([1, shard], False),
              'Attention_6_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1728': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1729': ShardSpec([1, shard], False),
              'encoder.layer.6.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1730': ShardSpec([shard, 1], False),
              # layer 7
              'Attention_7_qkv_weight': ShardSpec([1, shard], False),
              'Attention_7_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1741': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1742': ShardSpec([1, shard], False),
              'encoder.layer.7.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1743': ShardSpec([shard, 1], False),
              # layer 8
              'Attention_8_qkv_weight': ShardSpec([1, shard], False),
              'Attention_8_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1754': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1755': ShardSpec([1, shard], False),
              'encoder.layer.8.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1756': ShardSpec([shard, 1], False),
              # layer 9
              'Attention_9_qkv_weight': ShardSpec([1, shard], False),
              'Attention_9_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1767': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1768': ShardSpec([1, shard], False),
              'encoder.layer.9.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1769': ShardSpec([shard, 1], False),
              # layer 10
              'Attention_10_qkv_weight': ShardSpec([1, shard], False),
              'Attention_10_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1780': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1781': ShardSpec([1, shard], False),
              'encoder.layer.10.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1782': ShardSpec([shard, 1], False),
              # layer 11
              'Attention_11_qkv_weight': ShardSpec([1, shard], False),
              'Attention_11_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1793': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1794': ShardSpec([1, shard], False),
              'encoder.layer.11.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1795': ShardSpec([shard, 1], False),
              }

allgather_gemm_shard_spec = {# layer 0
              'Attention_0_qkv_weight': ShardSpec([1, shard], False),
              'Attention_0_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1650': ShardSpec([1, shard], False),
              'onnx::MatMul_1651': ShardSpec([1, shard], False),
              'encoder.layer.0.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1652': ShardSpec([1, shard], False),
              # layer 1
              'Attention_1_qkv_weight': ShardSpec([1, shard], False),
              'Attention_1_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1663': ShardSpec([1, shard], False),
              'onnx::MatMul_1664': ShardSpec([1, shard], False),
              'encoder.layer.1.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1665': ShardSpec([1, shard], False),
              # layer 2
              'Attention_2_qkv_weight': ShardSpec([1, shard], False),
              'Attention_2_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1676': ShardSpec([1, shard], False),
              'onnx::MatMul_1677': ShardSpec([1, shard], False),
              'encoder.layer.2.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1678': ShardSpec([1, shard], False),
              # layer 3
              'Attention_3_qkv_weight': ShardSpec([1, shard], False),
              'Attention_3_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1689': ShardSpec([1, shard], False),
              'onnx::MatMul_1690': ShardSpec([1, shard], False),
              'encoder.layer.3.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1691': ShardSpec([1, shard], False),
              # layer 4
              'Attention_4_qkv_weight': ShardSpec([1, shard], False),
              'Attention_4_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1702': ShardSpec([1, shard], False),
              'onnx::MatMul_1703': ShardSpec([1, shard], False),
              'encoder.layer.4.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1704': ShardSpec([1, shard], False),
              # layer 5
              'Attention_5_qkv_weight': ShardSpec([1, shard], False),
              'Attention_5_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1715': ShardSpec([1, shard], False),
              'onnx::MatMul_1716': ShardSpec([1, shard], False),
              'encoder.layer.5.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1717': ShardSpec([1, shard], False),
              # layer 6
              'Attention_6_qkv_weight': ShardSpec([1, shard], False),
              'Attention_6_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1728': ShardSpec([1, shard], False),
              'onnx::MatMul_1729': ShardSpec([1, shard], False),
              'encoder.layer.6.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1730': ShardSpec([1, shard], False),
              # layer 7
              'Attention_7_qkv_weight': ShardSpec([1, shard], False),
              'Attention_7_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1741': ShardSpec([1, shard], False),
              'onnx::MatMul_1742': ShardSpec([1, shard], False),
              'encoder.layer.7.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1743': ShardSpec([1, shard], False),
              # layer 8
              'Attention_8_qkv_weight': ShardSpec([1, shard], False),
              'Attention_8_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1754': ShardSpec([1, shard], False),
              'onnx::MatMul_1755': ShardSpec([1, shard], False),
              'encoder.layer.8.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1756': ShardSpec([1, shard], False),
              # layer 9
              'Attention_9_qkv_weight': ShardSpec([1, shard], False),
              'Attention_9_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1767': ShardSpec([1, shard], False),
              'onnx::MatMul_1768': ShardSpec([1, shard], False),
              'encoder.layer.9.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1769': ShardSpec([1, shard], False),
              # layer 10
              'Attention_10_qkv_weight': ShardSpec([1, shard], False),
              'Attention_10_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1780': ShardSpec([1, shard], False),
              'onnx::MatMul_1781': ShardSpec([1, shard], False),
              'encoder.layer.10.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1782': ShardSpec([1, shard], False),
              # layer 11
              'Attention_11_qkv_weight': ShardSpec([1, shard], False),
              'Attention_11_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1793': ShardSpec([1, shard], False),
              'onnx::MatMul_1794': ShardSpec([1, shard], False),
              'encoder.layer.11.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1795': ShardSpec([1, shard], False),
              }

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--shard-prefix', type=str)
    parser.add_argument('--mode', type=str, default='allreduce')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model = onnx.load(args.model_file)
    if args.mode == 'allreduce':
        shard_model(model, megatron_shard_spec, args.shard_prefix, shard,12)
    elif args.mode == 'allgather':
        shard_model(model, allgather_gemm_shard_spec, args.shard_prefix, shard,12)
    print('Shard Done')
