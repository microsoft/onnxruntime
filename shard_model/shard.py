import onnx
from enum import Enum
from onnx import helper
from onnx import numpy_helper
import numpy as np
import argparse
from shard_spec import ShardSpec, megatron_shard_spec, allgather_gemm_shard_spec, get_gpt2_spec

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
            if shard is None:
                actions.append(Actions.NoAction)
                continue
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
        #assert len(self.output_shapes) == 1
        assert input_sharded_shape[1][1] == input_sharded_shape[2][0]
        spec = [int(self.output_shapes[0][0] / input_sharded_shape[0][0]),
                int(self.output_shapes[0][1] / input_sharded_shape[0][1]),
                int(self.output_shapes[0][2] / int(input_sharded_shape[1][1] / 3))]
        ret = [ShardSpec(spec, partial_result)]
        if len(self.output_shapes) > 1:
            # not need to split second output
            ret.append(ShardSpec([int(1)] * len(self.output_shapes[1]), False))
        return ret

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
                actions[i] = Actions.AllReduce
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
        assert len(w) == 1, f'no shape for arg: {arg_name}'
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

def lookup_shape(graph, name_list):
    ret = []
    for n in name_list:
        if n != "":
            ret.append(lookup_arg_shape(graph, n))
        else:
            ret.append([])
    return ret

def get_shard_op(node, graph):
    input_shapes = lookup_shape(graph, node.input)
    output_shapes = lookup_shape(graph, node.output)
    elementwise_op_list = ['FastGelu', 'BiasGelu', 'Cast']
    if node.op_type == "Gemm":
        return GemmOp(node.op_type, 
                      input_shapes,
                      output_shapes,
                      ((1, 0),))
    elif node.op_type == "MatMul":
        return MatmulOp(node.op_type, 
                      input_shapes,
                      output_shapes,
                      ((len(input_shapes[0]) - 1, 0),))
    elif node.op_type == "Attention":
        return AttentionOp(node.op_type,
                      input_shapes,
                      output_shapes,
                      ((2, 0, -1, -1),))
    elif node.op_type in elementwise_op_list:
        return ElementwiseOp(node.op_type,
                      input_shapes,
                      output_shapes,
                      ((2, (2, 0),),))
    return ShardedOnnxOp(node.op_type, 
                      input_shapes,
                      output_shapes)

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
        input_shard_spec = [shard_dict[_] if _ != "" else None for _ in node.input]
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
            coolective_nodes.extend(ag_nodes)
            collective_initializer.extend(ag_inits)
            collective_args_map[arg_name] = new_arg.name
        else:
            raise Exception("Not Implemented.")
    
    qkv_weight = []
    qkv_bias = []
    past_names = []
    present_names = []
    for node in graph.node:
        new_inputs = [collective_args_map[_] if _ in collective_args_map else _ for _ in node.input]
        del node.input[:]
        node.input.extend(new_inputs)
        # change attention's head to shard
        if node.op_type == 'Attention':
            qkv_weight.append(node.input[1])
            qkv_bias.append(node.input[2])
            for a in node.attribute:
                if a.name == 'num_heads':
                    a.i = int(a.i / rank)
            # modify past value_info
            if len(node.input) >= 5:
                past_names.append(node.input[4])
            if len(node.output) >= 2:
                present_names.append(node.output[1])

    # modify attention past and present
    for i in graph.input:
        if i.name in past_names:
            # past shape is (2, batch, num_head, past_seq_len, hid/num_heads)
            shape = i.type.tensor_type.shape.dim
            shape[2].dim_value = shape[2].dim_value // rank
    for o in graph.output:
        if o.name in present_names:
            # present shape is (2, batch, num_head, past_seq+seq_len, hid/num_heads)
            shape = o.type.tensor_type.shape.dim
            shape[2].dim_value = shape[2].dim_value // rank


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
        output_file = f'{output_name}_{_}.onnx'
        external_data_file = f'{output_file}.data'
        onnx.save(model,
                output_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_file
        )
        del graph.initializer[:]

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--shard-prefix', type=str)
    parser.add_argument('--mode', type=str, default='allreduce')
    parser.add_argument('--num-shards', type=int, default=4)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model = onnx.load(args.model_file)
    #if args.mode == 'allreduce':
    #    shard_model(model, megatron_shard_spec, args.shard_prefix, args.num_shards,12)
    #elif args.mode == 'allgather':
    #    shard_model(model, allgather_gemm_shard_spec, args.shard_prefix, args.num_shards,12)
    num_layers=36
    spec = get_gpt2_spec(args.num_shards, num_layers, mode=args.mode)
    shard_model(model, spec, args.shard_prefix, args.num_shards, num_layers)
    print('Shard Done')
