# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import argparse
import numpy as np
import onnx
import sys
from onnx import helper, numpy_helper, shape_inference
import sympy

from packaging import version
assert version.parse(onnx.__version__) >= version.parse("1.5.0") # need at least opset 10 for MatMulInteger shape inference

def get_attribute(node, attr_name, default_value=None):
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value

def get_shape_from_type_proto(type_proto):
    return [getattr(i, i.WhichOneof('value')) if type(i.WhichOneof('value')) == str else None for i in type_proto.tensor_type.shape.dim]

def get_shape_from_sympy_shape(sympy_shape):
    return [None if i is None else (int(i) if is_literal(i) else str(i)) for i in sympy_shape]

def is_literal(dim):
    return type(dim) in [int, np.int64, np.int32, sympy.Integer] or (hasattr(dim, 'is_number') and dim.is_number)

def handle_negative_axis(axis, rank):
    assert axis < rank and axis >= -rank
    return axis if axis >= 0 else rank + axis

def get_opset(mp, domain=['', 'onnx']):
    if type(domain) != list:
        domain = [domain]
    for opset in mp.opset_import:
        if opset.domain in domain:
            return opset.version
    return None

def as_scalar(x):
    if type(x) == list:
        assert len(x) == 1
        return x[0]
    elif type(x) == np.ndarray:
        return np.asscalar(x)
    else:
        return x

def sympy_reduce_product(x):
    if type(x) == list:
        value = sympy.Integer(1)
        for v in x:
            value = value * v
    else:
        value = x
    return value

class SymbolicShapeInference:
    def __init__(self, int_max, auto_merge, verbose):
        self.dispatcher_ = {
            'Add'                   : self._infer_binary_ops,
            'ArrayFeatureExtractor' : self._infer_ArrayFeatureExtractor,
            'AveragePool'           : self._infer_Pool,
            'Cast'                  : self._infer_Cast,
            'CategoryMapper'        : self._infer_CategoryMapper,
            'Compress'              : self._infer_Compress,
            'Concat'                : self._infer_Concat,
            'ConstantOfShape'       : self._infer_ConstantOfShape,
            'Conv'                  : self._infer_Conv,
            'CumSum'                : self._pass_on_shape_and_type,
            'Div'                   : self._infer_binary_ops,
            'Expand'                : self._infer_Expand,
            'Gather'                : self._infer_Gather,
            'GatherElements'        : self._infer_GatherElements,
            'Loop'                  : self._infer_Loop,
            'MatMul'                : self._infer_MatMul,
            'MatMulInteger16'       : self._infer_MatMulInteger,
            'MaxPool'               : self._infer_Pool,
            'Max'                   : self._infer_binary_ops,
            'Min'                   : self._infer_binary_ops,
            'Mul'                   : self._infer_binary_ops,
            'NonMaxSuppression'     : self._infer_NonMaxSuppression,
            'NonZero'               : self._infer_NonZero,
            'OneHot'                : self._infer_OneHot,
            'Pad'                   : self._infer_Pad,
            'Range'                 : self._infer_Range,
            'ReduceProd'            : self._infer_ReduceProd,
            'Reshape'               : self._infer_Reshape,
            'Resize'                : self._infer_Resize,
            'Round'                 : self._pass_on_shape_and_type,
            'Scan'                  : self._infer_Scan,
            'ScatterElements'       : self._infer_ScatterElements,
            'Shape'                 : self._infer_Shape,
            'Size'                  : self._infer_Size,
            'Slice'                 : self._infer_Slice,
            'Split'                 : self._infer_Split,
            'Squeeze'               : self._infer_Squeeze,
            'Sub'                   : self._infer_binary_ops,
            'Tile'                  : self._infer_Tile,
            'TopK'                  : self._infer_TopK,
            'Unsqueeze'             : self._infer_Unsqueeze,
            'ZipMap'                : self._infer_ZipMap}
        self.run_ = True
        self.suggested_merge_ = {}
        self.symbolic_dims_ = {}
        self.auto_merge_ = auto_merge
        self.verbose_ = verbose
        self.int_max_ = int_max

    def _add_suggested_merge(self, symbols):
        assert all([(type(s) == str and s in self.symbolic_dims_) or is_literal(s) for s in symbols])
        symbols = set(symbols)
        for k,v in self.suggested_merge_.items():
            if k in symbols:
                symbols.remove(k)
                symbols.add(v)
        map_to = None
        # if there is literal, map to it first
        for s in symbols:
            if is_literal(s):
                map_to = s
                break
        # when no literals, map to existing symbolic dims
        if map_to is None:
            for s in symbols:
                if type(self.symbolic_dims_[s]) == sympy.Symbol:
                    map_to = s
                    break
        # when nothing to map to, use the first one
        if map_to is None:
            if self.verbose_ > 0:
                print('Potential unsafe merge between symbolic expressions: ({})'.format(','.join(symbols)))
            map_to = symbols.pop() # force merge when unable to determine
        for s in symbols:
            if s == map_to:
                continue
            if is_literal(map_to) and is_literal(s):
                assert int(map_to) == int(s)
            self.suggested_merge_[s] = int(map_to) if is_literal(map_to) else map_to
            for k,v in self.suggested_merge_.items():
                if v == s:
                    self.suggested_merge_[k] = map_to

    def _apply_suggested_merge_to_graph_input(self):
        if not self.suggested_merge_:
            return
        for i in self.out_mp_.graph.input:
            for d in i.type.tensor_type.shape.dim:
                if d.dim_param in self.suggested_merge_:
                    v = self.suggested_merge_[d.dim_param]
                    if is_literal(v):
                        d.dim_value = int(v)
                    else:
                        d.dim_param = v

    def _preprocess(self, in_mp):
        out_mp = onnx.ModelProto()
        out_mp.CopyFrom(in_mp)
        out_mp.graph.ClearField('node')
        self.out_mp_ = out_mp

        defined = set([i.name for i in list(in_mp.graph.input) + list(in_mp.graph.initializer)])
        pending_nodes = []

        # returns True if no more ready nodes
        def _insert_ready_nodes():
            ready_nodes = [pn for pn in pending_nodes if all([i in defined for i in pn.input if i])]
            for rn in ready_nodes:
                self.out_mp_.graph.node.add().CopyFrom(rn)
                for o in rn.output:
                    defined.add(o)
                pending_nodes.remove(rn)
            return not ready_nodes

        # constant op -> initializer, topological sort
        for in_n in in_mp.graph.node:
            if in_n.op_type == 'Constant':
                t = get_attribute(in_n, 'value')
                t.name = in_n.output[0]
                self.out_mp_.graph.initializer.add().CopyFrom(t)
                defined.add(t.name)
            else:
                pending_nodes.append(in_n)
            _insert_ready_nodes()

        while pending_nodes:
            if _insert_ready_nodes():
                break

        if pending_nodes and self.verbose_ > 0:
            print('SymbolicShapeInference: orphaned nodes discarded: ')
            print(*[n.op_type + ': ' + n.output[0] for n in pending_nodes], sep='\n')

        self.initializers_ = dict([(i.name, i) for i in self.out_mp_.graph.initializer])
        self.known_vi_ = dict([(i.name, i) for i in list(self.out_mp_.graph.input)])
        self.known_vi_.update(dict([(i.name, helper.make_tensor_value_info(i.name, i.data_type, list(i.dims))) for i in self.out_mp_.graph.initializer]))

    def _merge_symbols(self, dims):
        if not all([type(d) == str for d in dims]):
            if self.auto_merge_:
                assert len(dims) == 2 # only allow symbol->int merge in binary ops for now
                is_int = [is_literal(d) for d in dims]
                assert sum(is_int) == 1
                int_dim = is_int.index(1)
                if self.verbose_ > 0:
                    print('dim {} has been merged with value {}'.format(dims[1 - int_dim], dims[int_dim]))
                return dims[int_dim]
            else:
                return None
        if all([d == dims[0] for d in dims]):
            return dims[0]
        merged = [self.suggested_merge_[d] if d in self.suggested_merge_ else d for d in dims]
        if all([d == merged[0] for d in merged]):
            assert merged[0] in self.symbolic_dims_
            return merged[0]
        else:
            return None

    # broadcast from right to left, and merge symbolic dims if needed
    def _broadcast_shapes(self, shape1, shape2):
        new_shape = []
        rank1 = len(shape1)
        rank2 = len(shape2)
        new_rank = max(rank1, rank2)
        for i in range(new_rank):
            dim1 = shape1[rank1 - 1 - i] if i < rank1 else 1
            dim2 = shape2[rank2 - 1 - i] if i < rank2 else 1
            if dim1 == 1 or dim1 == dim2:
                new_dim = dim2
            elif dim2 == 1:
                new_dim = dim1
            else:
                new_dim = self._merge_symbols([dim1, dim2])
                if not new_dim:
                    print('unsupported broadcast between ' + str(dim1) + ' ' + str(dim2))
            new_shape = [new_dim] + new_shape
        return new_shape

    def _get_shape(self, node, idx):
        name = node.input[idx]
        if name in self.known_vi_:
            return get_shape_from_type_proto(self.known_vi_[name].type)
        else:
            assert name in self.initializers_
            return list(self.initializers_[name].dims)

    def _get_shape_rank(self, node, idx):
        return len(self._get_shape(node, idx))

    def _get_sympy_shape(self, node, idx):
        sympy_shape = []
        for d in self._get_shape(node, idx):
            if type(d) == str:
                sympy_shape.append(self.symbolic_dims_[d] if d in self.symbolic_dims_ else sympy.Symbol(d, integer=True))
            else:
                assert None != d
                sympy_shape.append(d)
        return sympy_shape

    def _get_value(self, node, idx):
        name = node.input[idx]
        assert name in self.sympy_data_ or name in self.initializers_
        return self.sympy_data_[name] if name in self.sympy_data_ else numpy_helper.to_array(self.initializers_[name])

    def _try_get_value(self, node, idx):
        if idx >= len(node.input):
            return None
        name = node.input[idx]
        if name in self.sympy_data_ or name in self.initializers_:
            return self._get_value(node, idx)
        return None

    def _update_computed_dims(self, new_sympy_shape):
        for i, new_dim in enumerate(new_sympy_shape):
            if not is_literal(new_dim) and not type(new_dim) == str:
                str_dim = str(new_dim)
                if str_dim in self.suggested_merge_:
                    new_sympy_shape[i] = self.symbolic_dims_[self.suggested_merge_[str_dim]]
                else:
                    # add new_dim if it's a computational expression
                    if not str(new_dim) in self.symbolic_dims_:
                        self.symbolic_dims_[str(new_dim)] = new_dim

    def _onnx_infer_single_node(self, node):
        # skip onnx shape inference for Scan/Loop
        skip_infer = node.op_type in ['Scan', 'Loop']
        if not skip_infer:
            # run single node inference with self.known_vi_ shapes
            # note that inference rely on initializer values is not handled
            # as we don't copy initializer weights to tmp_graph for inference speed purpose
            tmp_graph = helper.make_graph([node],
                                          'tmp',
                                          [self.known_vi_[i] for i in node.input if i],
                                          [helper.make_tensor_value_info(i, onnx.TensorProto.UNDEFINED, None) for i in node.output])
            self.tmp_mp_.graph.CopyFrom(tmp_graph)
            self.tmp_mp_ = shape_inference.infer_shapes(self.tmp_mp_)
        for i_o in range(len(node.output)):
            o = node.output[i_o]
            vi = self.out_mp_.graph.value_info.add()
            if not skip_infer:
                vi.CopyFrom(self.tmp_mp_.graph.output[i_o])
            self.known_vi_[o] = vi

    def _onnx_infer_subgraph(self, node, subgraph):
        if self.verbose_ > 2:
            print('Inferencing subgraph of node {} with output({}...): {}'.format(node.name, node.output[0], node.op_type))
        # node inputs are not passed directly to the subgraph
        # it's up to the node dispatcher to prepare subgraph input
        # for example, with Scan/Loop, subgraph input shape would be trimmed from node input shape
        # besides, inputs in subgraph could shadow implicit inputs
        subgraph_inputs = set([i.name for i in list(subgraph.initializer) + list(subgraph.input)])
        subgraph_implicit_input = set()
        for sn in subgraph.node:
            subgraph_implicit_input.update([i for i in sn.input if i in self.known_vi_ and i not in subgraph_inputs])
        tmp_graph = helper.make_graph(list(subgraph.node),
                                      'tmp',
                                      list(subgraph.input) + [self.known_vi_[i] for i in subgraph_implicit_input],
                                      [helper.make_tensor_value_info(i.name, onnx.TensorProto.UNDEFINED, None) for i in subgraph.output])
        tmp_graph.initializer.extend([i for i in self.out_mp_.graph.initializer if i.name in subgraph_implicit_input])
        tmp_graph.initializer.extend(subgraph.initializer)
        self.tmp_mp_.graph.CopyFrom(tmp_graph)

        symbolic_shape_inference = SymbolicShapeInference(self.int_max_, self.auto_merge_, self.verbose_)
        all_shapes_inferred = False
        symbolic_shape_inference._preprocess(self.tmp_mp_)
        symbolic_shape_inference.suggested_merge_ = self.suggested_merge_.copy()
        while symbolic_shape_inference.run_:
            all_shapes_inferred = symbolic_shape_inference._infer_impl(self.tmp_mp_)
        symbolic_shape_inference._update_output_from_vi()
        subgraph.ClearField('input')
        subgraph.input.extend(symbolic_shape_inference.out_mp_.graph.input[:len(node.input)])
        subgraph.ClearField('output')
        subgraph.output.extend(symbolic_shape_inference.out_mp_.graph.output)
        # for new symbolic dims from subgraph output, add to main graph symbolic dims
        subgraph_shapes = [get_shape_from_type_proto(o.type) for o in symbolic_shape_inference.out_mp_.graph.output]
        subgraph_new_symbolic_dims = set([d for s in subgraph_shapes if s for d in s if type(d) == str and not d in self.symbolic_dims_])
        self.symbolic_dims_.update({d:symbolic_shape_inference.symbolic_dims_[d] for d in subgraph_new_symbolic_dims})

    def _get_int_values(self, node, broadcast=False):
        values = [self._try_get_value(node, i) for i in range(len(node.input))]
        if all([v is not None for v in values]):
            # some shape compute is in floating point, cast to int for sympy
            for i,v in enumerate(values):
                if type(v) != np.ndarray:
                    continue
                assert len(v.shape) <= 1
                if len(v.shape) == 0:
                    new_v = int(np.asscalar(v))
                else:
                    assert len(v.shape) == 1
                    new_v = [int(vv) for vv in v]
                values[i] = new_v
        values_len = [len(v) if type(v) == list else 0 for v in values]
        max_len = max(values_len)
        if max_len >= 1 and broadcast:
            # broadcast
            for i,v in enumerate(values):
                if v is None:
                    continue # don't broadcast if value is unknown
                if type(v) == list:
                    if len(v) < max_len:
                        values[i] = v*max_len
                    else:
                        assert len(v) == max_len
                else:
                    values[i] = [v]*max_len
        return values

    def _compute_on_sympy_data(self, node, op_func):
        assert len(node.output) == 1
        values = self._get_int_values(node, broadcast=True)
        if all([v is not None for v in values]):
            is_list = [type(v) == list for v in values]
            as_list = any(is_list)
            if as_list:
                self.sympy_data_[node.output[0]] = [op_func(vs) for vs in zip(*values)]
            else:
                self.sympy_data_[node.output[0]] = op_func(values)

    def _pass_on_sympy_data(self, node):
        assert len(node.input) == 1 or node.op_type == 'Reshape'
        self._compute_on_sympy_data(node, lambda x: x[0])

    def _pass_on_shape_and_type(self, node):
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                                                  self._get_shape(node, 0)))

    def _new_symbolic_dim_from_output(self, node, out_idx=0, dim=0):
        new_dim = '{}{}_o{}_d{}'.format(node.op_type, list(self.out_mp_.graph.node).index(node), out_idx, dim)
        if new_dim in self.suggested_merge_:
            v = self.suggested_merge_[new_dim]
            new_dim = sympy.Integer(int(v)) if is_literal(v) else v
        else:
            self.symbolic_dims_[new_dim] = sympy.Symbol(new_dim, integer=True)
        return new_dim

    def _compute_conv_pool_shape(self, node):
        sympy_shape = self._get_sympy_shape(node, 0)
        if len(node.input) > 1:
            W_shape = self._get_sympy_shape(node, 1)
            rank = len(W_shape) - 2 # number of spatial axes
            kernel_shape = W_shape[-rank:]
            sympy_shape[1] = W_shape[0]
        else:
            W_shape = None
            kernel_shape = get_attribute(node, 'kernel_shape')
            rank = len(kernel_shape)

        assert len(sympy_shape) == rank + 2

        # only need to symbolic shape inference if input has symbolic dims in spatial axes
        is_symbolic_dims = [not is_literal(i) for i in sympy_shape[-rank:]]

        if not any(is_symbolic_dims):
            shape = get_shape_from_type_proto(self.known_vi_[node.output[0]].type)
            if len(shape) > 0:
                assert len(sympy_shape) == len(shape)
                sympy_shape[-rank:] = [sympy.Integer(d) for d in shape[-rank:]]
                return sympy_shape

        dilations = get_attribute(node, 'dilations', [1]*rank)
        strides = get_attribute(node, 'strides', [1]*rank)
        effective_kernel_shape = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]
        pads = get_attribute(node, 'pads')
        if pads is None:
            pads = [0]*(2*rank)
            auto_pad = get_attribute(node, 'auto_pad', b'NOTSET').decode('utf-8')
            if auto_pad != 'VALID' and auto_pad != 'NOTSET':
                try:
                    residual = [sympy.Mod(d, s) for d, s in zip(sympy_shape[-rank:], strides)]
                    total_pads = [max(0, (k - s) if r == 0 else (k - r)) for k, s, r in zip(effective_kernel_shape, strides, residual)]
                except TypeError: # sympy may throw TypeError: cannot determine truth value of Relational
                    total_pads = [max(0, (k - s)) for k, s in zip(effective_kernel_shape, strides)] # assuming no residual if sympy throws error
            elif auto_pad == 'VALID':
                total_pads = []
            else:
                total_pads = [0]*rank
        else:
            assert len(pads) == 2*rank
            total_pads = [p1 + p2 for p1, p2 in zip(pads[:rank], pads[rank:])]

        ceil_mode = get_attribute(node, 'ceil_mode', 0)
        for i in range(rank):
            effective_input_size = sympy_shape[-rank + i]
            if len(total_pads) > 0:
                effective_input_size = effective_input_size + total_pads[i]
            if ceil_mode:
                strided_kernel_positions = sympy.ceiling((effective_input_size - effective_kernel_shape[i]) / strides[i])
            else:
                strided_kernel_positions = (effective_input_size - effective_kernel_shape[i]) // strides[i]
            sympy_shape[-rank + i] = strided_kernel_positions + 1
        return sympy_shape
        
    def _compute_matmul_shape(self, node, output_dtype=None):
        lhs_shape = self._get_shape(node, 0)
        rhs_shape = self._get_shape(node, 1)
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)
        lhs_reduce_dim = 0
        rhs_reduce_dim = 0
        assert lhs_rank > 0 and rhs_rank > 0
        if lhs_rank == 1 and rhs_rank == 1:
            new_shape = []
        elif lhs_rank == 1:
            rhs_reduce_dim = -2
            new_shape = rhs_shape[:rhs_reduce_dim] + [rhs_shape[-1]]
        elif rhs_rank == 1:
            lhs_reduce_dim = -1
            new_shape = lhs_shape[:lhs_reduce_dim]
        else:
            lhs_reduce_dim = -1
            rhs_reduce_dim = -2
            new_shape = self._broadcast_shapes(lhs_shape[:-2], rhs_shape[:-2]) + [lhs_shape[-2]] + [rhs_shape[-1]]
        # record inconsistent reduce dim as suggested merge
        if lhs_shape[lhs_reduce_dim] != rhs_shape[rhs_reduce_dim]:
            merge_dims = [lhs_shape[lhs_reduce_dim], rhs_shape[rhs_reduce_dim]]
            self._add_suggested_merge(merge_dims)
        if output_dtype is None:
            # infer output_dtype from input type when not specified
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _infer_ArrayFeatureExtractor(self, node):
        data_shape = self._get_shape(node, 0)
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                                                  data_shape[:-1] + indices_shape))

    def _infer_binary_ops(self, node):
        funcs = {'Add' : lambda l: l[0] + l[1],
                 'Div' : lambda l: l[0] // l[1], # integer div in sympy
                 'Max' : lambda l: l[1] if is_literal(l[0]) and int(l[0]) < -self.int_max_ else (l[0] if is_literal(l[1]) and int(l[1]) < -self.int_max_ else sympy.Max(l[0], l[1])),
                 'Min' : lambda l: l[1] if is_literal(l[0]) and int(l[0]) >  self.int_max_ else (l[0] if is_literal(l[1]) and int(l[1]) >  self.int_max_ else sympy.Min(l[0], l[1])),
                 'Mul' : lambda l: l[0] * l[1],
                 'Sub' : lambda l: l[0] - l[1]}
        assert node.op_type in funcs
        self._compute_on_sympy_data(node, funcs[node.op_type])

    def _infer_Cast(self, node):
        self._pass_on_sympy_data(node)

    def _infer_CategoryMapper(self, node):
        input_type = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        if input_type == onnx.TensorProto.STRING:
            output_type = onnx.TensorProto.INT64
        else:
            output_type = onnx.TensorProto.STRING
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  output_type,
                                                  self._get_shape(node, 0)))

    def _infer_Compress(self, node):
        input_shape = self._get_shape(node, 0)
        # create a new symbolic dimension for Compress output
        compress_len = self._new_symbolic_dim_from_output(node)
        axis = get_attribute(node, 'axis')
        if axis == None:
            # when axis is not specified, input is flattened before compress so output is 1D
            output_shape = [compress_len]
        else:
            output_shape = input_shape
            output_shape[handle_negative_axis(axis, len(input_shape))] = compress_len
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, output_shape))

    def _infer_Concat(self, node):
        if any([i in self.sympy_data_ for i in node.input]):
            values = self._get_int_values(node)
            if all([v is not None for v in values]):
                assert 0 == get_attribute(node, 'axis')
                self.sympy_data_[node.output[0]] = []
                for i in range(len(node.input)):
                    value = values[i]
                    if type(value) == list:
                        self.sympy_data_[node.output[0]].extend(value)
                    else:
                        self.sympy_data_[node.output[0]].append(value)

        sympy_shape = self._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, 'axis'), len(sympy_shape))
        for i_idx in range(1, len(node.input)):
            sympy_shape[axis] = sympy_shape[axis] + self._get_sympy_shape(node, i_idx)[axis]
        self._update_computed_dims(sympy_shape)
        # merge symbolic dims for non-concat axes
        for d in range(len(sympy_shape)):
            if d == axis:
                continue
            dims = [self._get_shape(node, i_idx)[d] for i_idx in range(len(node.input))]
            if all([d == dims[0] for d in dims]):
                continue
            merged = self._merge_symbols(dims)
            if type(merged) == str:
                sympy_shape[d] = self.symbolic_dims_[merged] if merged else None
            else:
                sympy_shape[d] = merged
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, get_shape_from_sympy_shape(sympy_shape)))

    def _infer_Conv(self, node):
        sympy_shape = self._compute_conv_pool_shape(node)
        self._update_computed_dims(sympy_shape)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], vi.type.tensor_type.elem_type, get_shape_from_sympy_shape(sympy_shape)))

    def _infer_ConstantOfShape(self, node):
        sympy_shape = self._get_int_values(node)[0]
        if sympy_shape is not None:
            if type(sympy_shape) != list:
                sympy_shape = [sympy_shape]
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                      vi.type.tensor_type.elem_type,
                                                      get_shape_from_sympy_shape(sympy_shape)))

    def _infer_Expand(self, node):
        expand_to_shape = self._try_get_value(node, 1)
        if expand_to_shape is not None:
            input_shape = self._get_shape(node, 0)
            target_shape = get_shape_from_sympy_shape(expand_to_shape)
            new_shape = self._broadcast_shapes(input_shape, target_shape)
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, new_shape))

    def _infer_Gather(self, node):
        data_shape = self._get_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, 'axis', 0), len(data_shape))
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  vi.type.tensor_type.elem_type,
                                                  data_shape[:axis] + indices_shape + data_shape[axis+1:]))
        if node.input[0] in self.sympy_data_:
            assert 0 == get_attribute(node, 'axis', 0) # only handle 1D sympy compute
            idx = self._get_value(node, 1)
            data = self.sympy_data_[node.input[0]]
            if type(data) == list:
                if type(idx) == np.ndarray and len(idx.shape) == 1:
                    self.sympy_data_[node.output[0]] = [data[int(i)] for i in idx]
                else:
                    self.sympy_data_[node.output[0]] = data[int(idx)]
            else:
                assert idx == 0
                self.sympy_data_[node.output[0]] = data

    def _infer_GatherElements(self, node):
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                                                  indices_shape))

    def _infer_Loop(self, node):
        subgraph = get_attribute(node, 'body')
        assert len(subgraph.input) == len(node.input)
        for i, si in enumerate(subgraph.input):
            subgraph_name = si.name
            si.CopyFrom(self.known_vi_[node.input[i]])
            si.name = subgraph_name
        self._onnx_infer_subgraph(node, subgraph)
        # create a new symbolic dimension for iteration dependent dimension
        loop_iter_dim = self._new_symbolic_dim_from_output(node)
        num_loop_carried = len(node.input) - 2
        for i in range(len(node.output)):
            vi = self.known_vi_[node.output[i]]
            vi.CopyFrom(subgraph.output[i + 1]) # first subgraph output is condition, not in node output
            if i >= num_loop_carried:
                vi_dim = vi.type.tensor_type.shape.dim
                if len(vi_dim) > 0:
                    vi_dim[0].dim_param = loop_iter_dim
                else:
                    vi_dim.add().dim_param = loop_iter_dim
            vi.name = node.output[i]

    def _infer_MatMul(self, node):
        self._compute_matmul_shape(node)

    def _infer_MatMulInteger(self, node):
        self._compute_matmul_shape(node, onnx.TensorProto.INT32)

    def _infer_NonMaxSuppression(self, node):
        selected = self._new_symbolic_dim_from_output(node)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, [selected, 3]))

    def _infer_NonZero(self, node):
        input_rank = self._get_shape_rank(node, 0)
        # create a new symbolic dimension for NonZero output
        nz_len = self._new_symbolic_dim_from_output(node, 0, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], vi.type.tensor_type.elem_type, [input_rank, nz_len]))

    def _infer_OneHot(self, node):
        shape = self._get_shape(node, 0)
        axis = get_attribute(node, 'axis', -1)
        axis = handle_negative_axis(axis, len(shape)+1)
        new_shape = shape[:axis] + [self._new_symbolic_dim_from_output(node)] + shape[axis:]
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], self.known_vi_[node.input[2]].type.tensor_type.elem_type, new_shape))

    def _infer_Pad(self, node):
        if get_opset(self.out_mp_) <= 10:
            pads = get_attribute(node, 'pads')
        else:
            pads = self._get_value(node, 1)

        vi = self.known_vi_[node.output[0]]
        output_shape = get_shape_from_type_proto(vi.type)
        if len(output_shape) == 0 or None in output_shape:
            sympy_shape = self._get_sympy_shape(node, 0)
            rank = len(sympy_shape)
            assert len(pads) == 2*rank
            new_shape = [d + pad_up + pad_down for d, pad_up, pad_down in zip(sympy_shape, pads[:rank], pads[rank:])]
            self._update_computed_dims(new_shape)
            output_tp = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_tp, get_shape_from_sympy_shape(new_shape)))

    def _infer_Pool(self, node):
        sympy_shape = self._compute_conv_pool_shape(node)
        self._update_computed_dims(sympy_shape)
        for o in node.output:
            if not o:
                continue
            vi = self.known_vi_[o]
            vi.CopyFrom(helper.make_tensor_value_info(o, vi.type.tensor_type.elem_type, get_shape_from_sympy_shape(sympy_shape)))

    def _infer_Range(self, node):
        vi = self.known_vi_[node.output[0]]
        input_data = self._get_int_values(node)
        if all([i is not None for i in input_data]):
            start = as_scalar(input_data[0])
            limit = as_scalar(input_data[1])
            delta = as_scalar(input_data[2])
            new_shape = [sympy.Max(sympy.ceiling((limit - start)/delta), 0)]
        else:
            new_dim = self._new_symbolic_dim_from_output(node)
            new_shape = [self.symbolic_dims_[new_dim]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, get_shape_from_sympy_shape(new_shape)))

    def _infer_ReduceProd(self, node):
        axes = get_attribute(node, 'axes')
        keep_dims = get_attribute(node, 'keepdims')
        if keep_dims == 0 and axes == [0]:
            data = self._get_int_values(node)[0]
            if data is not None:
                self.sympy_data_[node.output[0]] = sympy_reduce_product(data)

    def _infer_Reshape(self, node):
        shape_value = self._get_value(node, 1)
        input_shape = self._get_shape(node, 0)
        input_sympy_shape = self._get_sympy_shape(node, 0)
        total = int(1)
        for d in input_sympy_shape:
            total = total * d
        new_sympy_shape = []
        deferred_dim_idx = -1
        non_deferred_size = int(1)
        for i, d in enumerate(shape_value):
            if type(d) == sympy.Symbol:
                new_sympy_shape.append(d)
            elif d == 0:
                new_sympy_shape.append(input_sympy_shape[i])
                non_deferred_size = non_deferred_size * input_sympy_shape[i]
            else:
                new_sympy_shape.append(d)
            if d == -1:
                deferred_dim_idx = i
            elif d != 0:
                non_deferred_size = non_deferred_size * d

        assert new_sympy_shape.count(-1) < 2
        if -1 in new_sympy_shape:
            new_dim = total // non_deferred_size
            new_sympy_shape[deferred_dim_idx] = new_dim
            self._update_computed_dims(new_sympy_shape)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  vi.type.tensor_type.elem_type,
                                                  get_shape_from_sympy_shape(new_sympy_shape)))
        self._pass_on_sympy_data(node)

    def _infer_Resize(self, node):
        assert get_opset(self.out_mp_) <= 10 # only support opset 10 Resize for now
        scales = self._try_get_value(node, 1)
        if scales is not None:
            input_sympy_shape = self._get_sympy_shape(node, 0)
            new_sympy_shape = [sympy.simplify(sympy.floor(d*s)) for d,s in zip(input_sympy_shape, scales)]
            self._update_computed_dims(new_sympy_shape)
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, get_shape_from_sympy_shape(new_sympy_shape)))

    def _infer_Scan(self, node):
        subgraph = get_attribute(node, 'body')
        num_scan_inputs = get_attribute(node, 'num_scan_inputs')
        scan_input_axes = get_attribute(node, 'scan_input_axes', [0]*num_scan_inputs)
        num_scan_states = len(node.input) - num_scan_inputs
        scan_input_axes = [handle_negative_axis(ax, self._get_shape_rank(node, i + num_scan_states)) for i, ax in enumerate(scan_input_axes)]
        for i, si in enumerate(subgraph.input):
            subgraph_name = si.name
            si.CopyFrom(self.known_vi_[node.input[i]])
            if i >= num_scan_states:
                scan_input_dim = si.type.tensor_type.shape.dim
                scan_input_dim.remove(scan_input_dim[scan_input_axes[i - num_scan_states]])
            si.name = subgraph_name
        self._onnx_infer_subgraph(node, subgraph)
        num_scan_outputs = len(node.output) - num_scan_states
        scan_output_axes = get_attribute(node, 'scan_output_axes', [0]*num_scan_outputs)
        scan_input_dim = get_shape_from_type_proto(self.known_vi_[node.input[-1]].type)[scan_input_axes[-1]]
        for i, o in enumerate(node.output):
            vi = self.known_vi_[o]
            if i >= num_scan_states:
                shape = get_shape_from_type_proto(subgraph.output[i].type)
                new_dim = handle_negative_axis(scan_output_axes[i - num_scan_states], len(shape) + 1)
                shape = shape[:new_dim] + [scan_input_dim] + shape[new_dim:]
                vi.CopyFrom(helper.make_tensor_value_info(o, subgraph.output[i].type.tensor_type.elem_type, shape))
            else:
                vi.CopyFrom(subgraph.output[i])
            vi.name = o

    def _infer_ScatterElements(self, node):
        data_shape = self._get_shape(node, 0)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                                                  data_shape))

    def _infer_Shape(self, node):
        self.sympy_data_[node.output[0]] = self._get_sympy_shape(node, 0)

    def _infer_Size(self, node):
        sympy_shape = self._get_sympy_shape(node, 0)
        self.sympy_data_[node.output[0]] = sympy_reduce_product(sympy_shape)
        self.known_vi_[node.output[0]].CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, []))

    def _infer_Slice(self, node):
        if get_opset(self.out_mp_) <= 9:
            axes = get_attribute(node, 'axes')
            starts = get_attribute(node, 'starts')
            ends = get_attribute(node, 'ends')
            steps = [1]*len(axes)
        else:
            starts = self._try_get_value(node, 1)
            ends = self._try_get_value(node, 2)
            axes = self._try_get_value(node, 3)
            steps = self._try_get_value(node, 4)
            if axes is None and not (starts is None and ends is None):
                axes = list(range(0, len(starts if starts is not None else ends)))
            if steps is None and not (starts is None and ends is None):
                steps = [1]*len(starts if starts is not None else ends)

        new_sympy_shape = self._get_sympy_shape(node, 0)
        if starts is None or ends is None:
            if axes is None:
                for i in range(len(new_sympy_shape)):
                    new_sympy_shape[i] = self._new_symbolic_dim_from_output(node,0,i)
            else:
                new_sympy_shape = get_shape_from_sympy_shape(new_sympy_shape)
                for i in axes:
                    new_sympy_shape[i] = self._new_symbolic_dim_from_output(node,0,i)
        else:
            for i,s,e,t in zip(axes, starts, ends, steps):
                idx = handle_negative_axis(i, len(new_sympy_shape))
                if is_literal(e):
                    if e >= self.int_max_:
                        e = new_sympy_shape[i]
                    elif e <= -self.int_max_:
                        e = 0 if step > 0 else -1
                    elif is_literal(new_sympy_shape[i]):
                        if e < 0:
                            e = e + new_sympy_shape[i]
                        e = min(e, new_sympy_shape[i])
                    else:
                        if e > 0:
                            e = sympy.Min(e, new_sympy_shape[i]) if e > 1 else e #special case for slicing first to make computation easier
                        else:
                            e = new_sympy_shape[i] + e
                else:
                    if is_literal(new_sympy_shape[i]):
                        e = sympy.Min(e, new_sympy_shape[i])
                    else:
                        try:
                            if e >= new_sympy_shape[i]:
                                e = new_sympy_shape[i]
                        except Exception:
                            print('Unable to determine if {} <= {}, treat as equal'.format(e, new_sympy_shape[i]))
                            e = new_sympy_shape[i]

                if is_literal(s) and int(s) < 0:
                    s = new_sympy_shape[i] + s

                new_sympy_shape[idx] = (e - s + t + (-1 if t > 0 else 1)) // t

            self._update_computed_dims(new_sympy_shape)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  vi.type.tensor_type.elem_type,
                                                  get_shape_from_sympy_shape(new_sympy_shape)))

        # handle sympy_data if needed, for slice in shape computation
        if node.input[0] in self.sympy_data_:
            assert [0] == axes
            assert len(starts) == 1
            assert len(ends) == 1
            self.sympy_data_[node.output[0]] = self.sympy_data_[node.input[0]][starts[0]:ends[0]]

    def _infer_Split(self, node):
        input_sympy_shape = self._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, 'axis', 0), len(input_sympy_shape))
        split = get_attribute(node, 'split')
        if not split:
            num_outputs = len(node.output)
            split = [input_sympy_shape[axis]/sympy.Integer(num_outputs)]*num_outputs
            self._update_computed_dims(split)
        else:
            split = [sympy.Integer(s) for s in split]

        for i_o in range(len(split)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[i_o],
                                                      self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                                                      get_shape_from_sympy_shape(input_sympy_shape[:axis] + [split[i_o]] + input_sympy_shape[axis+1:])))
            self.known_vi_[vi.name] = vi

    def _infer_Squeeze(self, node):
        self._pass_on_sympy_data(node)

    def _infer_Tile(self, node):
        repeats_value = self._get_value(node, 1)
        input_sympy_shape = self._get_sympy_shape(node, 0)
        new_shape = []
        for i,d in enumerate(input_sympy_shape):
            new_dim = d * repeats_value[i]
            new_shape.append(new_dim)
        self._update_computed_dims(new_shape)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  vi.type.tensor_type.elem_type,
                                                  get_shape_from_sympy_shape(new_shape)))

    def _infer_TopK(self, node):
        rank = self._get_shape_rank(node, 0)
        axis = handle_negative_axis(get_attribute(node, 'axis', -1), rank)
        new_shape = self._get_shape(node, 0)

        if get_opset(self.out_mp_) <= 9:
            k = get_attribute(node, 'k')
        else:
            k = self._get_int_values(node)[1]

        if k == None:
            k = self._new_symbolic_dim_from_output(node)
        else:
            k = as_scalar(k)

        if type(k) in [int, str]:
            new_shape[axis] = k
        else:
            new_sympy_shape = self._get_sympy_shape(node, 0)
            new_sympy_shape[axis] = k
            self._update_computed_dims(new_sympy_shape) # note that TopK dim could be computed in sympy_data, so need to update computed_dims when it enters shape
            new_shape = get_shape_from_sympy_shape(new_sympy_shape)

        for i_o in range(len(node.output)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[i_o], vi.type.tensor_type.elem_type, new_shape))

    def _infer_Unsqueeze(self, node):
        self._pass_on_sympy_data(node)

    def _infer_ZipMap(self, node):
        map_key_type = None
        if get_attribute(node, 'classlabels_int64s') is not None:
            map_key_type = onnx.TensorProto.INT64
        elif get_attribute(node, 'classlabels_strings') is not None:
            map_key_type = onnx.TensorProto.STRING

        assert map_key_type is not None
        new_vi = onnx.ValueInfoProto()
        new_vi.name = node.output[0]
        new_vi.type.sequence_type.elem_type.map_type.value_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        new_vi.type.sequence_type.elem_type.map_type.key_type = map_key_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(new_vi)

    def _infer_impl(self, in_mp):
        self.sympy_data_ = {}
        self.out_mp_.graph.ClearField('value_info')
        self._apply_suggested_merge_to_graph_input()
        input_symbols = set()
        for i in self.out_mp_.graph.input:
            input_symbols.update([d for d in get_shape_from_type_proto(i.type) if type(d) == str])
        for s in input_symbols:
            if s in self.suggested_merge_:
                s_merge = self.suggested_merge_[s]
                assert s_merge in self.symbolic_dims_
                self.symbolic_dims_[s] = self.symbolic_dims_[s_merge]
            else:
                self.symbolic_dims_[s] = sympy.Symbol(s, integer=True)

        # create a temporary ModelProto for single node inference
        # note that we remove initializer to have faster inference
        # for tensor ops like Reshape/Tile/Expand that read initializer, we need to do sympy computation based inference anyways
        self.tmp_mp_ = onnx.ModelProto()
        self.tmp_mp_.CopyFrom(self.out_mp_)
        self.tmp_mp_.graph.ClearField('initializer')

        for node in self.out_mp_.graph.node:
            assert all([i in self.known_vi_ for i in node.input if i])
            self._onnx_infer_single_node(node)
            if node.op_type in self.dispatcher_:
                self.dispatcher_[node.op_type](node)

            if self.verbose_ > 2:
                print(node.op_type + ': ' + node.name)
            for i_o in range(len(node.output)):
                out_type = self.known_vi_[node.output[i_o]].type
                out_type_kind = out_type.WhichOneof('value')
                # only TensorProto and SparseTensorProto have shape
                if out_type_kind != 'tensor_type' and out_type_kind != 'sparse_tensor_type':
                    continue
                out_shape = get_shape_from_type_proto(self.known_vi_[node.output[i_o]].type)
                out_type_undefined = out_type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED
                if self.verbose_ > 2:
                    print('  {}: {} {}'.format(node.output[i_o], str(out_shape), self.known_vi_[node.output[i_o]].type.tensor_type.elem_type))
                    if node.output[i_o] in self.sympy_data_:
                        print('  Sympy Data: ' + str(self.sympy_data_[node.output[i_o]]))
                if None in out_shape or out_type_undefined:
                    if self.auto_merge_:
                        if node.op_type in ['Add', 'Sub', 'Mul', 'Div', 'MatMul', 'MatMulInteger', 'MatMulInteger16', 'Concat', 'Where', 'Sum']:
                            shapes = [self._get_shape(node, i) for i in range(len(node.input))]
                            if node.op_type in ['MatMul', 'MatMulInteger', 'MatMulInteger16']:
                                # only support auto merge for MatMul for dim < rank-2 when rank > 2
                                assert len(shapes[0]) > 2 and dim_idx[0] < len(shapes[0]) - 2
                                assert len(shapes[1]) > 2 and dim_idx[1] < len(shapes[1]) - 2
                        elif node.op_type == 'Expand':
                            # auto merge for cases like Expand([min(batch, 1), min(seq, 512)], [batch, seq])
                            shapes = [self._get_shape(node, 0), self._get_value(node, 1)]
                        else:
                            shapes = []

                        if shapes:
                            for idx in range(len(out_shape)):
                                if out_shape[idx] is not None:
                                    continue
                                dim_idx = [len(s) - len(out_shape) + idx for s in shapes]
                                assert all([d >= 0 for d in dim_idx])
                                self._add_suggested_merge([s[i] if is_literal(s[i]) else str(s[i]) for s, i in zip(shapes, dim_idx)])
                            self.run_ = True
                        else:
                            self.run_ = False
                    else:
                        self.run_ = False

                    if self.verbose_ > 0 or not self.auto_merge_ or out_type_undefined:
                        print('Stopping at incomplete shape inference at ' + node.op_type + ': ' + node.name)
                        print(node)
                        print('node inputs:')
                        for i in node.input:
                            print(self.known_vi_[i])
                        print('node outputs:')
                        for o in node.output:
                            print(self.known_vi_[o])
                        if self.auto_merge_ and not out_type_undefined:
                            print('Merging: ' + str(self.suggested_merge_))
                    return False

        self.run_ = False
        return True

    def _update_output_from_vi(self):
        for output in self.out_mp_.graph.output:
            if output.name in self.known_vi_:
                output.CopyFrom(self.known_vi_[output.name])

    @staticmethod
    def infer_shapes(input_model, output_model, int_max=2**31 - 1, auto_merge=False, verbose=0):
        in_mp = onnx.load(input_model)
        symbolic_shape_inference = SymbolicShapeInference(int_max, auto_merge, verbose)
        all_shapes_inferred = False
        symbolic_shape_inference._preprocess(in_mp)
        while symbolic_shape_inference.run_:
            all_shapes_inferred = symbolic_shape_inference._infer_impl(in_mp)
        symbolic_shape_inference._update_output_from_vi()
        onnx.save(symbolic_shape_inference.out_mp_, output_model)
        if not all_shapes_inferred:
            sys.exit(1)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, help='The input model file')
  parser.add_argument('--output', required=True, help='The input model file')
  parser.add_argument('--auto_merge', help='Automatically merge symbolic dims when confliction happens', action='store_true', default=False)
  parser.add_argument('--int_max', help='maximum value for integer to be treated as boundless for ops like slice', type=int, default=2**31 - 1)
  parser.add_argument('--verbose', help='Prints detailed logs of inference, 0: turn off, 1: warnings, 3: detailed', type=int, default=0)
  return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print('input model: ' + args.input)
    print('output model ' + args.output)
    print('Doing symbolic shape inference...')
    out_mp = SymbolicShapeInference.infer_shapes(args.input, args.output, args.int_max, args.auto_merge, args.verbose)
    print('Done!')
