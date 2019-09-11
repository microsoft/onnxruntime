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
    return [int(i) if is_literal(i) or i.is_number else str(i) for i in sympy_shape]

def is_literal(dim):
    return type(dim) in [int, np.int64, sympy.Integer]

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

class SymbolicShapeInference:
    def __init__(self, auto_merge, verbose):
        self.dispatcher_ = {
            'Cast'              : self._infer_Cast,
            'CategoryMapper'    : self._infer_CategoryMapper,
            'Compress'          : self._infer_Compress,
            'Concat'            : self._infer_Concat,
            'ConstantOfShape'   : self._infer_ConstantOfShape,
            'Expand'            : self._infer_Expand,
            'Gather'            : self._infer_Gather,
            'Min'               : self._infer_Min,
            'Mul'               : self._infer_Mul,
            'NonMaxSuppression' : self._infer_NonMaxSuppression,
            'NonZero'           : self._infer_NonZero,
            'Reshape'           : self._infer_Reshape,
            'Shape'             : self._infer_Shape,
            'Slice'             : self._infer_Slice,
            'Split'             : self._infer_Split,
            'Squeeze'           : self._infer_Squeeze,
            'Tile'              : self._infer_Tile,
            'TopK'              : self._infer_TopK,
            'Unsqueeze'         : self._infer_Unsqueeze}
        self.suggested_merge_ = {}
        self.run_ = True
        self.auto_merge_ = auto_merge
        self.verbose_ = verbose

    def _set_input_model(self, in_mp):
        self._preprocess(in_mp)
        self.initializers_ = dict([(i.name, i) for i in self.out_mp_.graph.initializer])
        self.known_vi_ = dict([(i.name, i) for i in list(self.out_mp_.graph.input)])
        self.known_vi_.update(dict([(i.name, helper.make_tensor_value_info(i.name, i.data_type, list(i.dims))) for i in self.out_mp_.graph.initializer]))
        self.sympy_data_ = {}
        self.dynamic_dims_ = {} # new symbolic dims from some ops' dynamic output, i.e. NonZero
        self.computed_dims_ = {}
        # create a temporary ModelProto for single node inference
        # note that we remove initializer to have faster inference
        # for tensor ops like Reshape/Tile/Expand that read initializer, we need to do sympy computation based inference anyways
        self.tmp_mp_ = onnx.ModelProto()
        self.tmp_mp_.CopyFrom(self.out_mp_)
        self.tmp_mp_.graph.ClearField('initializer')

    def _add_suggested_merge(self, d1, d2):
        existing = set(self.suggested_merge_.values())
        if d1 in existing and d2 in existing:
            self.suggested_merge_[d2] = d1
            for k, v in self.suggested_merge_.items():
                if v == d2:
                    self.suggested_merge_[k] = d1
        elif d1 in existing:
            self.suggested_merge_[d2] = d1
        elif d2 in existing:
            self.suggested_merge_[d1] = d2
        elif d1 in self.suggested_merge_:
            self.suggested_merge_[d2] = self.suggested_merge_[d1]
        elif d2 in self.suggested_merge_:
            self.suggested_merge_[d1] = self.suggested_merge_[d2]
        else:
            self.suggested_merge_[d1] = d2

    def _apply_suggested_merge(self):
        if not self.suggested_merge_:
            return
        for i in self.out_mp_.graph.input:
            for d in i.type.tensor_type.shape.dim:
                if d.dim_param in self.suggested_merge_:
                    d.dim_param = self.suggested_merge_[d.dim_param]

    def _preprocess(self, in_mp):
        out_mp = onnx.ModelProto()
        out_mp.CopyFrom(in_mp)
        out_mp.graph.ClearField('value_info')
        out_mp.graph.ClearField('node')
        self.out_mp_ = out_mp

        self._apply_suggested_merge()

        # constant op -> initializer, topological sort
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

    def _merge_dynamic_dims(self, d1, d2):
        if d1 in self.suggested_merge_:
            return self.suggested_merge_[d1]
        if d2 in self.suggested_merge_:
            return self.suggested_merge_[d2]
        if not d1 in self.dynamic_dims_ and not d2 in self.dynamic_dims_:
            return None
        if d1 in self.dynamic_dims_:
            if self.dynamic_dims_[d1] == None:
                self.dynamic_dims_[d1] = d2
            else:
                assert self.dynamic_dims_[d1] == d2
            return d2
        return _merge_dynamic_dims(d2, d1)

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
            elif not dim1 == dim2:
                new_dim = self._merge_dynamic_dims(dim1, dim2)
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
                sym_dim = self.computed_dims_[d] if d in self.computed_dims_ else sympy.Symbol(d, integer=True)
                sympy_shape.append(sym_dim)
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
        for new_dim in new_sympy_shape:
            if not is_literal(new_dim) and not type(new_dim) == str: # add new_dim if it's a computational expression
                if not str(new_dim) in self.computed_dims_:
                    self.computed_dims_[str(new_dim)] = new_dim

    def _onnx_infer_single_node(self, node):
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
            vi.CopyFrom(self.tmp_mp_.graph.output[i_o])
            self.known_vi_[vi.name] = vi

    def _get_int_values(self, node):
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
        return values

    def _compute_on_sympy_data(self, node, op_func):
        assert len(node.output) == 1
        values = self._get_int_values(node)
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
        compress_len = node.output[0] + '_compress_len'
        self.dynamic_dims_[compress_len] = None # set to None as it may merge with other symbolic dims later
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
            assert all([v is not None for v in values])
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
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, get_shape_from_sympy_shape(sympy_shape)))

    def _infer_ConstantOfShape(self, node):
        sympy_shape = self._get_value(node, 0)
        if sympy_shape:
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                      vi.type.tensor_type.elem_type,
                                                      [int(i) if is_literal(i) else str(i) for i in sympy_shape]))

    def _infer_Expand(self, node):
        expand_to_shape = self._try_get_value(node, 1)
        if expand_to_shape is not None:
            input_shape = self._get_shape(node, 0)
            target_shape = get_shape_from_sympy_shape(expand_to_shape)
            new_shape = self._broadcast_shapes(input_shape, target_shape)
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, new_shape))

    def _infer_Gather(self, node):
        axis = get_attribute(node, 'axis', 0)
        data_shape = self._get_shape(node, 0)
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  vi.type.tensor_type.elem_type,
                                                  data_shape[:axis] + indices_shape + data_shape[axis+1:]))
        if node.input[0] in self.sympy_data_:
            assert 0 == get_attribute(node, 'axis') # only handle 1D sympy compute
            idx = int(self._get_value(node, 1))
            data = self.sympy_data_[node.input[0]]
            if type(data) == list:
                self.sympy_data_[node.output[0]] = data[idx]
            else:
                assert idx == 0
                self.sympy_data_[node.output[0]] = data

    def _infer_Min(self, node):
        self._compute_on_sympy_data(node, lambda l: sympy.Min(l[0], l[1]))

    def _infer_Mul(self, node):
        self._compute_on_sympy_data(node, lambda l: l[0] * l[1])

    def _infer_NonMaxSuppression(self, node):
        selected = node.output[0] + '_num_selected'
        self.dynamic_dims_[selected] = None # set to None as it may merge with other symbolic dims later
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, [selected, 3]))

    def _infer_NonZero(self, node):
        input_rank = self._get_shape_rank(node, 0)
        # create a new symbolic dimension for NonZero output
        nz_len = node.output[0] + '_nz_len'
        self.dynamic_dims_[nz_len] = None # set to None as it may merge with other symbolic dims later
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], vi.type.tensor_type.elem_type, [input_rank, nz_len]))

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
            new_dim = total / non_deferred_size
            new_sympy_shape[deferred_dim_idx] = new_dim
            self._update_computed_dims(new_sympy_shape)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  vi.type.tensor_type.elem_type,
                                                  get_shape_from_sympy_shape(new_sympy_shape)))
        self._pass_on_sympy_data(node)

    def _infer_Shape(self, node):
        self.sympy_data_[node.output[0]] = self._get_sympy_shape(node, 0)

    def _infer_Slice(self, node):
        if get_opset(self.out_mp_) <= 9:
            axes = get_attribute(node, 'axes')
            starts = get_attribute(node, 'starts')
            ends = get_attribute(node, 'ends')
            steps = [1]*len(axes)
        else:
            starts = self._get_value(node, 1)
            ends = self._get_value(node, 2)
            assert starts is not None and ends is not None
            axes = self._try_get_value(node, 3)
            steps = self._try_get_value(node, 4)
            if axes is None:
                axes = list(range(0, len(starts)))
            if steps is None:
                steps = [1]*len(starts)

        new_shape = self._get_sympy_shape(node, 0)
        for i,s,e,t in zip(axes, starts, ends, steps):
            # TODO: handle step
            assert t == 1
            idx = handle_negative_axis(i, len(new_shape))
            if is_literal(e):
                if e >= int(2 ** 31 - 1): # max value of int32
                    e = new_shape[i]
                elif is_literal(new_shape[i]):
                    e = min(e, new_shape[i])
                else:
                    if e > 0:
                        e = sympy.Min(e, new_shape[i])
                    else:
                        e = new_shape[i] + e
            else:
                if is_literal(new_shape[i]):
                    e = sympy.Min(e, new_shape[i])
                else:
                    try:
                        if e >= new_shape[i]:
                            e = new_shape[i]
                    except Exception:
                        print('Unable to determine if {} <= {}, treat as equal'.format(e, new_shape[i]))
                        e = new_shape[i]

            if is_literal(s) and int(s) < 0:
                s = new_shape[i] + s

            new_shape[idx] = e - s

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0],
                                                  vi.type.tensor_type.elem_type,
                                                  get_shape_from_sympy_shape(new_shape)))
        if node.input[0] in self.sympy_data_:
            assert [0] == axes
            assert len(starts) == 1
            assert len(ends) == 1
            self.sympy_data_[node.output[0]] = self.sympy_data_[node.input[0]][starts[0]:ends[0]]

    def _infer_Split(self, node):
        shape = self._get_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, 'axis', 0), len(shape))
        split = get_attribute(node, 'split')
        if not split:
            num_outputs = len(node.output)
            split = [int(shape[axis]/num_outputs)]*num_outputs
        for i_o in range(len(split)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[i_o],
                                                      self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                                                      shape[:axis] + [split[i_o]] + shape[axis+1:]))
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
        axis = handle_negative_axis(get_attribute(node, 'axis'), rank)
        new_shape = self._get_sympy_shape(node, 0)

        if get_opset(self.out_mp_) <= 9:
            k = get_attribute(node, 'k')
        else:
            k = self._try_get_value(node, 1)

        if k == None:
            k = sympy.Symbol(node.output[0] + '_num_topK', integer=True)
            self.dynamic_dims_[k] = None # set to None as it may merge with other symbolic dims later

        new_shape[axis] = k

        for i_o in range(len(node.output)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[i_o], vi.type.tensor_type.elem_type, get_shape_from_sympy_shape(new_shape)))

    def _infer_Unsqueeze(self, node):
        self._pass_on_sympy_data(node)

    def _infer_impl(self, in_mp):
        self._set_input_model(in_mp)
        for node in self.out_mp_.graph.node:
            assert all([i in self.known_vi_ for i in node.input if i])
            self._onnx_infer_single_node(node)
            if node.op_type in self.dispatcher_:
                self.dispatcher_[node.op_type](node)

            if self.verbose_ > 2:
                print(node.op_type + ': ' + node.name)
            for i_o in range(len(node.output)):
                out_shape = get_shape_from_type_proto(self.known_vi_[node.output[i_o]].type)
                if self.verbose_ > 2:
                    print('  {}: {} {}'.format(node.output[i_o], str(out_shape), self.known_vi_[node.output[i_o]].type.tensor_type.elem_type))
                    if node.output[i_o] in self.sympy_data_:
                        print('  Sympy Data: ' + str(self.sympy_data_[node.output[i_o]]))
                if None in out_shape:
                    if self.auto_merge_:
                        if node.op_type in ['Add', 'Sub', 'Mul', 'Div', 'MatMul']:
                            idx = out_shape.index(None)
                            shapes = [self._get_shape(node, i) for i in range(2)]
                            dim_idx = [len(s) - len(out_shape) + idx for s in shapes]
                            assert dim_idx[0] >= 0 and dim_idx[1] >= 0
                            if node.op_type == 'MatMul':
                                # only support auto merge for MatMul for dim < rank-2 when rank > 2
                                assert len(shapes[0]) > 2 and dim_idx[0] < len(shapes[0]) - 2
                                assert len(shapes[1]) > 2 and dim_idx[1] < len(shapes[1]) - 2
                            dim0, dim1 = [s[i] for s, i in zip(shapes, dim_idx)]
                            assert type(dim0) == str and type(dim1) == str
                            if dim0 in self.computed_dims_ and dim1 in self.computed_dims_:
                                dd = self.computed_dims_[dim0]/self.computed_dims_[dim1]
                                dim0 = dd.args[0]
                                dim1 = dd.args[1].args[0]
                                assert type(dim0) == sympy.Symbol
                                assert type(dim1) == sympy.Symbol
                                self._add_suggested_merge(str(dim1), str(dim0))
                            else:
                                assert not dim0 in self.computed_dims_ or type(self.computed_dims_[dim0]) == sympy.Symbol
                                assert not dim1 in self.computed_dims_ or type(self.computed_dims_[dim1]) == sympy.Symbol
                                self._add_suggested_merge(str(dim1), str(dim0))
                            self.run_ = True
                        elif node.op_type == 'Expand':
                            # auto merge for cases like Expand([min(batch, 1), min(seq, 512)], [batch, seq])
                            input_shape = self._get_shape(node, 0)
                            expand_shape = self._get_value(node, 1)
                            for reverse_idx in range(min(len(input_shape), len(expand_shape))):
                                dim0 = input_shape[len(input_shape) - 1 - reverse_idx]
                                dim1 = expand_shape[len(expand_shape) - 1 - reverse_idx]
                                if any([type(d) == str for d in [dim0, dim1]]):
                                    if type(dim0) == str:
                                        self._add_suggested_merge(str(dim0), str(dim1))
                                    else:
                                        self._add_suggested_merge(str(dim1), str(dim0))
                            self.run_ = True
                        else:
                            self.run_ = False
                    else:
                        self.run_ = False

                    if self.verbose_ > 0 or not self.auto_merge_:
                        print('Stopping at incomplete shape inference at ' + node.op_type + ': ' + node.name)
                        for o in node.output:
                            print(self.known_vi_[o])
                        if self.auto_merge_:
                            print('Merging: ' + str(self.suggested_merge_))
                    return False

        self.run_ = False
        return True

    def _replace_dynamic_dims(self):
        # replace all self.dynamic_dims_ if there's a matching
        for vi in list(self.out_mp_.graph.value_info) + list(self.out_mp_.graph.output):
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_param in self.dynamic_dims_:
                    v = self.dynamic_dims_[d.dim_param]
                    if v != None:
                        if type(v) == str:
                            d.dim_param = v
                        else:
                            assert type(v) == int
                            d.dim_value = v

    def _update_output_from_vi(self):
        for output in self.out_mp_.graph.output:
            if output.name in self.known_vi_:
                output.CopyFrom(self.known_vi_[output.name])

    @staticmethod
    def infer_shapes(input_model, output_model, auto_merge=False, verbose=0):
        in_mp = onnx.load(input_model)
        symbolic_shape_inference = SymbolicShapeInference(auto_merge, verbose)
        all_shapes_inferred = False
        while symbolic_shape_inference.run_:
            all_shapes_inferred = symbolic_shape_inference._infer_impl(in_mp)
            symbolic_shape_inference._replace_dynamic_dims()
        symbolic_shape_inference._update_output_from_vi()
        onnx.save(symbolic_shape_inference.out_mp_, output_model)
        if not all_shapes_inferred:
            sys.exit(1)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', help='The input model file')
  parser.add_argument('--output', help='The input model file')
  parser.add_argument('--auto_merge', help='Automatically merge symbolic dims when confliction happens', action='store_true', default=False)
  parser.add_argument('--verbose', help='Prints detailed logs of inference, 0: turn off, 1: warnings, 3: detailed', type=int, default=0)
  return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print('input model: ' + args.input)
    print('output model ' + args.output)
    print('Doing symbolic shape inference...')
    out_mp = SymbolicShapeInference.infer_shapes(args.input, args.output, args.auto_merge, args.verbose)
    print('Done!')