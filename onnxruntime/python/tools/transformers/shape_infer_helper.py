import os
import sys

# In ORT Package the symbolic_shape_infer.py is in ../tools
if os.path.exists(os.path.join(os.path.dirname(__file__), "../tools/symbolic_shape_infer.py")):
    sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from symbolic_shape_infer import *

import onnx


class SymbolicShapeInferenceHelper(SymbolicShapeInference):
    def __init__(self, model, verbose=0, int_max=2**31 - 1, auto_merge=True, guess_output_rank=False):
        super().__init__(int_max, auto_merge, guess_output_rank, verbose)
        self.model_ = onnx.ModelProto()
        self.model_.CopyFrom(model)

        self.dispatcher_['Gelu'] = self._infer_Gelu
        self.dispatcher_['FastGelu'] = self._infer_FastGelu
        self.dispatcher_['BiasGelu'] = self._infer_BiasGelu
        self.dispatcher_['Attention'] = self._infer_Attention
        self.dispatcher_['LongformerAttention'] = self._infer_LongformerAttention
        self.dispatcher_['LayerNormalization'] = self._infer_LayerNormalization
        self.dispatcher_['SkipLayerNormalization'] = self._infer_SkipLayerNormalization

        self.all_shapes_inferred_ = False
        self.inferred_ = False

    # The goal is to remove dynamic_axis_mapping
    def infer(self, dynamic_axis_mapping={"batch_size": 4, "seq_len": 7}):
        if self.inferred_:
            return self.all_shapes_inferred_

        self.dynamic_axis_mapping_ = dynamic_axis_mapping  # e.g {"batch_size" : 4, "seq_len" :7}

        self._preprocess(self.model_)
        while self.run_:
            self.all_shapes_inferred_ = self._infer_impl()

        self.inferred_ = True
        return self.all_shapes_inferred_

    def _infer_Gelu(self, node):
        self._infer_BertOp_SameInputOutputShape_Impl(node)

    def _infer_FastGelu(self, node):
        self._infer_BertOp_SameInputOutputShape_Impl(node)

    def _infer_BiasGelu(self, node):
        self._infer_BertOp_SameInputOutputShape_Impl(node)

    def _infer_Attention(self, node):
        self._infer_BertOp_SameInputOutputShape_Impl(node)

    def _infer_LongformerAttention(self, node):
        self._infer_BertOp_SameInputOutputShape_Impl(node)

    def _infer_LayerNormalization(self, node):
        self._infer_BertOp_SameInputOutputShape_Impl(node)

    def _infer_SkipLayerNormalization(self, node):
        self._infer_BertOp_SameInputOutputShape_Impl(node)

    def _infer_BertOp_SameInputOutputShape_Impl(self, node):
        shape = self._get_shape(node, 0)
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, shape))

    # override _preprocess() to avoid unnecessary model copy since ctor copies the model
    def _preprocess(self, in_mp):
        self.out_mp_ = in_mp
        self.initializers_ = dict([(i.name, i) for i in self.out_mp_.graph.initializer])
        self.known_vi_ = dict([(i.name, i) for i in list(self.out_mp_.graph.input)])
        self.known_vi_.update(
            dict([(i.name, helper.make_tensor_value_info(i.name, i.data_type, list(i.dims)))
                  for i in self.out_mp_.graph.initializer]))

    # Override _get_sympy_shape() in symbolic_shape_infer.py to ensure shape inference by giving the actual value of dynamic axis
    def _get_sympy_shape(self, node, idx):
        sympy_shape = []
        for d in self._get_shape(node, idx):
            if type(d) == str:
                if d in self.dynamic_axis_mapping_.keys():
                    sympy_shape.append(self.dynamic_axis_mapping_[d])
                elif d in self.symbolic_dims_:
                    sympy_shape.append(self.symbolic_dims_[d])
                else:
                    sympy_shape.append(sympy.Symbol(d, integer=True))
            else:
                assert None != d
                sympy_shape.append(d)
        return sympy_shape

    def get_edge_shape(self, edge):
        if edge not in self.known_vi_:
            print("Cannot retrive the shape of " + str(edge))
            return None
        type_proto = self.known_vi_[edge].type
        shape = get_shape_from_type_proto(type_proto)
        for i in range(len(shape)):
            d = shape[i]
            if type(d) == str and d in self.dynamic_axis_mapping_.keys():
                shape[i] = self.dynamic_axis_mapping_[d]
        return shape

    def compare_shape(self, edge, edge_other):
        shape = self.get_edge_shape(edge)
        shape_other = self.get_edge_shape(edge_other)
        if shape is None or shape_other is None:
            raise Exception("At least one shape is missed for edges to compare")
        return shape == shape_other
