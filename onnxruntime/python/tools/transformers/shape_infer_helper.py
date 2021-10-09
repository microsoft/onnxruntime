#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import os
import sys
import onnx

# In ORT Package the symbolic_shape_infer.py is in ../tools
file_path = os.path.dirname(__file__)
if os.path.exists(os.path.join(file_path, "../tools/symbolic_shape_infer.py")):
    sys.path.append(os.path.join(file_path, '../tools'))
else:
    sys.path.append(os.path.join(file_path, '..'))

from symbolic_shape_infer import SymbolicShapeInference, get_shape_from_type_proto, sympy


class SymbolicShapeInferenceHelper(SymbolicShapeInference):
    def __init__(self, model, verbose=0, int_max=2**31 - 1, auto_merge=True, guess_output_rank=False):
        super().__init__(int_max, auto_merge, guess_output_rank, verbose)
        self.model_ = onnx.ModelProto()
        self.model_.CopyFrom(model)
        self.all_shapes_inferred_ = False
        self.inferred_ = False

    # The goal is to remove dynamic_axis_mapping
    def infer(self, dynamic_axis_mapping):
        if self.inferred_:
            return self.all_shapes_inferred_

        self.dynamic_axis_mapping_ = dynamic_axis_mapping  # e.g {"batch_size" : 4, "seq_len" :7}

        self._preprocess(self.model_)
        while self.run_:
            self.all_shapes_inferred_ = self._infer_impl()

        self.inferred_ = True
        return self.all_shapes_inferred_

    # override _preprocess() to avoid unnecessary model copy since ctor copies the model
    def _preprocess(self, in_mp):
        self.out_mp_ = in_mp
        self.graph_inputs_ = dict([(i.name, i) for i in list(self.out_mp_.graph.input)])
        self.initializers_ = dict([(i.name, i) for i in self.out_mp_.graph.initializer])
        self.known_vi_ = dict([(i.name, i) for i in list(self.out_mp_.graph.input)])
        self.known_vi_.update(
            dict([(i.name, onnx.helper.make_tensor_value_info(i.name, i.data_type, list(i.dims)))
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
        assert (self.all_shapes_inferred_ == True)
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
        assert (self.all_shapes_inferred_ == True)
        shape = self.get_edge_shape(edge)
        shape_other = self.get_edge_shape(edge_other)
        if shape is None or shape_other is None:
            raise Exception("At least one shape is missed for edges to compare")
        return shape == shape_other
