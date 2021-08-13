#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from logging import getLogger
from onnx import helper, numpy_helper, TensorProto
from onnx_model import OnnxModel
from fusion_base import Fusion
import numpy as np

logger = getLogger(__name__)


class FusionReshape(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Reshape", "Reshape")

    def replace_reshape_node(self, shape, reshape_node, concat_node):
        shape_value = np.asarray(shape, dtype=np.int64)
        constant_shape_name = self.model.create_node_name('Constant', 'constant_shape')
        new_node = helper.make_node('Constant',
                                    inputs=[],
                                    outputs=[constant_shape_name],
                                    value=helper.make_tensor(name='const_tensor',
                                                             data_type=TensorProto.INT64,
                                                             dims=shape_value.shape,
                                                             vals=bytes(shape_value),
                                                             raw=True))
        reshape_node.input[1] = constant_shape_name
        reshape_node.name = self.model.create_node_name('Reshape', 'Reshape_Fuse')
        self.nodes_to_remove.extend([concat_node])
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    def fuse(self, reshape_node, input_name_to_nodes, output_name_to_node):
        if reshape_node.input[1] not in output_name_to_node:
            return

        concat_node = output_name_to_node[reshape_node.input[1]]
        if concat_node.op_type != 'Concat' or len(concat_node.input) < 3 or len(concat_node.input) > 4:
            return

        path0 = self.model.match_parent_path(concat_node, ['Unsqueeze', 'Gather', 'Shape'], [0, 0, 0],
                                             output_name_to_node)
        if path0 is None:
            return

        (unsqueeze_0, gather_0, shape_0) = path0

        shape = []
        gather_value = self.model.get_constant_value(gather_0.input[1])
        if gather_value == 0:
            shape.append(0)

        path1 = self.model.match_parent_path(concat_node, ['Unsqueeze', 'Gather', 'Shape'], [1, 0, 0],
                                             output_name_to_node)
        if path1 is None:
            # Adjust for Bart
            if len(concat_node.input) == 4:
                input_1_proto = self.model.get_initializer(concat_node.input[1])
                input_2_proto = self.model.get_initializer(concat_node.input[2])
                input_3_proto = self.model.get_initializer(concat_node.input[3])
                if input_1_proto is None or input_2_proto is None or input_3_proto is None:
                    return
                input_1 = numpy_helper.to_array(input_1_proto)
                input_2 = numpy_helper.to_array(input_2_proto)
                input_3 = numpy_helper.to_array(input_3_proto)
                if input_1[0] == -1 and input_2[0] > 0 and input_3[0] > 0:
                    shape.extend(input_1)
                    shape.extend(input_2)
                    shape.extend(input_3)

                    gemm_path = self.model.match_parent_path(reshape_node, ['Add', 'MatMul'], [0, 1],
                                                             output_name_to_node)
                    if gemm_path is None:
                        return
                    top_matmul = gemm_path[-1]
                    root_input = top_matmul.input[0]

                    if shape_0.input[0] != root_input:
                        return

                    self.replace_reshape_node(shape, reshape_node, concat_node)
            return

        (unsqueeze_1, gather_1, shape_1) = path1

        gather_value = self.model.get_constant_value(gather_1.input[1])
        if gather_value == 1:
            shape.append(0)

        if len(shape) != 2:
            return

        path2 = []
        path3 = []
        shape_nodes = [shape_0, shape_1]
        if len(concat_node.input) == 3 and self.model.get_initializer(concat_node.input[2]) is None:
            path2 = self.model.match_parent_path(concat_node, ['Unsqueeze', 'Mul', 'Gather', 'Shape'], [2, 0, 0, 0],
                                                 output_name_to_node)
            if path2 is None:
                path2 = self.model.match_parent_path(
                    concat_node, ['Unsqueeze', 'Mul', 'Squeeze', 'Slice', 'Shape'], [2, 0, 0, 0, 0],
                    output_name_to_node)  # GPT2 exported by PyTorch 1.4 with opset_version=11
                if path2 is None:
                    return

            path3 = self.model.match_parent_path(concat_node, ['Unsqueeze', 'Mul', 'Gather', 'Shape'], [2, 0, 1, 0],
                                                 output_name_to_node)
            if path3 is None:
                path3 = self.model.match_parent_path(
                    concat_node, ['Unsqueeze', 'Mul', 'Squeeze', 'Slice', 'Shape'], [2, 0, 1, 0, 0],
                    output_name_to_node)  # GPT2 exported by PyTorch 1.4 with opset_version=11
                if path3 is None:
                    return

            shape_nodes.extend([path2[-1], path3[-1]])
            shape.append(-1)
        elif (len(concat_node.input) > 2):
            concat_2 = self.model.get_initializer(concat_node.input[2])
            if concat_2 is None:
                return
            concat_value = numpy_helper.to_array(concat_2)
            if isinstance(concat_value, list):
                shape.extend(concat_value)
            else:
                shape.append(concat_value)

        if len(concat_node.input) == 4 and self.model.get_initializer(concat_node.input[3]) is None:
            if -1 in shape:
                return

            path2 = self.model.match_parent_path(concat_node, ['Unsqueeze', 'Div', 'Gather', 'Shape'], [3, 0, 0, 0],
                                                 output_name_to_node)
            if path2 is None:
                path2 = self.model.match_parent_path(
                    concat_node, ['Unsqueeze', 'Div', 'Squeeze', 'Slice', 'Shape'], [3, 0, 0, 0, 0],
                    output_name_to_node)  # GPT2 exported by PyTorch 1.4 with opset_version=11
                if path2 is None:
                    return
            shape_nodes.extend([path2[-1]])
            shape.append(-1)
        elif (len(concat_node.input) > 3):
            concat_3 = self.model.get_initializer(concat_node.input[3])
            if concat_3 is None:
                return

            concat_value = numpy_helper.to_array(concat_3)
            if isinstance(concat_value, list):
                shape.extend(concat_value)
            else:
                shape.append(concat_value)

        root_input = reshape_node.input[0]
        root_input_1 = None
        ok_to_delete_path = True
        # Adjust for Bart
        gemm_path = self.model.match_parent_path(reshape_node, ['Mul', 'Add', 'MatMul'], [0, 0, 1], output_name_to_node)
        if gemm_path is not None:
            top_matmul = gemm_path[-1]
            root_input_1 = top_matmul.input[0]

        same_shape_input = True
        for shape_node in shape_nodes:
            if shape_node.input[0] != root_input:
                same_shape_input = False
            if root_input_1 is not None and shape_node.input[0] == root_input_1:
                same_shape_input = True
                ok_to_delete_path = False

        if not same_shape_input:
            return

        self.replace_reshape_node(shape, reshape_node, concat_node)

        # TODO: generic deletion from bottom-up
        if ok_to_delete_path:
            self.nodes_to_remove.extend(path0)
            self.nodes_to_remove.extend(path1)
            self.nodes_to_remove.extend(path2)
            self.nodes_to_remove.extend(path3)
