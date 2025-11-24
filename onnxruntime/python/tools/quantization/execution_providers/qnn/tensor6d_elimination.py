# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Define 6D tensor elimination."""

import logging

import onnx

from ... import fusions, onnx_model
from ...fusions import Fusion


def get_tensor_shape(model: onnx_model.ONNXModel, tensor_name: str) -> list[int] | None:
    """Get shape for given tensor name."""
    tensor_type = model.get_tensor_type(tensor_name)
    if not tensor_type:
        return None

    tensor_shape = Fusion.tensor_shape_to_list(tensor_type)
    if not tensor_shape:
        return None

    return tensor_shape


def is_dynamic(shape: onnx.TensorShapeProto) -> bool:
    """Check if shape has any dynamic dimensions."""
    return any(dim.dim_value <= 0 for dim in shape.dim)


def is_6d_supportable(
    node: onnx.NodeProto,
    model: onnx_model.ONNXModel,
    input_name_to_nodes: dict[str, list[onnx.NodeProto]],
    output_name_to_node: dict[str, onnx.NodeProto],
) -> bool:
    """Check if we can eliminate 6D tensors in inputs/outputs to `node`."""
    # We insert pre-reshape after input buffer / before the op to reduce the rank
    # We insert post-reshape before output buffer / after the op to recover the shape
    # These ops_handle_6d_tensor override replace_6d_operation function to insert reshapes
    ops_handle_6d_tensor = {"Transpose"}
    # However, ReshapeOp does not support 6D inputs / outputs as well.
    # 6d inputs should be produced by the cases so that the inserted Reshape can be squashed
    # Case 1: ConstantOp
    # Case 2: Another ReshapeOp
    for input_ in node.input:
        input_tensor_type = model.get_tensor_type(input_)
        if input_tensor_type is None:
            logging.warning(f"Failed to get tensor type for input {input_}")
            return False
        # this optimization adds reshape, and reshape is not supported for dynamic shaped inputs
        if is_dynamic(input_tensor_type.shape):
            logging.warning(f"inputs of {node.name} are dynamic and not supported in 6D")
            return False
        producer = output_name_to_node[input_]
        if len(input_tensor_type.shape.dim) >= 6 and producer.op_type not in ("Reshape", "Constant", "ConstantOfShape"):
            logging.warning(f"inputs of {node.name} are not supported in 6D")
            return False

    # 6d outputs should be consumed by the cases so that the inserted Reshape can be squashed
    # Case 1: Another ReshapeOp
    # Case 2: ops_handle_6d_tensor (It will insert ReshapeOp)
    # 6D tensor is supported at in[1](indices) for ScatterNDOp
    output = node.output[0]
    output_tensor_type = model.get_tensor_type(output)
    if output_tensor_type is None:
        logging.warning(f"Failed to get tensor type for output {output}")
        return False
    for consumer in input_name_to_nodes[output]:
        if len(output_tensor_type.shape.dim) >= 6 and consumer.op_type not in (
            *ops_handle_6d_tensor,
            "Reshape",
            "Constant",
            "ConstantOfShape",
        ):
            logging.warning(f"outputs of {node.name} are not supported in 6D")
            return False

    return True


def post_reshape_insertion(
    model: onnx_model.ONNXModel,
    node: onnx.NodeProto,
    new_node: onnx.NodeProto,
    nodes_to_add: list[onnx.NodeProto],
    purpose="6d",
) -> bool:
    """
    Insert reshapes for each output of `node` into the output list of `new_node` to recover the original output shapes.

    Args:
        model: The model
        node: The original node with 6D in/out tensors.
        new_node: The substitute for `node`, with 6D tensor eliminated. Its output list will be filled out with the outputs
                  to the insert post-reshapes.
        nodes_to_add: The nodes to be added to the model graph. This will be filled out with the inserted post-reshapes.
        purpose: The purpose of the reshape insertion.
    """
    if any(output_name in model.graph().output for output_name in node.output):
        logging.warning(f"Currently we do not support post_reshape_insertion for {purpose} on nodes with output buffer")
        return False

    # We need to prepare post reshape for each output buffer
    value_infos_to_add = []
    initializers_to_add = []
    new_node_outputs = []
    reshape_nodes_to_add = []

    for output in node.output:
        output_tensor_type = model.get_tensor_type(output)
        if output_tensor_type is None:
            logging.warning(f"Failed to get tensor type for output {output}")
            return False

        post_reshape_input = onnx.helper.make_tensor_value_info(
            name=f"{node.name}_{output}_{purpose}_post_reshape_input",
            elem_type=output_tensor_type.elem_type,
            shape=None,
        )  # let ORT infer the shape of the post-reshape input at session-create time

        value_infos_to_add.append(post_reshape_input)
        new_node_outputs.append(post_reshape_input.name)

        post_reshape_output_shape = get_tensor_shape(model, output)
        if post_reshape_output_shape is None:
            logging.warning(f"Failed to get tensor shape for output {output}")
            return False

        post_reshape_shape = onnx.helper.make_tensor(
            name=f"{node.name}_{output}_{purpose}_post_reshape_shape",
            data_type=onnx.TensorProto.INT64,
            dims=[len(post_reshape_output_shape)],
            vals=post_reshape_output_shape,
        )

        initializers_to_add.append(post_reshape_shape)

        post_reshape_node = onnx.helper.make_node(
            "Reshape",
            name=f"{node.name}_{output}_{purpose}_post_reshape",
            inputs=[post_reshape_input.name, post_reshape_shape.name],
            outputs=[output],
        )
        reshape_nodes_to_add.append(post_reshape_node)

    model.graph().value_info.extend(value_infos_to_add)
    model.graph().initializer.extend(initializers_to_add)
    new_node.output.extend(new_node_outputs)
    nodes_to_add.extend(reshape_nodes_to_add)

    return True


# Although this surgery is not really a fusion, we can leverage the fusion class nonetheless.
class Tensor6DEliminationTranspose(fusions.Fusion):
    """6D tensor elimination for Transpose."""

    def __init__(self, model: onnx_model.ONNXModel):
        """Initialize.

        Args:
            model: An onnx_model.ONNXModel instance.
        """
        super().__init__(model, "Transpose", "Transpose")

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """Replace 6D Transpose by inserting Reshape around.

        In order to getting rid of >6D shapes, this optimization tries to reduce dimensions by
        merging those axes which are still consecutive after Transpose. Taking a Transpose with
        perm [0,1,3,4,5,2] for example, axes [0,1] and [3,4,5] are consucutive even after Transpose,
        and therefore they can be respectively merged into single dimension beforehand and recovered
        afterwards. Note that Transpose perm must be updated accordingly.

        Args:
            node: An onnx.NodeProto matching the specified search type (i.e., Reshape).
            input_name_to_nodes: A dict mapping tensor name to consumed nodes.
            output_name_to_node: A dict mapping tensor name to produced node.
        """

        input_shape = get_tensor_shape(self.model, node.input[0])
        if input_shape is None:
            return False

        transpose_perm: list[int] | None = self.get_node_attribute(node, "perm")
        if transpose_perm is None:
            return False

        input_tensor_type = self.model.get_tensor_type(node.input[0])
        if input_tensor_type is None:
            return False

        # Check rank.
        if len(input_shape) < 6:
            return False

        if not is_6d_supportable(node, self.model, input_name_to_nodes, output_name_to_node):
            return False

        # Calculate target shape by merging consecutive axes.
        target_shape, remaining_axes = [input_shape[0]], [0]
        for idx in range(1, len(input_shape)):
            if transpose_perm.index(idx - 1) + 1 == transpose_perm.index(idx):
                target_shape[-1] *= input_shape[idx]
            else:
                target_shape.append(input_shape[idx])
                remaining_axes.append(idx)

        # Current solution only supports shapes that could be merged into < 6D ones. For those
        # non-mergable cases should be handled by splitting and concating which is much more
        # complicated and therefore is left as future work.
        if len(target_shape) >= 6:
            return False

        # Calculate updated perm according to target shape and original perm.
        target_perm = list(range(len(target_shape)))
        target_perm.sort(key=lambda axis: transpose_perm.index(remaining_axes[axis]))

        value_infos_to_add = []
        initializers_to_add = []

        new_transpose_name = node.name

        pre_reshape_output = onnx.helper.make_tensor_value_info(
            name=f"{new_transpose_name}_6d_pre_reshape_output",
            elem_type=input_tensor_type.elem_type,
            shape=target_shape,
        )
        value_infos_to_add.append(pre_reshape_output)

        pre_reshape_shape = onnx.helper.make_tensor(
            name=f"{new_transpose_name}_6d_pre_reshape_shape",
            data_type=onnx.TensorProto.INT64,
            dims=[len(target_shape)],
            vals=target_shape,
        )

        initializers_to_add.append(pre_reshape_shape)

        pre_reshape_node = onnx.helper.make_node(
            "Reshape",
            name=f"{new_transpose_name}_6d_pre_reshape",
            inputs=[node.input[0], pre_reshape_shape.name],
            outputs=[pre_reshape_output.name],
        )

        new_transpose_node = onnx.helper.make_node(
            "Transpose",
            name=new_transpose_name,
            inputs=[pre_reshape_output.name],
            outputs=[],  # Will be filled out during post_reshape_insertion
            perm=target_perm,
        )

        success = post_reshape_insertion(self.model, node, new_transpose_node, self.nodes_to_add)

        # Only modify the graph if post reshape insertion succeeded
        if success:
            self.model.graph().value_info.extend(value_infos_to_add)
            self.model.graph().initializer.extend(initializers_to_add)
            self.nodes_to_remove.append(node)
            self.nodes_to_add.extend(
                [
                    pre_reshape_node,
                    new_transpose_node,
                ]
            )

        return success
