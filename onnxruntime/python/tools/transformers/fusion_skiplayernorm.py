# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSkipLayerNormalization(Fusion):
    """
    Fuse Add + LayerNormalization into one node: SkipLayerNormalization
    Note: This fusion does not check the input shape of Add and LayerNormalization.
    """

    def __init__(
        self,
        model: OnnxModel,
        fused_op_type: str = "SkipLayerNormalization",
        search_op_types: str = "LayerNormalization",
    ):
        super().__init__(model, fused_op_type, search_op_types)
        # Update shape inference is needed since other fusions might add new edge which does not have shape info yet.
        self.shape_infer_helper = self.model.infer_runtime_shape({"batch_size": 4, "seq_len": 7}, update=True)

        if self.shape_infer_helper is None:
            # TODO(tianleiwu): support subgraph in shape inference or add broadcasting in SkipLayerNormalization op.
            logger.warning("symbolic shape inference disabled or failed.")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        add = self.model.get_parent(node, 0, output_name_to_node)

        # In some models there is input_ids->gather->add->LayerNorm and one of input of the
        # add node is initializer with fixed shape which should not be fused into SkipLayerNorm
        if add is None:
            return

        for add_input in add.input:
            if self.model.get_initializer(add_input) is not None:
                return

        # The number of input node of add should be 2
        if len(self.model.get_parents(add)) != 2:
            return

        # To avoid an Add node have two children of LayerNormalization, we shall only fuse one SkipLayerNormalization
        if add in self.nodes_to_remove:
            return

        # Root Mean Square Layer Normalization
        simplified = node.op_type == "SimplifiedLayerNormalization"

        if self.shape_infer_helper is not None:
            if not self.shape_infer_helper.compare_shape(add.input[0], add.input[1]):
                logger.debug(
                    "skip SkipLayerNormalization fusion since shape of inputs (%s, %s) are not same",
                    add.input[0],
                    add.input[1],
                )
                return
        else:
            logger.debug("skip SkipLayerNormalization fusion since symbolic shape inference failed")
            return

        gather_path = self.model.match_parent_path(add, ["Gather"], [None])
        if gather_path is not None and self.model.find_graph_input(gather_path[0].input[1]) is None:
            if self.model.match_parent_path(gather_path[0], ["ConstantOfShape"], [1]) is None:
                return

        residual_add_has_multiple_consumers = False
        add_children = self.model.get_children(add, input_name_to_nodes)

        # This means that the residual Add before the LayerNormalization produces an output
        # that is consumed by some other nodes other than the LayerNormalization itself
        # We can still go ahead with the SkipLayerNormalization fusion but we need to
        # preserve the output of Add and that needs to be produced by SkipLayerNormalization.
        if len(add_children) != 1:
            residual_add_has_multiple_consumers = True

        outputs_to_keep = node.output

        if residual_add_has_multiple_consumers:
            outputs_to_keep.extend([add.output[0]])

        outputs = [node.output[0]]

        # Skip the other optional outputs of SkipLayerNormalization before adding the Add's output
        if residual_add_has_multiple_consumers:
            outputs.extend(["", "", add.output[0]])

        if (
            add is not None
            and add.op_type == "Add"
            and self.model.is_safe_to_fuse_nodes([add, node], outputs_to_keep, input_name_to_nodes, output_name_to_node)
        ):
            self.nodes_to_remove.extend([add, node])

            inputs = (
                [add.input[0], add.input[1], node.input[1], node.input[2]]
                if not simplified
                else [add.input[0], add.input[1], node.input[1]]
            )
            normalize_node = helper.make_node(
                self.fused_op_type,
                inputs=inputs,
                outputs=outputs,
                name=self.model.create_node_name(self.fused_op_type, name_prefix="SkipLayerNorm"),
            )
            normalize_node.domain = "com.microsoft"

            # Pass attribute "epsilon" from layernorm node to SkipLayerNormalization
            for att in node.attribute:
                if att.name == "epsilon":
                    normalize_node.attribute.extend([att])

            # Set default epsilon if no epsilon exists from layernorm
            if len(normalize_node.attribute) == 0:
                normalize_node.attribute.extend([helper.make_attribute("epsilon", 1.0e-12)])

            self.nodes_to_add.append(normalize_node)
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name


class FusionBiasSkipLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipLayerNormalization", "SkipLayerNormalization", "add bias")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if len(node.input) != 4:
            return

        return_indice = []
        nodes = self.model.match_parent_path(node, ["Add", "MatMul"], [None, None], None, return_indice)
        if nodes is None:
            # In case of fp16, we could have a Cast between the MatMul and the bias Add
            nodes = self.model.match_parent_path(
                node, ["Add", "Cast", "MatMul"], [None, None, None], None, return_indice
            )
            if nodes is None:
                return

        assert len(return_indice) == 2 or len(return_indice) == 3
        add_input_index = return_indice[0]
        if add_input_index >= 2:
            return

        (add, matmul) = nodes

        # bias should be one dimension
        bias_index = -1
        bias_weight = None
        for i, input in enumerate(add.input):
            initializer = self.model.get_initializer(input)
            if initializer is None:
                continue
            bias_index = i
            bias_weight = NumpyHelper.to_array(initializer)
            break
        if bias_weight is None:
            logger.debug("Bias weight not found")
            return
        if len(bias_weight.shape) != 1:
            logger.debug("Bias weight is not 1D")
            return

        subgraph_nodes = [node, add]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, node.output, input_name_to_nodes, output_name_to_node):
            logger.debug("Skip fusing SkipLayerNormalization with Bias since it is not safe")
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        inputs = [
            node.input[1 - add_input_index],
            matmul.output[0],
            node.input[2],
            node.input[3],
            add.input[bias_index],
        ]
        new_node = helper.make_node(
            "SkipLayerNormalization",
            inputs=inputs,
            outputs=node.output,
            name=self.model.create_node_name("SkipLayerNormalization", "SkipLayerNorm_AddBias_"),
        )
        new_node.domain = "com.microsoft"

        # Pass attribute "epsilon" from skiplayernorm node to skiplayernorm(add bias)
        for att in node.attribute:
            if att.name == "epsilon":
                new_node.attribute.extend([att])

        # Set default epsilon if no epsilon exists from skiplayernorm
        if len(new_node.attribute) == 0:
            new_node.attribute.extend([helper.make_attribute("epsilon", 1.0e-12)])

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
