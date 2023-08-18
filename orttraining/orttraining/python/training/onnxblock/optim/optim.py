# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple

import onnx

import onnxruntime.training.onnxblock._graph_utils as _graph_utils
import onnxruntime.training.onnxblock.blocks as blocks
import onnxruntime.training.onnxblock.onnxblock as onnxblock_module


class ClipGradNorm(blocks.Block):
    """Builds a gradient clipping by norm sub graph for the onnx model.

    Creates a block that performs gradient clipping by l2 norm for the calculated
    gradient.

    Args:
        max_norm: float indicating the max norm of the gradients.

    Returns:
        Returns a string of the output names of the gradients after clipping.
    """

    def __init__(self, max_norm: float):
        super().__init__()

        self._max_norm = max_norm

    def build(self, gradients_name: str):
        """Adds a clip grad norm sub graph to the onnx model."""

        # get the model to manipulate
        onnx_model = self.base

        node_attributes = {
            "max_norm": self._max_norm,
        }

        # create the graph node for InplaceClipGradNorm
        cgn_node_input_names = [gradients_name]
        cgn_node_output_name = _graph_utils.generate_graph_name("clip_grad_norm_output")
        cgn_node_output_names = [cgn_node_output_name]
        cgn_node = onnx.helper.make_node(
            "InplaceClipGradNorm",
            cgn_node_input_names,
            cgn_node_output_names,
            name=_graph_utils.generate_graph_name("InplaceClipGradNorm"),
            domain="com.microsoft",
            **node_attributes,
        )
        onnx_model.graph.node.append(cgn_node)

        # Add the output to the value info of the model.
        onnx_model.graph.value_info.append(
            onnx.helper.make_tensor_sequence_value_info(cgn_node_output_name, onnx.TensorProto.FLOAT, None)
        )

        return cgn_node_output_name


class _OptimizerBase(blocks.Block):
    def __init__(self):
        super().__init__()

    def _build_optimizer_node(
        self,
        input_names: List[str],
        output_name: str,
        node_name: str,
        node_attributes: Dict,
    ) -> str:
        """
        Build and append an optimizer node to the ONNX graph.

        Args:
            input_names (list): List of input tensor names for the optimizer node.
            output_name (str): Output tensor name of the optimizer node.
            node_name (str): Name of the optimizer node.
            node_attributes (dict): Additional attributes for the optimizer node.

        Returns:
            str: The output tensor name of the optimizer node.
        """
        onnx_model = self.base

        # add the optimizer node to the onnx model
        optimizer_node = onnx.helper.make_node(
            node_name,
            input_names,
            [output_name],
            name=_graph_utils.generate_graph_name(node_name),
            domain="com.microsoft",
            **node_attributes,
        )

        onnx_model.graph.node.append(optimizer_node)

        return output_name


class SGDOptimizer(_OptimizerBase):
    def __init__(self):
        super().__init__()

    def build(
        self,
        learning_rate_name: str,
        gradients_name: str,
        params_name: str,
    ) -> str:
        """
        Build an SGD optimizer node.

        Args:
            learning_rate_name (str): Name of the learning rate input tensor.
            gradients_name (str): Name of the gradients input tensor.
            params_name (str): Name of the weights input tensor.

        Returns:
            str: The output tensor name of the SGD optimizer node.
        """

        input_names = [learning_rate_name, gradients_name, params_name]

        return self._build_optimizer_node(
            input_names,
            _graph_utils.generate_graph_name("update_completed"),
            "SGDOptimizerV2",
            {},
        )


class AdamWOptimizer(_OptimizerBase):
    def __init__(
        self,
        bias_correction: Optional[bool] = True,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: Optional[float] = 1e-6,
        weight_decay: Optional[float] = 0.0,
    ):
        super().__init__()

        self._bias_correction = bias_correction
        self._betas = betas
        self._eps = eps
        self._weight_decay = weight_decay

    def build(
        self,
        learning_rate_name: str,
        step_name: str,
        parameter_sequence_name: str,
        gradient_sequence_name: str,
        first_order_moment_sequence_name: str,
        second_order_moment_sequence_name: str,
    ) -> str:
        """
        Build an AdamW optimizer node.

        Args:
            learning_rate_name (str): Name of the learning rate input tensor.
            step_name (str): Name of the step input tensor.
            parameter_sequence_name (str): Name of the parameter sequence input tensor.
            gradient_sequence_name (str): Name of the gradient sequence input tensor.
            first_order_moment_sequence_name (str): Name of the first order moment sequence input tensor.
            second_order_moment_sequence_name (str): Name of the second order moment sequence input tensor.

        Returns:
            str: The output tensor name of the AdamW optimizer node.
        """

        input_names = [
            learning_rate_name,
            step_name,
            parameter_sequence_name,
            gradient_sequence_name,
            first_order_moment_sequence_name,
            second_order_moment_sequence_name,
        ]

        # define the node attributes
        node_attributes = {
            "alpha": self._betas[0],  # beta1
            "beta": self._betas[1],  # beta2
            "epsilon": self._eps,  # epsilon
            "weight_decay": self._weight_decay,  # weight decay
            "correct_bias": 1 if self._bias_correction else 0,  # bias_correction
            "adam_mode": 1,  # adam mode (1 for hf/transformers/AdamW)
        }

        return self._build_optimizer_node(
            input_names,
            _graph_utils.generate_graph_name("adamw.updated_flag"),
            "AdamWOptimizer",
            node_attributes,
        )


class _Optimizer(onnxblock_module.ForwardBlock):
    """Base class for building optimizer onnxblocks."""

    def __init__(self, clip_grad=None):
        super().__init__()
        self._clip_grad = clip_grad

    def build(self, parameters):
        onnx_model = self.base

        learning_rate_name = "learning_rate"
        params_name = "params"
        gradients_name = "gradients"
        step_name = "step"
        first_order_moments_name = "first_order_moments"

        trainable_parameters, _ = parameters

        onnx_model.graph.input.extend(
            [
                onnx.helper.make_tensor_value_info(learning_rate_name, onnx.TensorProto.FLOAT, [1]),
                onnx.helper.make_tensor_value_info(step_name, onnx.TensorProto.INT64, [1]),
            ]
        )

        for input_name in [params_name, gradients_name, first_order_moments_name]:
            onnx_model.graph.input.append(
                onnx.helper.make_tensor_sequence_value_info(input_name, trainable_parameters[0].data_type, None)
            )

        if self._clip_grad is not None:
            gradients_name = self._clip_grad(gradients_name)

        updated_flag_name = self._optimizer_specific_logic(
            learning_rate_name, params_name, gradients_name, trainable_parameters
        )

        return updated_flag_name

    def _optimizer_specific_logic(
        self,
        learning_rate_name: str,
        params_name: str,
        gradients_name: str,
        trainable_parameters: Tuple[List[onnx.TensorProto], List[onnx.TensorProto]],
    ) -> str:
        raise NotImplementedError("Subclasses must implement _optimizer_specific_logic method.")


class AdamW(_Optimizer):
    """Builds AdamW optimizer onnxblock for the given training parameters."""

    def __init__(self, bias_correction=True, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, clip_grad=None):
        super().__init__(clip_grad)
        self._adamw = AdamWOptimizer(
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def _optimizer_specific_logic(
        self,
        learning_rate_name: str,
        params_name: str,
        gradients_name: str,
        trainable_parameters: Tuple[List[onnx.TensorProto], List[onnx.TensorProto]],
    ) -> str:
        onnx_model = self.base
        step_name = "step"
        first_order_moments_name = "first_order_moments"
        second_order_moments_name = "second_order_moments"

        # Prepare the tensor sequence inputs for moments
        onnx_model.graph.input.append(
            onnx.helper.make_tensor_sequence_value_info(
                second_order_moments_name, trainable_parameters[0].data_type, None
            )
        )

        updated_flag_name = self._adamw(
            learning_rate_name,
            step_name,
            params_name,
            gradients_name,
            first_order_moments_name,
            second_order_moments_name,
        )

        # Create graph outputs for AdamW
        onnx_model.graph.output.append(
            onnx.helper.make_tensor_value_info(updated_flag_name, onnx.TensorProto.BOOL, [1])
        )

        return updated_flag_name


class SGD(_Optimizer):
    """Builds SGD optimizer onnxblock for the given training parameters."""

    def __init__(self, clip_grad=None):
        super().__init__(clip_grad)
        self._sgd = SGDOptimizer()

    def _optimizer_specific_logic(
        self,
        learning_rate_name: str,
        params_name: str,
        gradients_name: str,
        trainable_parameters: Tuple[List[onnx.TensorProto], List[onnx.TensorProto]],
    ) -> str:
        onnx_model = self.base
        updated_flag_name = self._sgd(learning_rate_name, params_name, gradients_name)

        # Create graph outputs for SGD
        onnx_model.graph.output.append(
            onnx.helper.make_tensor_value_info(updated_flag_name, onnx.TensorProto.BOOL, [1])
        )

        return updated_flag_name
