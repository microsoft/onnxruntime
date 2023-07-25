# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Optional, Tuple

import onnx

import onnxruntime.training.onnxblock._graph_utils as _graph_utils
import onnxruntime.training.onnxblock.blocks as blocks
import onnxruntime.training.onnxblock.onnxblock as onnxblock_module


class AdamWOptimizer(blocks.Block):
    """Adds an AdamWOptimizer node to the onnx model."""

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

    def build(  # pylint: disable=too-many-arguments
        self,
        learning_rate_name: str,
        step_name: str,
        parameter_sequence_name: str,
        gradient_sequence_name: str,
        first_order_moment_sequence_name: str,
        second_order_moment_sequence_name: str,
    ):
        """Adds the AdamWOptimizer node to the model."""

        # get the model to manipulate
        onnx_model = self.base

        # define the node attributes
        node_attributes = {
            "alpha": self._betas[0],  # beta1
            "beta": self._betas[1],  # beta2
            "epsilon": self._eps,  # epsilon
            "weight_decay": self._weight_decay,  # weight decay
            "correct_bias": 1 if self._bias_correction else 0,  # bias_correction
            "adam_mode": 1,  # adam mode (1 for hf/transformers/AdamW)
        }

        # add the adamw node to the onnx model
        adamw_input_names = [
            learning_rate_name,  # learning rate
            step_name,  # training step
            parameter_sequence_name,  # param to be updated
            gradient_sequence_name,  # gradient of the param to be used for update
            first_order_moment_sequence_name,  # first order moment for this param
            second_order_moment_sequence_name,  # second order moment for this param
        ]
        adamw_output_name = _graph_utils.generate_graph_name("adamw.updated_flag")
        adamw_output_names = [adamw_output_name]
        adamw_node = onnx.helper.make_node(
            "AdamWOptimizer",
            adamw_input_names,
            adamw_output_names,
            name=_graph_utils.generate_graph_name("AdamWOptimizer"),
            domain="com.microsoft",
            **node_attributes,
        )
        onnx_model.graph.node.append(adamw_node)

        return adamw_output_name


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


class AdamW(onnxblock_module.ForwardBlock):
    """Builds AdamW optimizer onnxblock for the given training parameters.

    Creates a block that updates the model parameters based on the calculated
    gradient following the AdamW algorithm.

    Args:
        bias_correction: bool indicating whether to perform bias correction.
        betas: AdamW decay rate hyperparameters.
        eps: term added to the denominator for computing the moments.
        weight_decay: AdamW weight decay
        clip_grad (optional): an instance of the ClipGradNorm. If not provided,
                              gradient clipping will not be done.

    Returns:
        Returns a string of the output names from this optimizer node.
    """

    def __init__(
        self,
        bias_correction: Optional[bool] = True,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: Optional[float] = 1e-6,
        weight_decay: Optional[float] = 0.0,
        clip_grad=None,
    ):  # pylint: disable=too-many-arguments
        super().__init__()

        self._adamw = AdamWOptimizer(
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self._clip_grad = clip_grad

    def build(self, parameters):
        """Returns an AdamW optimizer model based on the input parameters."""

        # get the model to manipulate and update its namespace
        onnx_model = self.base

        # TODO: Avoid hard coded input/output strings
        learning_rate_name = "learning_rate"
        step_name = "step"
        params_name = "params"
        first_order_moments_name = "first_order_moments"
        second_order_moments_name = "second_order_moments"
        gradients_name = "gradients"

        trainable_parameters, _ = parameters

        # create the graph inputs for the lr, step, params, grads, moments
        onnx_model.graph.input.extend(
            [
                onnx.helper.make_tensor_value_info(learning_rate_name, onnx.TensorProto.FLOAT, [1]),
                onnx.helper.make_tensor_value_info(step_name, onnx.TensorProto.INT64, [1]),
            ]
        )

        # Prepare the tensor sequence inputs for params and moments
        for input_name in [params_name, gradients_name, first_order_moments_name, second_order_moments_name]:
            onnx_model.graph.input.append(  # noqa: PERF401
                onnx.helper.make_tensor_sequence_value_info(input_name, trainable_parameters[0].data_type, None)
            )

        # Clip the gradients if needed
        if self._clip_grad is not None:
            gradients_name = self._clip_grad(gradients_name)

        # Run multi tensor AdamWOptimizer
        updated_flag_name = self._adamw(
            learning_rate_name,
            step_name,
            params_name,
            gradients_name,
            first_order_moments_name,
            second_order_moments_name,
        )

        # Create the graph outputs
        onnx_model.graph.output.append(
            onnx.helper.make_tensor_value_info(updated_flag_name, onnx.TensorProto.INT64, [1])
        )

        return updated_flag_name
