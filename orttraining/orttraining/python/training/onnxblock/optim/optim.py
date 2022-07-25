# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# optim.py

import typing

import onnx

import onnxruntime.training.onnxblock._graph_utils as graph_utils
import onnxruntime.training.onnxblock.building_blocks as building_blocks
import onnxruntime.training.onnxblock.model as model
import onnxruntime.training.onnxblock.model_accessor as accessor

# TODO: Find a better place for these constants
_PRODUCER_NAME = "onnxblock offline tooling"
_OPSET_IMPORTS = (onnx.helper.make_opsetid("com.microsoft", 1), onnx.helper.make_opsetid("", 14))


class AdamWOptimizer(building_blocks.Block):
    """Adds an AdamWOptimizer node to the onnx model."""

    def __init__(
        self,
        bias_correction: typing.Optional[bool] = True,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: typing.Optional[float] = 1e-6,
        weight_decay: typing.Optional[float] = 0.0,
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
        onnx_model = accessor.global_accessor.model

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
        adamw_output_name = graph_utils.generate_random_graph_name("adamw.updated_flag")
        adamw_output_names = [adamw_output_name]
        adamw_node = onnx.helper.make_node(
            "AdamWOptimizer",
            adamw_input_names,
            adamw_output_names,
            name=graph_utils.generate_random_graph_name("AdamWOptimizer"),
            domain="com.microsoft",
            **node_attributes,
        )
        onnx_model.graph.node.append(adamw_node)

        return adamw_output_name


class ClipGradNorm(building_blocks.Block):
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

        self._reduce = building_blocks.ReduceAllL2()
        self._add = building_blocks.Add()
        self._div = building_blocks.Div()
        self._mul = building_blocks.Mul()
        self._clip = building_blocks.Clip(clip_max=1.0)

    def build(self, *gradient_names):
        """Adds a clip grad norm sub graph to the onnx model."""

        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # add the necessary graph initializers
        add_node_eps_name = graph_utils.generate_random_graph_name("add_eps")
        onnx_model.graph.initializer.append(
            onnx.helper.make_tensor(add_node_eps_name, onnx.TensorProto.FLOAT, [1], [1e-6])
        )
        max_norm_name = graph_utils.generate_random_graph_name("max_norm")
        onnx_model.graph.initializer.append(
            onnx.helper.make_tensor(max_norm_name, onnx.TensorProto.FLOAT, [1], [self._max_norm])
        )

        # perform gradient clipping
        total_norm_name = self._reduce(*gradient_names)
        adjusted_total_norm_name = self._add(total_norm_name, add_node_eps_name)
        clip_coef_name = self._clip(self._div(max_norm_name, adjusted_total_norm_name))
        return [self._mul(grad_name, clip_coef_name) for grad_name in gradient_names]


class AdamW(model.Model):
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
        bias_correction: typing.Optional[bool] = True,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: typing.Optional[float] = 1e-6,
        weight_decay: typing.Optional[float] = 0.0,
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
        self._sc = building_blocks.SequenceConstruct()

    def build(self, parameters):
        """Returns an AdamW optimizer model based on the input parameters."""

        # get the model to manipulate and update its namespace
        onnx_model = accessor.global_accessor.model
        onnx_model.graph.name = "AdamW Optimizer Model"
        onnx_model.producer_name = _PRODUCER_NAME
        onnx_model.opset_import.extend(_OPSET_IMPORTS)
        onnx_model.ir_version = onnx.IR_VERSION

        # TODO: Avoid hard coded input/output strings
        learning_rate_name = "learning_rate"
        step_name = "step"
        params_name = "params"
        first_order_moments_name = "first_order_moments"
        second_order_moments_name = "second_order_moments"
        gradient_suffix = "_grad"

        trainable_parameters, _ = parameters

        # create the graph inputs for the lr, step, params, grads, moments
        onnx_model.graph.input.extend(
            [
                onnx.helper.make_tensor_value_info(learning_rate_name, onnx.TensorProto.FLOAT, [1]),
                onnx.helper.make_tensor_value_info(step_name, onnx.TensorProto.INT64, [1]),
            ]
        )

        # Prepare the tensor sequence inputs for params and moments
        for input_name in [params_name, first_order_moments_name, second_order_moments_name]:
            onnx_model.graph.input.append(
                onnx.helper.make_tensor_sequence_value_info(input_name, trainable_parameters[0].data_type, None)
            )

        # TODO: Make the grads as a tensor sequence input after implementing clip grad
        # normalization implementation which takes in a tensor sequence.
        grad_names = []
        for param in trainable_parameters:
            grad_names.append(f"{param.name}{gradient_suffix}")
            onnx_model.graph.input.append(
                onnx.helper.make_tensor_value_info(grad_names[-1], param.data_type, param.dims)
            )

        # Clip the gradients if needed
        if self._clip_grad is not None:
            grad_names = self._clip_grad(*grad_names)

        # Run multi tensor AdamWOptimizer
        updated_flag_name = self._adamw(
            learning_rate_name,
            step_name,
            params_name,
            self._sc(*grad_names),
            first_order_moments_name,
            second_order_moments_name,
        )

        # Create the graph outputs
        onnx_model.graph.output.append(
            onnx.helper.make_tensor_value_info(updated_flag_name, onnx.TensorProto.INT64, [1])
        )

        return updated_flag_name
