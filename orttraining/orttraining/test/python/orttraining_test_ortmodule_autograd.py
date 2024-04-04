# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# pylint: disable=missing-docstring
# pylint: disable=C0103
# pylint: disable=W0212

import copy
import os
from typing import Tuple

import onnx
import pytest
import torch
from _test_helpers import (
    assert_gradients_match_and_reset_gradient,
    assert_values_are_close,
    compare_tensor_list,
    run_evaluate_test_and_compare,
    run_training_test_and_compare,
)
from packaging.version import Version
from torch.nn.parameter import Parameter

# Import external libraries.
import onnxruntime
from onnxruntime.training.ortmodule import ORTModule

torch.manual_seed(1)
onnxruntime.set_seed(1)


def torch_version_lower_than(v):
    return Version(torch.__version__) < Version(v)


@pytest.fixture(scope="session", autouse=True)
def run_before_test_session(request):
    def insert_disable_fallback_in_env():
        os.environ["ORTMODULE_FALLBACK_POLICY"] = "FALLBACK_DISABLE"

    def remove_disable_fallback_from_env():
        del os.environ["ORTMODULE_FALLBACK_POLICY"]

    insert_disable_fallback_in_env()
    request.addfinalizer(remove_disable_fallback_from_env)


def test_gelu():
    @torch.jit.script
    def bias_gelu(bias, y):
        x = bias + y
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    @torch.jit.script
    def bias_gelu_backward(g, bias, y):
        x = bias + y
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        return ff * g

    class GeLUFunction1(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, bias):
            ctx.save_for_backward(input, bias)
            return bias_gelu(bias, input)

        @staticmethod
        def backward(ctx, grad_output):
            input, bias = ctx.saved_tensors
            tmp = bias_gelu_backward(grad_output, bias, input)
            return tmp, tmp

    class GeLUModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = GeLUFunction1.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            out = self.relu(model_input, self.bias)
            return out

    output_size = 1024

    def model_builder():
        return GeLUModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)


def test_gelu_custom_func_rets_not_as_module_output():
    @torch.jit.script
    def bias_gelu(bias, y):
        x = bias + y
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    @torch.jit.script
    def bias_gelu_backward(g, bias, y):
        x = bias + y
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        return ff * g

    class GeLUFunction2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, bias):
            ctx.save_for_backward(input, bias)
            return bias_gelu(bias, input)

        @staticmethod
        def backward(ctx, grad_output):
            input, bias = ctx.saved_tensors
            tmp = bias_gelu_backward(grad_output, bias, input)
            return tmp, tmp

    class GeLUModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = GeLUFunction2.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            out = self.relu(model_input, self.bias)
            # add * 9 by intention to make custom function's output
            # NOT as module outputs (which are consumed by subsequent computations).
            # This aims to trigger a GC for "out", saying, out is released,
            # the underlying std::shared<PyNode> still have other references.
            # Otherwise, a segment fault will be triggered.
            out = out * 9
            return out

    output_size = 1024

    def model_builder():
        return GeLUModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)


def test_gelu_multiple_forward_runs():
    @torch.jit.script
    def bias_gelu(bias, y):
        x = bias + y
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    @torch.jit.script
    def bias_gelu_backward(g, bias, y):
        x = bias + y
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        return ff * g

    class GeLUFunction3(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, bias):
            ctx.save_for_backward(input, bias)
            return bias_gelu(bias, input)

        @staticmethod
        def backward(ctx, grad_output):
            input, bias = ctx.saved_tensors
            tmp = bias_gelu_backward(grad_output, bias, input)
            return tmp, tmp

    class GeLUModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = GeLUFunction3.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            out = self.relu(model_input, self.bias)
            return out

    output_size = 1024

    def model_builder():
        return GeLUModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input, run_forward_twice=True)


def test_megatronf():
    # MegatronGFunction is tested in distributed test files.
    class MegatronFFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_):
            return input_

        @staticmethod
        def backward(ctx, grad_output):
            # Bypass the reduce as if we are using only 1 GPU.
            return grad_output

    class MegatronFModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.copy_ = MegatronFFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            model_input = model_input + self.bias
            out = self.copy_(model_input)
            return out

    output_size = 1024

    def model_builder():
        return MegatronFModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)


def test_scalar_and_tuple():
    alpha_value = 5.0
    beta_value = (-1.0, 2.0)
    gamma_value = -1.0
    delta_value = True
    epsilon_value = (False, True)
    zeta_value = 1
    eta_value = (2, 3)
    theta_value = (3.0, 4.0)

    class ScalarAndTupleFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            input,
            alpha: float,
            beta: Tuple[float, float],
            gamma: float,
            delta: bool,
            epsilon: Tuple[bool, bool],
            zeta: int,
            eta: Tuple[int, int],
            theta: Tuple[float, float],
        ):
            ctx.save_for_backward(input)
            ctx.alpha = alpha
            ctx.beta = beta
            ctx.gamma = gamma
            ctx.delta = delta
            ctx.epsilon = epsilon
            ctx.zeta = zeta
            ctx.eta = eta
            ctx.theta = theta

            return alpha * beta[0] * beta[1] * gamma * input.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            alpha = ctx.alpha
            beta = ctx.beta
            gamma = ctx.gamma
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0

            assert alpha == alpha_value
            assert isinstance(alpha, float)

            assert all(a == b for a, b in zip(beta, beta_value))
            assert all(isinstance(x, float) for x in beta)

            assert gamma == gamma_value
            assert isinstance(gamma, float)

            assert ctx.delta == delta_value
            assert isinstance(ctx.delta, bool)

            assert all(a == b for a, b in zip(ctx.epsilon, epsilon_value))
            assert all(isinstance(x, bool) for x in ctx.epsilon)

            assert ctx.zeta == zeta_value
            assert isinstance(ctx.zeta, int)

            assert all(a == b for a, b in zip(ctx.eta, eta_value))
            assert all(isinstance(x, int) for x in ctx.eta)

            assert all(a == b for a, b in zip(ctx.theta, theta_value))
            assert all(isinstance(x, float) for x in ctx.theta)

            return alpha * beta[0] * beta[1] * gamma * grad_input, None, None, None, None, None, None, None, None

    class ScalarAndTupleModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.activation = ScalarAndTupleFunction.apply
            self.linear_a = torch.nn.Linear(output_size, output_size)
            self.linear_b = torch.nn.Linear(output_size, output_size)

        def forward(self, x):
            h = self.linear_a(x)
            h = self.activation(
                h, alpha_value, beta_value, gamma_value, delta_value, epsilon_value, zeta_value, eta_value, theta_value
            )
            h = self.linear_b(h)
            return h

    output_size = 2

    def model_builder():
        return ScalarAndTupleModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)


def test_scalar_and_tuple_reordered():
    class ScalarAndTupleReorderedFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, alpha, beta, input, gamma):
            ctx.save_for_backward(input)
            ctx.alpha = alpha
            ctx.beta = beta
            ctx.gamma = gamma
            return alpha * beta[0] * beta[1] * gamma * input.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            alpha = ctx.alpha
            beta = ctx.beta
            gamma = ctx.gamma
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return None, None, alpha * beta[0] * beta[1] * gamma * grad_input, None

    class ScalarAndTupleReorderedModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.activation = ScalarAndTupleReorderedFunction.apply
            self.linear_a = torch.nn.Linear(output_size, output_size)
            self.linear_b = torch.nn.Linear(output_size, output_size)

        def forward(self, x):
            h = self.linear_a(x)
            h = self.activation(5.0, (-1.0, 2.0), h, -1.0)
            h = self.linear_b(h)
            return h

    output_size = 2

    def model_builder():
        return ScalarAndTupleReorderedModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)


def test_pointer_type():
    class StringInputFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, name: str):
            ctx.save_for_backward(input)
            ctx.name = name
            return input.detach()

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    class StringInputFunctionTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.func = StringInputFunction.apply

        def forward(self, x):
            h = self.func(x, "temp_name")
            return h

    output_size = 2

    def model_builder():
        return StringInputFunctionTestModel()

    def input_generator():
        return torch.randn(output_size, dtype=torch.float).requires_grad_()

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)


@pytest.mark.skip(
    reason="This test is not correct. All tensors modified by in-place operattions should be mark_dirty(...)."
)
def test_InplaceUpdateInputAsOutputNotRequireGrad():
    class InplaceUpdateInputAsOutputNotRequireGradFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bias, inplace_update_input):
            # without mark_ditry, the inner computation graph is extracted into
            # another subgraph, which is a duplicated computation with the PythonOp.
            # so for the weights that are used twice BUT SHOULD only used once,
            # the gradients are almost 2x than PyTorch's grad, this is the reason we
            # ignore the gradient compare here.
            ctx.save_for_backward(inplace_update_input, bias)
            return inplace_update_input.add_(3 * bias)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    class InplaceUpdateInputAsOutputNotRequireGradModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.inplace_op = InplaceUpdateInputAsOutputNotRequireGradFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            x = model_input.mul(2)
            y1 = self.inplace_op(self.bias, x)  # x did not require grad
            y2 = x.add(self.bias)
            out = y1 + y2
            return out

    output_size = 1024

    def model_builder():
        return InplaceUpdateInputAsOutputNotRequireGradModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test when input is in-place updated, but does not require gradient.
    run_training_test_and_compare(model_builder, input_generator, label_input, ignore_grad_compare=True)


@pytest.mark.skip(
    reason="This test is not correct. All tensors modified by in-place operattions should be mark_dirty(...)."
)
def test_InplaceUpdateInputNotAsOutputNotRequireGrad():
    class InplaceUpdateInputNotAsOutputNotRequireGradFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bias, inplace_update_input):
            ctx.save_for_backward(inplace_update_input, bias)
            inplace_update_input.add_(3 * bias)
            return inplace_update_input * 5

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    class InplaceUpdateInputNotAsOutputNotRequireGradModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.inplace_op = InplaceUpdateInputNotAsOutputNotRequireGradFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            x = model_input.mul(2)
            y1 = self.inplace_op(self.bias, x)
            y2 = x.add(self.bias)
            out = y1 + y2
            return out

    output_size = 1024

    def model_builder():
        return InplaceUpdateInputNotAsOutputNotRequireGradModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Without mark_ditry, the inner computation graph is extracted into another subgraph,
    # which is a duplicated computation with the PythonOp.
    # So for the weights that are used twice BUT SHOULD only used once, the gradients are almost 2x than PyTorch's grad,
    # this is the reason we ignore the gradient compare here.
    run_training_test_and_compare(model_builder, input_generator, label_input, ignore_grad_compare=True)


@pytest.mark.skip(reason="disable due to exporter bug https://github.com/microsoft/onnx-converters-private/issues/37.")
def test_InplaceUpdateInputAsOutputNotRequireGradWithMarkDirty():
    class InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bias, inplace_update_input):
            ctx.save_for_backward(inplace_update_input, bias)
            ctx.mark_dirty(inplace_update_input)
            # Be noted: if we make the input dirty, we must also put the input in outputs, otherwise, we will get such an error:
            # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output.
            # The inputs that are modified inplace must all be outputs of the Function.""
            return inplace_update_input.add_(3 * bias)

        @staticmethod
        def backward(ctx, grad_output):
            # Bypass the reduce if we are using only 1 GPU.
            return grad_output, None

    class InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.inplace_op = InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            x = model_input.mul(2)
            y1 = self.inplace_op(self.bias, x)
            y2 = x.add(self.bias)
            out = y1 + y2
            return out

    output_size = 1024

    def model_builder():
        return InplaceUpdateInputAsOutputNotRequireGradWithMarkDirtyModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)


@pytest.mark.skip(
    reason="This test is not correct. All tensors modified by in-place operattions should be mark_dirty(...)."
)
def test_InplaceUpdateInputAsOutputRequireGrad():
    class InplaceUpdateInputAsOutputRequireGradFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bias, inplace_update_input):
            ctx.save_for_backward(inplace_update_input, bias)
            # Be noted: if we make the input dirty, we must also put the input in outputs, otherwise, we will get such an error:
            # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output. The inputs that are modified inplace must all be outputs of the Function.""
            return inplace_update_input.add_(3 * bias)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output

    class InplaceUpdateInputAsOutputRequireGradModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.inplace_op = InplaceUpdateInputAsOutputRequireGradFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            x = model_input + self.bias
            y1 = self.inplace_op(self.bias, x)
            y2 = x.add(self.bias)
            out = y1 + y2
            return out

    output_size = 1024

    def model_builder():
        return InplaceUpdateInputAsOutputRequireGradModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test when input is in-place updated, but does require gradient.
    #
    # without mark_ditry, the inner computation graph is extracted into another subgraph, which is a
    # duplicated computation with the PythonOp.  Thus, for the weights that are used twice BUT SHOULD
    # only used once, the gradients are almost 2x than PyTorch's grad, this is the reason we
    # ignore the gradient compare here.
    run_training_test_and_compare(model_builder, input_generator, label_input, ignore_grad_compare=True)


@pytest.mark.skip(
    reason="This test is not correct. All tensors modified by in-place operattions should be mark_dirty(...)."
)
def test_InplaceUpdateInputNotAsOutputRequireGrad():
    class InplaceUpdateInputNotAsOutputRequireGradFunction(torch.autograd.Function):
        # without mark_ditry, the inner computation graph is extracted into another subgraph, which is a duplicated computation with the PythonOp.
        # so for the weights that are used twice BUT SHOULD only used once, the gradients are almost 2x than PyTorch's grad, this is the reason we
        # ignore the gradient compare here.
        @staticmethod
        def forward(ctx, bias, inplace_update_input):
            ctx.save_for_backward(inplace_update_input, bias)
            inplace_update_input.add_(3 * bias)
            return inplace_update_input * 5

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output

    class InplaceUpdateInputNotAsOutputRequireGradModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.inplace_op = InplaceUpdateInputNotAsOutputRequireGradFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            x = model_input + self.bias
            y1 = self.inplace_op(self.bias, x)
            y2 = x.add(self.bias)
            out = y1 + y2
            return out

    output_size = 1024

    def model_builder():
        return InplaceUpdateInputNotAsOutputRequireGradModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # This case is known to have an warning message: "The output torch tensor @140214094625024, 140212816617984
    # should reuse the input torch tensor @140214095996104, 140212816617984 but actually not." It seems
    # if we don't have mark_dirty() in auto grad forward, the result is not using the input_,
    # (maybe a view of it, because data address is same)
    run_training_test_and_compare(model_builder, input_generator, label_input, ignore_grad_compare=True)


##########################################################################################


@pytest.mark.skip(reason="disable due to exporter bug https://github.com/microsoft/onnx-converters-private/issues/37.")
def test_InplaceUpdateInputAsOutputRequireGradWithMarkDirty():
    class InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bias, inplace_update_input):
            ctx.save_for_backward(inplace_update_input, bias)
            ctx.mark_dirty(inplace_update_input)
            # Be noted: if we make the input dirty, we must also put the input in outputs,
            # otherwise, we will get such an error:
            # "RuntimeError: Some elements marked as dirty during the forward method were not returned as output.
            # The inputs that are modified inplace must all be outputs of the Function.""
            return inplace_update_input.add_(3 * bias)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output

    class InplaceUpdateInputAsOutputRequireGradWithMarkDirtyModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.inplace_op = InplaceUpdateInputAsOutputRequireGradWithMarkDirtyFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            x = model_input + self.bias
            y1 = self.inplace_op(self.bias, x)
            y2 = x.add(self.bias)
            out = y1 + y2
            return out

    output_size = 1024

    def model_builder():
        return InplaceUpdateInputAsOutputRequireGradWithMarkDirtyModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)


def test_evaluation():
    class EvalTestFunction(torch.autograd.Function):
        @staticmethod
        # bias is an optional argument
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

        @staticmethod
        def backward(ctx, grad_output):
            return None

    class EvalTestModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.custom_fn = EvalTestFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            # model_input did not require_grad
            out = self.custom_fn(model_input)
            return out + self.bias

    output_size = 1024

    def model_builder():
        return EvalTestModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test pure inferencing scenarios, when inputs don't requires_grad.
    run_evaluate_test_and_compare(model_builder, input_generator, label_input)


@pytest.mark.skipif(
    torch_version_lower_than("1.10.0"),
    reason="PyTorch older than 1.10.0 has bugs for exporting multiple output custom function",
)
def test_two_outputs_function():
    class TwoOutputFunction1(torch.autograd.Function):
        @staticmethod
        # bias is an optional argument
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            w = x + y
            z = x * y
            return w, z

        @staticmethod
        def backward(ctx, dw, dz):
            x, y = ctx.saved_tensors
            # Based on chain rule, we can drive Jacobian
            # of this function.
            #   dL/dx = dL/dw * dw/dx + dL/dz * dz/dx
            # where
            #   dw/dx = 1
            #   dz/dx = y
            # Thus, dL/dx can be computed using the
            # following line. Note that dL is omitted
            # for convenience.
            dx = dw * 1.0 + dz * y
            # Similarly, we drive and then implement
            # the Jacobian for dy using chain rule
            #   dL/dw = dL/dw * dw/dy + dL/dz * dz/dy
            # where
            #   dw/dy = 1
            #   dz/dy = x
            dy = dw * 1.0 + dz * x
            return dx, dy

    class TwoOutputModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.fun = TwoOutputFunction1.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, x):
            a, b = self.fun(x, self.bias)
            return a + b

    output_size = 2

    def model_builder():
        return TwoOutputModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test multi-input and multi-output custom function.
    run_training_test_and_compare(model_builder, input_generator, label_input)


@pytest.mark.skipif(
    torch_version_lower_than("1.10.0"),
    reason="PyTorch older than 1.10.0 has bugs for exporting multiple output custom function",
)
def test_two_outputs_function_none_grad_fn():
    """This test is to verify the case that the first output tensor has grad_fn attr, but its grad_fn is None."""

    class TwoOutputFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, non_grad_fn_tensor, x, y):
            z = x * y
            # The first input is nn.Module input, who has grad_fn attr, but its grad_fn is None.
            return (
                non_grad_fn_tensor,
                z,
            )

        @staticmethod
        def backward(ctx, dc, dz):
            return dc, dz, dz

    class TwoOutputModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.fun = TwoOutputFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))
            self.int_type_tensor = torch.tensor([1, 4, 6, 7], device=torch.cuda.current_device(), dtype=torch.int64)

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, x):
            non_grad_fn_tensor, a = self.fun(self.int_type_tensor, x, self.bias)
            assert hasattr(non_grad_fn_tensor, "grad_fn")
            assert non_grad_fn_tensor.grad_fn is None
            return a

    output_size = 2

    def model_builder():
        return TwoOutputModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test multi-input and multi-output custom function.
    run_training_test_and_compare(model_builder, input_generator, label_input)


class MultipleOutputsFunctionWithNoGradOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, test_config_materialize_grad):
        ctx.save_for_backward(x, y)
        w = x + y
        z = x * y
        u = z.to(torch.float32).sum()

        ctx.w_shape = w.shape
        ctx.w_dtype = w.dtype
        ctx.w_device = w.device
        ctx.u_shape = u.shape
        ctx.u_dtype = u.dtype
        ctx.u_device = u.device
        ctx.z_shape = z.shape
        ctx.z_dtype = z.dtype
        ctx.z_device = z.device

        ctx.set_materialize_grads(test_config_materialize_grad)
        ctx.test_config_materialize_grad = test_config_materialize_grad

        # Note1:
        #   The requires_grad attribute values for w and z are both True, but in backward run, only z_grad will be needed.
        #   w_grad will be None/all zero.
        return w, u, z

    @staticmethod
    def backward(ctx, dw, du, dz):
        x, y = ctx.saved_tensors
        if ctx.test_config_materialize_grad:
            assert dw is not None
            assert du is not None
            assert dz is not None

            assert dw.shape == ctx.w_shape
            assert dw.dtype == ctx.w_dtype
            assert dw.device == ctx.w_device
            assert du.shape == ctx.u_shape
            assert du.dtype == ctx.u_dtype
            assert du.device == ctx.u_device
            assert dz.shape == ctx.z_shape
            assert dz.dtype == ctx.z_dtype
            assert dz.device == ctx.z_device
        else:
            assert dw is None
            assert du is not None
            assert dz is not None

            assert du.shape == ctx.u_shape
            assert du.dtype == ctx.u_dtype
            assert du.device == ctx.u_device
            assert dz.shape == ctx.z_shape
            assert dz.dtype == ctx.z_dtype
            assert dz.device == ctx.z_device

        dx = dz * y
        dy = dz * x
        return dx, dy, None


@pytest.mark.skipif(
    torch_version_lower_than("1.10.0"),
    reason="PyTorch older than 1.10.0 has bugs for exporting multiple output custom function",
)
@pytest.mark.parametrize("materialize_grad", [True, False])
def test_multiple_outputs_function_with_no_grad_output(materialize_grad: bool):
    class MultipleOutputsFunctionWithNoGradOutputModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.fun = MultipleOutputsFunctionWithNoGradOutput.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, x):
            # Be noted, the first output is not used in backward, so it should not require grad.
            _, u, b = self.fun(x, self.bias, materialize_grad)

            return b * u.to(b.dtype)

    output_size = 2

    def model_builder():
        return MultipleOutputsFunctionWithNoGradOutputModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test multi-input and multi-output custom function.
    run_training_test_and_compare(model_builder, input_generator, label_input)


def test_inner_module_call():
    class InnerModel(torch.nn.Module):
        def __init__(self, dim, device):
            super().__init__()
            self.bias = Parameter(torch.FloatTensor([1.0] * dim).to(device))

        def forward(self, x):
            z = 0.5 * x * x + self.bias
            return z

    class OuterFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dim, device, use_ort):
            ctx.save_for_backward(x)
            ctx.device = device
            ctx.inner = InnerModel(dim, device).to(device)
            if use_ort:
                ctx.inner = ORTModule(ctx.inner)
            z = ctx.inner(x)
            return z

        @staticmethod
        def backward(ctx, dv):
            (x,) = ctx.saved_tensors
            y = x.detach().to(ctx.device)
            y.requires_grad = True
            g = None
            with torch.enable_grad():
                z = ctx.inner(y)
            z.backward(dv)
            g = y.grad.detach()
            return g, None, None, None

    class OuterModel(torch.nn.Module):
        def __init__(self, dim, device, use_ort):
            super().__init__()
            self.fun = OuterFunction.apply
            self.dim = dim
            self.device = device
            self.use_ort = use_ort
            self.bias = Parameter(torch.FloatTensor([1.0] * dim).to(device))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, x):
            z = self.fun(x + self.bias, self.dim, self.device, self.use_ort)
            return z

    def get_inner_module_call_result(x, device, use_ort):
        torch.manual_seed(0)
        x = x.to(device)
        x.requires_grad = True
        model = OuterModel(2, device, use_ort)
        y = model(x).sum()
        y.backward()
        return y.detach(), x.grad.detach()

    x = torch.FloatTensor([1.0, -1.0])

    # Test indirect ORTModule call from custom function
    result_pth = get_inner_module_call_result(x.detach(), "cuda:0", False)
    result_ort = get_inner_module_call_result(x.detach(), "cuda:0", True)
    compare_tensor_list(result_ort, result_pth)

    # Test indirect ORTModule call from custom function
    result_ort = get_inner_module_call_result(x.detach(), "cpu", True)
    result_pth = get_inner_module_call_result(x.detach(), "cpu", False)
    compare_tensor_list(result_ort, result_pth)


@pytest.mark.skipif(
    torch_version_lower_than("1.10.0"),
    reason="PyTorch older than 1.10.0 has bugs for exporting multiple output custom function",
)
def test_share_input():
    class TwoOutputFunction2(torch.autograd.Function):
        @staticmethod
        # bias is an optional argument
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            w = x + y
            z = x * y
            return w, z

        @staticmethod
        def backward(ctx, dw, dz):
            x, y = ctx.saved_tensors
            dx = dw * 1.0 + dz * y
            dy = dw * 1.0 + dz * x
            return dx, dy

    class TwoOutputModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.fun = TwoOutputFunction2.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, x):
            a, b = self.fun(x, self.bias)
            c, d = self.fun(x, self.bias)
            return a + b + c + d

    output_size = 2

    def model_builder():
        return TwoOutputModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    def input_generator_with_requires_grad():
        return torch.randn(output_size, dtype=torch.float).requires_grad_()

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test multi-input and multi-output custom function.
    run_training_test_and_compare(model_builder, input_generator, label_input)

    run_training_test_and_compare(model_builder, input_generator_with_requires_grad, label_input)


def test_multiple_stream_in_forward_function():
    class MultipleStreamFunction1(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            default_stream = torch.cuda.current_stream()
            ctx.save_for_backward(input)
            stream = torch.cuda.Stream()
            torch.cuda._sleep(1000 * 1000)
            input = input * 0.2
            # on different stream
            with torch.cuda.stream(stream):
                stream.wait_stream(default_stream)
                input = input * 2
            default_stream.wait_stream(stream)
            return input

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output

    class MultipleStreamModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = MultipleStreamFunction1.apply

        def forward(self, model_input):
            b = model_input * 0.2
            out = self.relu(b)
            return out

    output_size = 2

    def model_builder():
        return MultipleStreamModel(output_size)

    def input_generator():
        return torch.tensor([2.8, 3.4], requires_grad=True)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test multi-input and multi-output custom function.
    run_training_test_and_compare(
        model_builder, input_generator, label_input, expected_outputs=[torch.tensor([0.224, 0.272])]
    )


def test_nondefault_stream_in_forward_function1():
    class MultipleStreamFunction2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            default_stream = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            # on different stream
            with torch.cuda.stream(stream):
                stream.wait_stream(default_stream)
                ctx.save_for_backward(input)
                input = input * 0.4

            default_stream.wait_stream(stream)
            return input

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output

    class MultipleStreamModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = MultipleStreamFunction2.apply

        def forward(self, model_input):
            model_input = model_input * 0.2
            torch.cuda._sleep(1000 * 1000)
            out = self.relu(model_input)
            return out

    output_size = 2

    def model_builder():
        return MultipleStreamModel(output_size)

    def input_generator():
        return torch.tensor([2.8, 3.4], requires_grad=True)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test multi-input and multi-output custom function.
    run_training_test_and_compare(
        model_builder, input_generator, label_input, expected_outputs=[torch.tensor([0.224, 0.272])]
    )


def test_nondefault_stream_in_forward_function2():
    class MultipleStreamFunction3(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            torch.cuda._sleep(1000 * 1000)
            input = input * 0.4
            return input

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output

    class MultipleStreamModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = MultipleStreamFunction3.apply

        def forward(self, model_input):
            model_input = model_input * 0.2
            stream = torch.cuda.Stream()
            default_stream = torch.cuda.current_stream()
            # on different stream
            with torch.cuda.stream(stream):
                stream.wait_stream(default_stream)
                out = self.relu(model_input)
            default_stream.wait_stream(stream)
            return out

    output_size = 2

    def model_builder():
        return MultipleStreamModel(output_size)

    def input_generator():
        return torch.tensor([2.8, 3.4], requires_grad=True)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test multi-input and multi-output custom function.
    run_training_test_and_compare(
        model_builder, input_generator, label_input, expected_outputs=[torch.tensor([0.224, 0.272])]
    )


def test_nondefault_stream_inplace_update_in_forward_function():
    class MultipleStreamFunction4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            default_stream = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            # on different stream
            with torch.cuda.stream(stream):
                stream.wait_stream(default_stream)
                ctx.save_for_backward(input)
                input.mul_(0.4)

            ctx.mark_dirty(input)
            default_stream.wait_stream(stream)
            return input

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output

    class MultipleStreamModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = MultipleStreamFunction4.apply

        def forward(self, model_input):
            model_input = model_input * 0.2
            torch.cuda._sleep(1000 * 1000)
            out = self.relu(model_input)
            return out

    output_size = 2

    def model_builder():
        return MultipleStreamModel(output_size)

    def input_generator():
        return torch.tensor([2.8, 3.4], requires_grad=True)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    # Test multi-input and multi-output custom function.
    run_training_test_and_compare(
        model_builder, input_generator, label_input, expected_outputs=[torch.tensor([0.224, 0.272])]
    )


def test_non_differentiable_autograd_function():
    class Bar(torch.autograd.Function):
        # A non-differentiable autograd Function whose forward output
        # doesn't have grad_fn attribute.
        @staticmethod
        def forward(ctx, x):
            y = torch.ones_like(x)
            return y

        @staticmethod
        def backward(ctx, dy):
            raise NotImplementedError()

    class Foo(torch.nn.Module):
        # Module calling non-differentiable function.
        def __init__(self):
            super().__init__()
            self._linear = torch.nn.Linear(2, 3)

        def forward(self, x):
            y = Bar.apply(x)
            z = self._linear(y)
            return z

    def run():
        m = Foo().to("cuda")
        x = torch.rand((2, 2), dtype=torch.float).to("cuda")

        # Baseline.
        y_ref = m(x)
        print("Ref:")
        print(y_ref)

        m = ORTModule(m)

        # Inferene mode.
        y_infer = m(x)
        print(y_infer)
        assert torch.allclose(y_ref, y_infer)

        # Training mode.
        m.train()
        y_train = m(x)
        print("Train:")
        assert torch.allclose(y_ref, y_train)

    run()


# There is bug in exporter side since 1.13 that will throw "RuntimeError: _Map_base::at" for this test.
@pytest.mark.skipif(Version(torch.__version__) >= Version("1.13.0"), reason="PyTorch 1.13+ incompatible")
def test_checkpoint_function():
    class A(torch.nn.Module):
        # A supported module.
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.l1(x)

    class B(torch.nn.Module):
        # This module is not exportable to ONNX because it
        # uses gradient-checkpointing. However, its two sub-module's
        # are exportable, so ORTModule should be used to compute them.
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(2, 2)
            self.a = A()

        def forward(self, x):
            def custom():
                def custom_forward(x_):
                    return self.a(x_)

                return custom_forward

            z = self.l1(torch.utils.checkpoint.checkpoint(custom(), x))
            return z

    def run():
        m = B().to("cuda")
        x = torch.rand((2, 2), dtype=torch.float).to("cuda")

        # Baseline.
        y_ref = m(x)
        print("Ref:")
        print(y_ref)

        os.environ["ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT"] = "1"

        m = ORTModule(m)

        # Inferene mode.
        y_infer = m(x)
        print(y_infer)
        assert torch.allclose(y_ref, y_infer)

        # Training mode.
        m.train()
        y_train = m(x)
        print("Train:")
        assert torch.allclose(y_ref, y_train)

        del os.environ["ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT"]

    run()


def test_pythonop_training_mode():
    def check_pythonop_training_mode(model, is_eval_mode):
        ## make sure the ort's PythonOp's training_mode is correct
        if is_eval_mode:
            onnx_nodes = (
                model._torch_module._execution_manager._inference_manager._onnx_models.exported_model.graph.node
            )
        else:
            onnx_nodes = model._torch_module._execution_manager._training_manager._onnx_models.exported_model.graph.node

        found_pythonop = False
        for node in onnx_nodes:
            if node.op_type == "PythonOp":
                found_pythonop = True
                for attr in node.attribute:
                    if attr.name == "training_mode":
                        if is_eval_mode:
                            assert attr.i == 0, f"in eval mode, it shoule be 0, while it is {attr.i} now"
                        else:
                            assert attr.i == 1, f"in training mode, it should be 1, while it is {attr.i} now"

        assert found_pythonop, "PythonOp should be found in the exported model"

    class TestFunction(torch.autograd.Function):
        @staticmethod
        # bias is an optional argument
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors
            tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
            ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
            return ff * grad_output

    class TestModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.custom_fn = TestFunction.apply
            self.bias = Parameter(torch.empty(output_size, dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            # model_input did not require_grad
            out = self.custom_fn(model_input)
            return out + self.bias

    output_size = 1024
    # 1 check traning mode
    ortmodule = ORTModule(TestModel(output_size)).train()
    _ = ortmodule(torch.randn(output_size, dtype=torch.float))
    check_pythonop_training_mode(ortmodule, is_eval_mode=False)
    # 2 check eval mode
    ortmodule = ORTModule(TestModel(output_size)).eval()
    _ = ortmodule(torch.randn(output_size, dtype=torch.float))
    check_pythonop_training_mode(ortmodule, is_eval_mode=True)


def test_python_op_save_input_for_backward():
    class GeLUFunctionTakeActivationInput(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
            ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
            g = ff * grad_output
            return g

    class TestLayer(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = GeLUFunctionTakeActivationInput.apply
            self._output_size = output_size
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))
            self.w = Parameter(
                torch.empty(output_size, output_size, device=torch.cuda.current_device(), dtype=torch.float)
            )
            with torch.no_grad():
                self.bias.uniform_()
                self.w.uniform_()

        def forward(self, model_input):
            activation0 = torch.add(model_input, 0.4)
            activation1 = activation0.view(self._output_size, -1)
            activation2 = torch.add(self.relu(activation1), self.bias)
            activation3 = torch.mul(activation2, 0.3)
            activation3 = torch.matmul(self.w, activation3)
            activation4 = torch.div(activation3, 1000)
            return activation4

    class TestModule(torch.nn.Module):
        def __init__(self, output_size) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TestLayer(output_size) for i in range(10)])

        def forward(self, x):
            # ModuleList can act as an iterable, or be indexed using ints
            for layer in self.layers:
                x = x.view(-1)
                x = torch.nn.functional.relu(layer(x))
            return x

    device = "cuda"
    output_size = 1024
    pt_model = TestModule(output_size).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def _run_step(model, input):
        loss = model(input).sum()
        loss.backward()
        return loss

    import warnings

    for _ in range(10):
        with warnings.catch_warnings(record=True):
            input = torch.randn(output_size, device=device, dtype=torch.float)
            pt_prediction = _run_step(pt_model, input)
            ort_prediction = _run_step(ort_model, input)

            assert_values_are_close(ort_prediction, pt_prediction, rtol=1e-04, atol=1.0)
            assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-5)


class DupNamedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        assert False  # should not be called # noqa: B011
        return bias + input

    @staticmethod
    def backward(ctx, grad_output):
        assert False  # should not be called # noqa: B011
        return grad_output, grad_output


def test_duplicate_named_functions():
    triggered = [False, False]

    class DupNamedFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, bias):
            ctx.save_for_backward(input, bias)
            triggered[0] = True
            return bias - input

        @staticmethod
        def backward(ctx, grad_output):
            triggered[1] = True
            return -grad_output, grad_output

    class DpNamedModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = DupNamedFunction.apply
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            out = self.relu(model_input, self.bias)
            return out

    output_size = 1024

    def model_builder():
        return DpNamedModel(output_size)

    def input_generator():
        return torch.randn(output_size, dtype=torch.float)

    # generate a label that have same shape as forward output.
    label_input = torch.ones([output_size])

    run_training_test_and_compare(model_builder, input_generator, label_input)

    assert triggered[0]
    assert triggered[1]


def test_customized_shape_inference():
    def _check_pythonop_shape(model):
        graph = model._torch_module._execution_manager._training_manager._onnx_models.optimized_model.graph
        found_pythonop = False
        python_op_input = []
        python_op_output = []
        for node in graph.node:
            if node.op_type == "PythonOp":
                found_pythonop = True
                python_op_input = node.input
                python_op_output = node.output
                break

        assert found_pythonop, "PythonOp should be found in the optimized_model"

        input_shapes = [None]
        input_dtypes = [None]

        output_shapes = [None, None]
        output_dtypes = [None, None]

        def _find_shape_and_dtype(value_infos):
            for value_info in value_infos:
                if value_info.name == python_op_input[0]:
                    input_shapes[0] = value_info.type.tensor_type.shape
                    input_dtypes[0] = value_info.type.tensor_type.elem_type

                if value_info.name == python_op_output[0]:
                    output_shapes[0] = value_info.type.tensor_type.shape
                    output_dtypes[0] = value_info.type.tensor_type.elem_type

                if value_info.name == python_op_output[1]:
                    output_shapes[1] = value_info.type.tensor_type.shape
                    output_dtypes[1] = value_info.type.tensor_type.elem_type

        _find_shape_and_dtype(graph.input)
        _find_shape_and_dtype(graph.value_info)

        assert all(s is not None for s in input_shapes), "PythonOp input shape should be found in the optimized_model"
        assert (
            all(d is not None for d in input_dtypes) is not None
        ), "PythonOp input dtype should be found in the optimized_model"

        assert all(s is not None for s in output_shapes), "PythonOp output shape should be found in the optimized_model"
        assert (
            all(d is not None for d in output_dtypes) is not None
        ), "PythonOp output dtype should be found in the optimized_model"

        def _compare_shape(shape1, shape2):
            if len(shape1.dim) != len(shape2.dim):
                return False

            for dim1, dim2 in zip(shape1.dim, shape2.dim):
                if dim1.HasField("dim_value") and dim1.HasField("dim_value") and dim1.dim_value == dim2.dim_value:
                    continue

                if dim1.HasField("dim_param") and dim1.HasField("dim_param") and dim1.dim_param == dim2.dim_param:
                    continue

                return False

            return True

        assert output_dtypes[0] == onnx.TensorProto.INT64
        assert len(output_shapes[0].dim) == 0
        assert _compare_shape(input_shapes[0], output_shapes[1])
        assert input_dtypes[0] == output_dtypes[1]

    class CustomShapeInferFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors
            tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
            ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
            return ff * grad_output

        @staticmethod
        def infer_shape(node: onnx.NodeProto, tensor_input_shapes, tensor_input_dtypes):
            return [tensor_input_shapes[0]], [tensor_input_dtypes[0]]

    class TestModel(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.custom_fn = CustomShapeInferFunction.apply
            self.bias = Parameter(torch.empty(output_size, dtype=torch.float))

            with torch.no_grad():
                self.bias.uniform_()

        def forward(self, model_input):
            # model_input did not require_grad
            out = self.custom_fn(model_input)
            return out + self.bias

    output_size = 1024
    ortmodule = ORTModule(
        TestModel(output_size),
    ).train()
    _ = ortmodule(torch.randn(output_size, dtype=torch.float))
    _check_pythonop_shape(ortmodule)


def test_python_op_return_persistent_param_as_value():
    """Some PythonOp return values that are still used by PyTorch computation. This test makes sure that ORTModule
    will not release/erase the storage of those return values during tear down OrtValue of the corresponding PythonOp
    return values.
    """

    class SimplePassThrough(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.detach()

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    class GeluWithExternalOutput(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, bias_param):
            ctx.save_for_backward(x)
            return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))), bias_param.detach()

        @staticmethod
        def backward(ctx, *grad_outputs):
            (x,) = ctx.saved_tensors
            tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
            ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
            g = ff * grad_outputs[0]
            return g, grad_outputs[1]

    class TestLayer(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.relu = GeluWithExternalOutput.apply
            self._output_size = output_size
            self.bias = Parameter(torch.empty(output_size, device=torch.cuda.current_device(), dtype=torch.float))
            self.w = Parameter(
                torch.empty(output_size, output_size, device=torch.cuda.current_device(), dtype=torch.float)
            )
            with torch.no_grad():
                self.bias.uniform_()
                self.w.uniform_()

        def forward(self, model_input):
            activation0 = torch.add(model_input, 0.4)
            activation1 = activation0.view(self._output_size, -1)

            # Returned detached_bias_param Tensor shares the same storage with self.bias
            # We are testing to make sure ORT will not erase the storage of self.bias during tear down OrtValue as
            # the returned value of the SimplePassThrough PythonOp.
            detached_bias_param = SimplePassThrough.apply(self.bias)
            relu_out, detached_bias_param = self.relu(activation1, detached_bias_param)
            activation2 = torch.add(relu_out, self.bias)
            activation3 = torch.add(activation2, detached_bias_param)
            activation3 = torch.matmul(self.w, activation3)
            activation4 = torch.div(activation3, 1000)
            return activation4

    class TestModule(torch.nn.Module):
        def __init__(self, output_size) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TestLayer(output_size) for i in range(6)])

        def forward(self, x):
            # ModuleList can act as an iterable, or be indexed using ints
            for layer in self.layers:
                x = x.view(-1)
                x = torch.nn.functional.relu(layer(x))
            return x

    device = "cuda"
    output_size = 1024
    pt_model = TestModule(output_size).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def _run_step(model, input):
        loss = model(input).sum()
        loss.backward()
        return loss

    for _ in range(5):
        input = torch.randn(output_size, device=device, dtype=torch.float)
        _run_step(pt_model, input)
        _run_step(ort_model, input)

        pt_params = {n: p for n, p in pt_model.named_parameters()}
        for name, param in ort_model.named_parameters():
            assert_values_are_close(param, pt_params[name], rtol=1e-04, atol=1e-3)
            if param.grad is not None:
                assert pt_params[name].grad is not None, f"pt param.grad is None for {name}"
                assert_values_are_close(param.grad, pt_params[name].grad, rtol=1e-04, atol=1e-3)
            else:
                assert pt_params[name].grad is None
