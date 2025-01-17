import copy
import os

import torch
from numpy.testing import assert_allclose

from onnxruntime.training.ortmodule import ORTModule
from onnxruntime.training.ortmodule._graph_execution_manager_factory import GraphExecutionManagerFactory  # noqa: F401


def is_all_or_nothing_fallback_enabled(model, policy=None):
    from onnxruntime.training.ortmodule import ORTMODULE_FALLBACK_POLICY
    from onnxruntime.training.ortmodule._fallback import _FallbackPolicy

    if os.getenv("ORTMODULE_FALLBACK_POLICY") == _FallbackPolicy.FALLBACK_DISABLE.name:
        return False

    if not policy:
        policy = _FallbackPolicy.FALLBACK_DISABLE

    fallback_on_env = policy in ORTMODULE_FALLBACK_POLICY
    fallback_on_model = False
    if model:
        fallback_on_model = (
            policy in model._torch_module._execution_manager(is_training=True)._fallback_manager.policy
            or policy in model._torch_module._execution_manager(is_training=False)._fallback_manager.policy
        )
    return fallback_on_env or fallback_on_model


def assert_model_outputs(output_a, output_b, verbose=False, rtol=1e-7, atol=0):
    r"""Asserts whether output_a and output_b difference is within specified tolerance

    Args:
        output_a, output_b (list): Two list with of numeric values
        verbose (bool, default is False): if True, prints absolute difference for each weight
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """
    assert isinstance(output_a, list) and isinstance(output_b, list), "output_a and output_b must be list of numbers"
    if len(output_a) != len(output_b):
        raise AssertionError(
            f"output_a and output_b must have the same length ({len(output_a)!r} != {len(output_b)!r})."
        )

    # for idx in range(len(output_a)):
    assert_allclose(output_a, output_b, rtol=rtol, atol=atol, err_msg="Model output value mismatch")


def is_dynamic_axes(model):
    # Check inputs
    for inp in model._torch_module._execution_manager(model._is_training())._onnx_models.optimized_model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape:
            for dim in shape.dim:
                if dim.dim_param and not isinstance(dim.dim_param, str):
                    return False

    # Check outputs
    for out in model._torch_module._execution_manager(model._is_training())._onnx_models.optimized_model.graph.output:
        shape = out.type.tensor_type.shape
        if shape:
            for dim in shape.dim:
                if dim.dim_param and not isinstance(dim.dim_param, str):
                    return False
    return True


# TODO: thiagofc: Checkpoint related for redesign
def _get_name(name):
    if os.path.exists(name):
        return name
    rel = os.path.join("testdata", name)
    if os.path.exists(rel):
        return rel
    this = os.path.dirname(__file__)
    data = os.path.join(this, "..", "testdata")
    res = os.path.join(data, name)
    if os.path.exists(res):
        return res
    raise FileNotFoundError(f"Unable to find '{name}' or '{rel}' or '{res}'")


# Depending on calling backward() from which outputs, it's possible that grad of some weights are not calculated.
# none_pt_params is to tell what these weights are, so we will not compare the tensors.
def assert_gradients_match_and_reset_gradient(
    ort_model, pt_model, none_pt_params=[], reset_gradient=True, rtol=1e-04, atol=1e-05  # noqa: B006
):
    ort_named_params = list(ort_model.named_parameters())
    pt_named_params = list(pt_model.named_parameters())
    assert len(ort_named_params) == len(pt_named_params)

    for ort_named_param, pt_named_param in zip(ort_named_params, pt_named_params):
        ort_name, ort_param = ort_named_param
        pt_name, pt_param = pt_named_param

        assert pt_name in ort_name
        if pt_name in none_pt_params:
            assert pt_param.grad is None
            assert ort_param.grad is None or not torch.is_nonzero(torch.count_nonzero(ort_param.grad))
        else:
            assert_values_are_close(ort_param.grad, pt_param.grad, rtol=rtol, atol=atol)

        if reset_gradient:
            ort_param.grad = None
            pt_param.grad = None


def assert_values_are_close(input, other, rtol=1e-04, atol=1e-05):
    are_close = torch.allclose(input, other, rtol=rtol, atol=atol)
    if not are_close:
        abs_diff = torch.abs(input - other)
        abs_other = torch.abs(other)
        max_atol = torch.max(abs_diff - rtol * abs_other)
        max_rtol = torch.max((abs_diff - atol) / abs_other)
        err_msg = f"The maximum atol is {max_atol}, maximum rtol is {max_rtol}"
        raise AssertionError(err_msg)


def _run_model_on_device(device, model, input_list, label_input, is_eval_mode=False, run_forward_twice=False):
    if is_eval_mode:
        model.eval()
    else:
        model.train()

    def generate_inputs(input_list_, label_input_):
        with torch.no_grad():
            inputs_on_device = [input_.to(device) for input_ in input_list_]
            for i, val in enumerate(input_list_):
                if val.requires_grad:
                    inputs_on_device[i].requires_grad_()
            with torch.no_grad():
                target = label_input_.to(device)
        return inputs_on_device, target

    inputs_on_device1, target1 = generate_inputs(input_list, label_input)
    if run_forward_twice is True:
        inputs_on_device2, target2 = generate_inputs(input_list, label_input)

    output1 = model(*inputs_on_device1)
    if run_forward_twice is True:
        output2 = model(*inputs_on_device2)

    forward_outputs = [output1]
    grad_outputs = []

    if not is_eval_mode:
        criterion = torch.nn.MSELoss()
        loss = criterion(output1, target1)

        if run_forward_twice is True:
            loss += criterion(output2, target2)

        loss.backward()
        for _name, param in model.named_parameters():
            if param.requires_grad:
                grad_outputs.append(param.grad)
    return forward_outputs, grad_outputs


def run_with_pytorch_on_device(device, model, input_list, label_input, is_eval_mode=False, run_forward_twice=False):
    with torch.no_grad():
        model = copy.deepcopy(model).to(device)

    return _run_model_on_device(device, model, input_list, label_input, is_eval_mode, run_forward_twice)


def run_with_ort_on_device(device, model, input_list, label_input, is_eval_mode=False, run_forward_twice=False):
    with torch.no_grad():
        model = copy.deepcopy(model)
        model.to(device)
    model = ORTModule(model)

    return _run_model_on_device(device, model, input_list, label_input, is_eval_mode, run_forward_twice)


def compare_tensor_list(val_list_a, val_list_b):
    for val_a, val_b in zip(val_list_a, val_list_b):
        assert_values_are_close(val_a, val_b, atol=1e-7, rtol=1e-6)


def run_training_test_and_compare(
    pt_model_builder_func,
    pt_model_inputs_generator,
    pt_model_label_input,
    run_forward_twice=False,
    ignore_grad_compare=False,
    expected_outputs=[],  # noqa: B006
    expected_grads=[],  # noqa: B006
):
    cpu = torch.device("cpu")

    def cpu_barrier_func():
        pass

    run_training_test_on_device_and_compare(
        cpu,
        pt_model_builder_func,
        pt_model_inputs_generator,
        pt_model_label_input,
        cpu_barrier_func,
        run_forward_twice,
        ignore_grad_compare,
        expected_outputs,
        expected_grads,
    )

    def cuda_barrier_func():
        torch.cuda.synchronize()

    cuda = torch.device("cuda:0")
    run_training_test_on_device_and_compare(
        cuda,
        pt_model_builder_func,
        pt_model_inputs_generator,
        pt_model_label_input,
        cuda_barrier_func,
        run_forward_twice,
        ignore_grad_compare,
        expected_outputs,
        expected_grads,
    )


def run_training_test_on_device_and_compare(
    device,
    pt_model_builder_func,
    pt_model_inputs_generator,
    pt_model_label_input,
    barrier_func,
    run_forward_twice=False,
    ignore_grad_compare=False,
    expected_outputs=[],  # noqa: B006
    expected_grads=[],  # noqa: B006
):
    repeats = 8
    for _i in range(repeats):
        m = pt_model_builder_func()
        x = pt_model_inputs_generator()

        with torch.no_grad():
            m_ort = copy.deepcopy(m)
            x_ort = copy.deepcopy(x)

        outputs, grads = run_with_pytorch_on_device(
            device, m, [x], pt_model_label_input, run_forward_twice=run_forward_twice
        )
        barrier_func()

        outputs_ort, grads_ort = run_with_ort_on_device(
            device, m_ort, [x_ort], pt_model_label_input, run_forward_twice=run_forward_twice
        )
        barrier_func()

        val_list_a = [o.detach().cpu() for o in outputs if o is not None]
        val_list_b = [o.detach().cpu() for o in outputs_ort if o is not None]
        compare_tensor_list(val_list_a, val_list_b)

        if len(expected_outputs) > 0:
            compare_tensor_list(val_list_a, expected_outputs)

        # For some test, it is expected the diff might be big due to inconsistent computation orders.
        if ignore_grad_compare is False:
            val_list_a = [o.detach().cpu() for o in grads if o is not None]
            val_list_b = [o.detach().cpu() for o in grads_ort if o is not None]
            compare_tensor_list(val_list_a, val_list_b)

            if len(expected_grads) > 0:
                compare_tensor_list(val_list_a, expected_grads)


def run_evaluate_test_and_compare(
    pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input, run_forward_twice=False
):
    cpu = torch.device("cpu")

    def cpu_barrier_func():
        pass

    run_evaluate_test_on_device_and_compare(
        cpu,
        pt_model_builder_func,
        pt_model_inputs_generator,
        pt_model_label_input,
        cpu_barrier_func,
        run_forward_twice=run_forward_twice,
    )

    def cuda_barrier_func():
        torch.cuda.synchronize()

    cuda = torch.device("cuda:0")
    run_evaluate_test_on_device_and_compare(
        cuda,
        pt_model_builder_func,
        pt_model_inputs_generator,
        pt_model_label_input,
        cuda_barrier_func,
        run_forward_twice=run_forward_twice,
    )


def run_evaluate_test_on_device_and_compare(
    device,
    pt_model_builder_func,
    pt_model_inputs_generator,
    pt_model_label_input,
    barrier_func,
    run_forward_twice=False,
):
    repeats = 16
    for _i in range(repeats):
        m = pt_model_builder_func()
        x = pt_model_inputs_generator()

        m_ort = copy.deepcopy(m)
        x_ort = copy.deepcopy(x)

        outputs, grads = run_with_pytorch_on_device(
            device, m, [x], pt_model_label_input, is_eval_mode=True, run_forward_twice=run_forward_twice
        )
        barrier_func()

        outputs_ort, grads_ort = run_with_ort_on_device(
            device, m_ort, [x_ort], pt_model_label_input, is_eval_mode=True, run_forward_twice=run_forward_twice
        )
        barrier_func()

        val_list_a = [o.detach().cpu() for o in outputs if o is not None]
        val_list_b = [o.detach().cpu() for o in outputs_ort if o is not None]
        compare_tensor_list(val_list_a, val_list_b)
