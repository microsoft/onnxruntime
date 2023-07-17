# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import dataclasses
import logging
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
import onnx
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
import torch.jit
import torch.onnx

# TODO(wschin,justinchuby): Since the internal APIs are not stable, please
# contact us if you hit errors.
import torch.onnx._internal
import torch.onnx._internal.diagnostics
import torch.onnx._internal.exporter
import torch.onnx._internal.fx.decomposition_table
import torch.onnx._internal.fx.passes
import torch.onnx._onnx_supported_ops
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree

import onnxruntime  # type: ignore
from onnxruntime.capi import _pybind_state as ORTC

# DEFAULT_ONNX_EXPORTER_OPTIONS contains shared information between exporter and DORT.
# For example, they should use the same decomposition table to maintain the same set
# operators when
#  1. capturing FX graph in torch.compile
#  2. call exporter's API to convert `torch.fx.GraphModule` to ONNX model.
DEFAULT_ONNX_EXPORTER_OPTIONS = torch.onnx._internal.exporter.ResolvedExportOptions(
    torch.onnx._internal.exporter.ExportOptions()
)

# TODO(wechi): This line must generate result identical to the call of
# _create_onnx_supports_op_overload_table(...) inside
# create_onnx_friendly_decomposition_table(...) in
# torch/onnx/_internal/fx/decomposition_table.py.
_SUPPORT_DICT = torch.onnx._internal.fx.decomposition_table._create_onnx_supports_op_overload_table(
    DEFAULT_ONNX_EXPORTER_OPTIONS.onnx_registry
)  # type: ignore

_EXTRA_SUPPORT_DICT: Dict[str, Any] = {
    "getattr": None,
    "_operator.getitem": None,
}

DORT_DECOMPOSITION_TABLE = DEFAULT_ONNX_EXPORTER_OPTIONS.decomposition_table

_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.longlong,
    torch.bool: np.bool_,
}


def _nvtx_range_push(name: str):
    """If PyTorch is installed with CUDA support, this starts NVTX range.

    Check torch.cuda.nvtx.range_push's document for more details.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_range_pop():
    """If PyTorch is installed with CUDA support, this terminates NVTX range.

    Check torch.cuda.nvtx.range_pop's document for more details.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _get_ort_device_type(device_type: str):
    if device_type == "cuda":
        return ORTC.OrtDevice.cuda()  # type: ignore
    if device_type == "cpu":
        return ORTC.OrtDevice.cpu()  # type: ignore
    # ort pytorch device is mapped to NPU OrtDevice type
    if device_type == "ort":
        return ORTC.OrtDevice.npu()  # type: ignore
    raise ValueError("Unsupported device type: " + device_type)


logger = logging.getLogger(__name__)
# Uncomment the following lines to print out development info.
# logging.basicConfig(level=logging.INFO)
# logger.setLevel(logging.INFO)


class OrtOperatorSupport(OperatorSupport):
    """
    Operator support for ONNXRuntime backend. It has two-level of support decision.
    One is via support_dict and the other one is via extra_support_dict. The logic
    of using support_dict is implemented in OrtOperatorSupport and extra_support_dict
    is used by OperatorSupport.is_node_supported.
    """

    def __init__(self):
        super().__init__(_EXTRA_SUPPORT_DICT)

    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        # OperatorSupport.is_node_supported returns True for non-callable nodes.
        # Since ORT can't execute them, we return False here to override the base
        # behavior.
        if node.op not in CALLABLE_NODE_OPS:
            return False
        # This is the and the only place to decide if aten op is supported.
        if node.op == "call_function" and node.target in _SUPPORT_DICT:
            logger.info("support_dict supports node.target: %s (type: %s)", node.target, type(node.target))
            return True
        logger.info("support_dict doesn't support node.target: %s (type: %s)", node.target, type(node.target))
        # If node.target is not in support_dict, we still want to check if torch.jit.script
        # can convert it to ONNX equivalence. Let's use base mechanism to do this.
        # See extra_support_dict  for supported ops.
        if super().is_node_supported(submodules, node):
            logger.info("extra_support_dict supports node.target: %s (type: %s)", node.target, type(node.target))
            return True
        logger.info("extra_support_dict doesn't supports node.target: %s (type: %s)", node.target, type(node.target))
        return False


def _move_placeholder_to_front(graph_module: torch.fx.GraphModule) -> None:
    """
    In torch.fx.Graph, placehoder is a special assignment node. If it's not
    executed in the beginning, it could overwrite values computed by upstream
    nodes.
    """

    graph = graph_module.graph
    placeholders = []
    first_not_placeholder = None
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
        if first_not_placeholder is None and node.op != "placeholder":
            first_not_placeholder = node
    if first_not_placeholder is None:
        return
    for placeholder in placeholders:
        first_not_placeholder.prepend(placeholder)


def _replace_to_copy_with_to(fx_module: torch.fx.GraphModule) -> None:
    # aten._to_copy doesn't have exporter so we replace it with aten.to.
    for node in fx_module.graph.nodes:
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.overloadpacket == torch.ops.aten._to_copy  # type: ignore
        ):
            is_default_layout = True
            is_on_same_device = True
            is_cast = True
            are_kwargs_supported = True
            if "layout" in node.kwargs and node.kwargs["layout"] != torch.strided:
                is_default_layout = False
            if "device" in node.kwargs and node.kwargs["device"] != node.args[0].meta["val"].device:
                is_on_same_device = False
            if "dtype" not in node.kwargs:
                is_cast = False
            for kwarg in node.kwargs:
                if kwarg not in ["layout", "device", "dtype"]:
                    are_kwargs_supported = False

            if len(node.args) == 1 and is_default_layout and is_on_same_device and is_cast and are_kwargs_supported:
                # This aten::_to_copy looks like ONNX Cast, so other kwargs are ignored.
                # This change could lead to invalid FX graph but it doesn't matter, as long as the downstream backend,
                # ONNXRuntime, can execute the exported ONNX graph.
                node.kwargs = {"dtype": node.kwargs["dtype"]}

                node.target = torch.ops.aten.to.dtype  # type: ignore
            else:
                raise RuntimeError(
                    f"aten._to_copy must be replaced with other ONNX-supported aten ops. \
                         args={[arg.meta for arg in node.args]}, kwargs={node.kwargs}"
                )
    fx_module.recompile()


def _create_onnx_model(onnx_proto):
    return onnx.ModelProto.FromString(onnx_proto)


def _create_onnx_session(onnx_proto, eps: Tuple[str, ...], session_options):
    # TODO(wechi): Add more EPs per PyTorch device types.
    # TODO(wechi): enable external allocators.
    return onnxruntime.InferenceSession(onnx_proto, providers=eps, sess_options=session_options)


def _infer_ep_from_device(*args) -> Tuple[str, ...]:
    """Return the first valid device (i.e., GPU or CPU) in argument list."""
    eps = []
    for arg in args:
        if hasattr(arg, "device"):
            device = arg.device
            if device.type == "cuda":
                eps.append("CUDAExecutionProvider")
            elif device.type == "cpu":
                eps.append("CPUExecutionProvider")
    return tuple(eps)


def _infer_ep_from_graph_module(graph_module: torch.fx.GraphModule) -> Tuple[str, ...]:
    """Return the first valid device (i.e., GPU or CPU) among outputs of this torch.fx.GraphModule."""
    for node in graph_module.graph.nodes:
        if node.op == "output":
            # Output node is unique. Let's retrieve output values from
            # this node's input list. And then just return.
            flattened_output_args, _ = _pytree.tree_flatten(node.args)
            output_args = []
            for output_arg in flattened_output_args:
                if hasattr(output_arg, "meta") and "val" in output_arg.meta:
                    # Select outputs with "val" information. Without "val",
                    # it's not possible access output_arg.meta["val"].device.
                    output_args.append(output_arg.meta["val"])
            return _infer_ep_from_device(*output_args)
    graph_module_str = graph_module.print_readable(print_output=False)
    raise ValueError(f"No output node is found in graph_module: {graph_module_str}")


def _sort_eps(eps: Tuple[str, ...]) -> Tuple[str, ...]:
    """Sort execution providers in eps based on pre-set priority."""

    def get_execution_provider_priority(ep: str) -> int:
        if ep == "CPUExecutionProvider":
            # Lowest priority.
            return 2
        if ep == "CUDAExecutionProvider":
            # Higher priority than CPU but lower than
            # other specialized EPs.
            return 1
        # Highest priority.
        return 0

    unique_eps = set(eps)
    return tuple(sorted(unique_eps, key=get_execution_provider_priority, reverse=True))


def _get_onnx_devices(values: Tuple[torch.Tensor, ...]) -> Tuple[ORTC.OrtDevice, ...]:  # type: ignore
    assert all(value.device == values[0].device for value in values), "All values must be on the same device."

    def _device_id_or_zero(device_id: int) -> int:
        return device_id or 0

    devices: Tuple[ORTC.OrtDevice, ...] = tuple(  # type: ignore
        ORTC.OrtDevice(  # type: ignore
            _get_ort_device_type(value.device.type),
            ORTC.OrtDevice.default_memory(),  # type: ignore
            _device_id_or_zero(value.device.index),
        )
        for value in values
    )
    return devices


def _get_ortvalues_from_torch_tensors(
    tensors: Tuple[torch.Tensor, ...], devices: Tuple[ORTC.OrtDevice, ...]
) -> Tuple[torch.Tensor, ...]:
    ortvalues = ORTC.OrtValueVector()  # type: ignore
    ortvalues.reserve(len(tensors))
    dtypes = []
    shapes = []
    data_ptrs = []

    for tensor in tensors:
        dtypes.append(_NP_DTYPE[tensor.dtype])
        shapes.append(tensor.size())
        data_ptrs.append(tensor.data_ptr())
    ortvalues.push_back_batch(tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues


def _to_real_tensor(tensor: FakeTensor) -> torch.Tensor:
    if tensor.is_sparse:
        raise ValueError("sparse tensor is not yet supported.")
    out = torch.empty(tensor.size(), dtype=tensor.dtype, device=tensor.device)
    return out


def _run_onnx_session_with_ortvaluevector(
    sess: onnxruntime.InferenceSession,
    input_names: Tuple[str, ...],
    inputs: Tuple[torch.Tensor, ...],
    input_devices: Tuple[ORTC.OrtDevice, ...],  # type: ignore
    output_names: Tuple[str, ...],
    outputs: Tuple[torch.Tensor, ...],
    output_devices: Tuple[ORTC.OrtDevice, ...],  # type: ignore
    preallocate_output: bool,
) -> Tuple[torch.Tensor, ...]:
    _nvtx_range_push("contiguous")
    inputs = tuple(a.contiguous() for a in inputs)
    _nvtx_range_pop()

    _nvtx_range_push("push_back_batch")

    ort_inputs = _get_ortvalues_from_torch_tensors(inputs, input_devices)

    # preallocate output pytorch Tensors and use the buffers affined to the torch device for the output ortvalue.
    # Because the output ortvalue is not allocated and owned by ort, it does not need to convert the output ortvalue
    # to torch Tensor transferring the ownership.
    if preallocate_output:
        pth_outputs = tuple(map(lambda t: _to_real_tensor(t) if isinstance(t, FakeTensor) else t, outputs))
        ort_outputs = _get_ortvalues_from_torch_tensors(pth_outputs, output_devices)
    else:
        ort_outputs = ORTC.OrtValueVector()  # type: ignore
    _nvtx_range_pop()

    _nvtx_range_push("run_with_ortvaluevector")
    run_options = onnxruntime.RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    sess.run_with_ortvaluevector(run_options, input_names, ort_inputs, output_names, ort_outputs, output_devices)
    _nvtx_range_pop()

    if preallocate_output:
        return pth_outputs
    else:
        _nvtx_range_push("after run_with_ortvaluevector")
        pth_outputs = onnxruntime.training.ortmodule._utils._ortvalues_to_torch_tensor(ort_outputs)  # type: ignore
        _nvtx_range_pop()
        return pth_outputs


def _assert_allclose_with_detailed_error_message(
    actual: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-03, atol: float = 1e-04
):
    diff = actual - expected
    real_atol = torch.max(torch.abs(diff))
    max_value = torch.max(torch.abs(actual), torch.abs(expected))
    max_value[max_value == 0.0] = 1.0
    real_rtol = torch.max(diff / max_value)
    allclose = bool(real_atol <= atol or real_rtol <= rtol)
    if not allclose:
        raise RuntimeError(
            "ONNX output doesn't match baseline output with "
            f"actual rtol={real_rtol} and actual atol={real_atol} "
            f"but expected rtol={rtol} and expected atol={atol}."
        )


@dataclasses.dataclass
class OrtExecutionInfo:
    """Information required to execute torch.fx.GraphModule using onnxruntime.InferenceSession"""

    def __init__(self):
        # session self.sessions[mod] is created for computing the graph in mod.
        self.sessions: Dict[torch.fx.GraphModule, onnxruntime.InferenceSession] = {}
        # self.input_names[mod] contains all input names in the ONNX model exported from mod.
        # self.input_names[mod][i] is the name of the i-th positional input of the graph in mod.
        self.input_names: Dict[torch.fx.GraphModule, Tuple[str, ...]] = {}
        # Similar to self.input_names, but for outputs of the graph.
        self.output_names: Dict[torch.fx.GraphModule, Tuple[str, ...]] = {}
        # self.input_devices[mod] contains devices of inputs fed to mod.forward (excluding self).
        # self.input_devices[mod][i] is the i-th positional input's device.
        self.input_devices: Dict[torch.fx.GraphModule, Tuple[ORTC.OrtDevice, ...]] = {}  # type: ignore
        # Similar to self.input_devices, but for outputs of the graph.
        self.output_devices: Dict[torch.fx.GraphModule, Tuple[ORTC.OrtDevice, ...]] = {}  # type: ignore
        # This is a debug flag. When True, this backend will compare its
        self.assert_allclose_to_baseline: bool = False
        # We need example outputs to determine output schema of ORT run.
        # self.example_outputs[mod] is the outputs of mod.forward(*self.example_inputs[mod]).
        self.example_outputs: Dict[torch.fx.GraphModule, Union[Tuple[torch.Tensor, ...], torch.Tensor]] = {}


class OrtBackend:
    """A backend compiles (sub-)graphs in torch.fx.GraphModule to onnxruntime.InferenceSession calls.

    The compiler entry point is OrtBackend.compile, which
        1. partitions the original graph into supported sub-graphs (type: torch.fx.GrpahModule) and unsupported
           sub-graphs.
        2. For each supported sub-graph, it replaces its _wrapped_call function with _ort_accelerated_call.
        3. Inside _ort_accelerated_call, it creates onnxruntime.InferenceSession and calls it to execute the sub-graph.
    """

    def __init__(self, ep: str = "CPUExecutionProvider", preallocate_output: bool = False, session_options=None):
        self._supported_ops = OrtOperatorSupport()
        # TODO: this is a naive implementation of cache without proper guard
        self._partitioner_cache: Dict[torch.fx.GraphModule, torch.fx.GraphModule] = {}
        # TODO: this is a naive implementation of cache without proper guard, this will only work for identical inputs
        self._ort_execution_info = OrtExecutionInfo()

        self.ep = ep
        self.session_options = session_options

        # preallocate_output allows for allocating output torch Tensor buffers and feeding them to InferenceSession
        # in order to avoid internal allocation of output buffers in InferenceSession.
        # If output ortvalue returned from InferenceSession is allocated internally,
        # it needs to be converted to torch Tensor for return, and the torch Tensor should hold the ownership.
        # When a custom torch device is used with a custom aten allocator, the conversion from ortvalue to torch Tensor
        # should be supported, which is currently done through dlpack. Note that dlpack might not support a custom torch device.
        # It can be avoided by allowing for preallocation for output buffers allocated by a custom aten allocator,
        # and use the preallocated output buffers for InferenceSession not holding any ownership for them.
        self.preallocate_output = preallocate_output

    def _ort_acclerated_call(self, graph_module: torch.fx.GraphModule, *args, **kwargs):
        if graph_module in self._ort_execution_info.sessions:
            # We have seen this graph before, so we can use cached objects including session.
            onnx_session = self._ort_execution_info.sessions[graph_module]
            input_names = self._ort_execution_info.input_names[graph_module]
            output_names = self._ort_execution_info.output_names[graph_module]
            input_devices = self._ort_execution_info.input_devices[graph_module]
            output_devices = self._ort_execution_info.output_devices[graph_module]
            prim_outputs = self._ort_execution_info.example_outputs[graph_module]
        else:
            # It's first time seeing such as graph. Let's make a new session
            # (type: onnxruntime.InferenceSession) for it.

            # TODO(wechi): this is a workaround for pytorch/pytorch#84311.
            _move_placeholder_to_front(graph_module)
            # Generate reference outputs. They are used to indicate output
            # tensors' types and devices when calling ORT.
            #
            # WARNING: The downstream code should not change prim_outputs and
            # this backend should always produces output with schema identical to prim_outputs'.
            try:
                prim_outputs = FakeTensorProp(graph_module).propagate(*args, **kwargs)
            except Exception:
                logger.info(f"FakeTensorProb failed for {graph_module}")
                # When FakeTensorProp fails, it is not possible to preallocate output buffers
                # because the output shapes are not inferred.
                self.preallocate_output = False

                # rethrow FakeTensorProb failure because it is not yet currently handled.
                raise
            self._ort_execution_info.example_outputs[graph_module] = prim_outputs

            from torch.onnx._internal.fx import fx_onnx_interpreter

            # Create the object to iterate through the nodes in graph one-by-one
            # and calls the corresponding ONNX exporter for each node.
            fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(
                diagnostic_context=DEFAULT_ONNX_EXPORTER_OPTIONS.diagnostic_context
            )
            # Start the per-node exporting process. It's conceptually a for loop
            # scanning through the nodes in the graph.
            exported = fx_interpreter.run(
                fx_graph_module=graph_module,
                onnxfunction_dispatcher=DEFAULT_ONNX_EXPORTER_OPTIONS.onnxfunction_dispatcher,
                op_level_debug=DEFAULT_ONNX_EXPORTER_OPTIONS.op_level_debug,
            )
            # Convert the exported result to ONNX ModelProto.
            onnx_proto = exported.to_model_proto(
                opset_version=DEFAULT_ONNX_EXPORTER_OPTIONS.opset_version
            ).SerializeToString()

            # Initialize a ORT session to execute this ONNX model.
            # Note that TorchDynamo assumes all inputs/outputs are on the
            # same device, but it's subject to change (very likely with
            # dynamic shape support), so we add execution providers
            # based on the all inputs/outputs plus a default OrtBackend.ep.
            eps_from_args = _infer_ep_from_device(args)
            eps_from_graph_module = _infer_ep_from_graph_module(graph_module)
            if eps_from_args:
                # If user feeds CUDA tensor as input argument,
                # we want to use CUDA EP.
                # Thus, `eps_from_args` (deduced from input arguments)
                # has highest priority.
                selected_eps = _sort_eps((*eps_from_args, self.ep))
            elif eps_from_graph_module:
                # If there is no EP in input arguments, we deduce EP from
                # graph_module's outputs. Those outputs may come from
                # FakeTensorProp or Dynamo's built-in symbolic shape inference.
                selected_eps = _sort_eps((*eps_from_graph_module, self.ep))
            else:
                # No EP found in inputs and outputs, let's use default.
                selected_eps = (self.ep,)

            onnx_session = _create_onnx_session(onnx_proto, selected_eps, self.session_options)
            # Cache ORT session. It's reused for the same "graph_module".
            self._ort_execution_info.sessions[graph_module] = onnx_session
            # Generate ONNX model and extract its input and output names.
            onnx_model = _create_onnx_model(onnx_proto)
            # TODO(wechi): ORT session should provide a API to extract
            # input and output names from the underlying model.
            input_names = tuple(input.name for input in onnx_model.graph.input)
            output_names = tuple(output.name for output in onnx_model.graph.output)
            input_devices = _get_onnx_devices(args)
            # Cache devices for inputs and outputs. They are used to invoke
            # ORT session. Output devices indicate where (e.g., GPU or CPU)
            # to store outputs
            if isinstance(prim_outputs, tuple):
                output_devices = _get_onnx_devices(prim_outputs)
            else:
                output_devices = _get_onnx_devices((prim_outputs,))
            self._ort_execution_info.input_names[graph_module] = input_names
            self._ort_execution_info.output_names[graph_module] = output_names
            self._ort_execution_info.input_devices[graph_module] = input_devices
            self._ort_execution_info.output_devices[graph_module] = output_devices

        if isinstance(prim_outputs, tuple):
            assert all(isinstance(elem, torch.Tensor) for elem in prim_outputs)
            # ORT always returns a tuple of outputs. If the original is a tuple, just returning
            # ORT output is ok.
            _nvtx_range_push("run_onnx_session_with_ortvaluevector")
            onnx_outputs = _run_onnx_session_with_ortvaluevector(
                onnx_session,
                input_names,
                args,
                input_devices,
                output_names,
                prim_outputs,
                output_devices,
                self.preallocate_output,
            )
            _nvtx_range_pop()
            if self._ort_execution_info.assert_allclose_to_baseline:
                # Compute baseline.
                baseline_outputs = torch._prims.executor.execute(graph_module, *args, executor="aten")
                # Ensure every output tensor is close to the corresponding baseline.
                for onnx_output, baseline_output in zip(onnx_outputs, baseline_outputs):
                    _assert_allclose_with_detailed_error_message(onnx_output, baseline_output)
            return onnx_outputs
        else:
            assert isinstance(prim_outputs, torch.Tensor)
            # ORT always returns a tuple of outputs. If the original output is a tensor,
            # ORT output's first element must be extracted and returned. Otherwise, type
            # mismatch may happen in downstream computation.
            onnx_outputs = _run_onnx_session_with_ortvaluevector(
                onnx_session,
                input_names,
                args,
                input_devices,
                output_names,
                (prim_outputs,),
                output_devices,
                self.preallocate_output,
            )
            assert len(onnx_outputs) == 1
            if self._ort_execution_info.assert_allclose_to_baseline:
                # Compute baseline.
                baseline_outputs = torch._prims.executor.execute(graph_module, *args, executor="aten")
                # Ensure output tensor is close to the corresponding baseline.
                _assert_allclose_with_detailed_error_message(onnx_outputs[0], baseline_outputs)
            return onnx_outputs[0]

    def compile(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        # FX graph based partitioning based on ONNX supported ops.
        if graph_module in self._partitioner_cache:
            partitioned_prim_graph_module = self._partitioner_cache[graph_module]
        else:
            prim_graph_module = graph_module
            # TODO(wechi): this is required for removing aten::_to_copy in _replace_to_copy_with_to.
            _replace_to_copy_with_to(prim_graph_module)
            partitioner = CapabilityBasedPartitioner(
                prim_graph_module, self._supported_ops, allows_single_node_partition=True
            )
            partitioned_prim_graph_module = partitioner.partition_and_fuse()
            self._partitioner_cache[graph_module] = partitioned_prim_graph_module

            # Overriding fused_module's __call__() function with ort_acclerated_call()
            # This loop goes through all graph partitions (each of them is an ONNX-representable graph)
            # and override their _wrappped_call function with _ort_accelerated_call.
            # Inside _ort_accelerated_call, the partition's graph is exported into ONNX and executed by ORT.
            for node in partitioned_prim_graph_module.graph.nodes:
                # TODO: use a better way to identify fused submodule
                if node.op == "call_module" and "fused_" in node.name:
                    fused_module = getattr(partitioned_prim_graph_module, node.name)
                    # self.ort_acclerated_call is responsible for exporting graph to ONNX,
                    # creating ORT session, and running ORT session.
                    fused_module._wrapped_call = self._ort_acclerated_call

        return partitioned_prim_graph_module

    def __call__(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        return self.compile(graph_module, args)
