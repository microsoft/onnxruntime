# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import dataclasses
import logging
from typing import Any, Callable, Dict, Mapping, Set, Tuple, Union

import numpy as np
import onnx
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
import torch.jit
import torch.onnx
import torch.onnx._onnx_supported_ops
from torch._decomp import decomposition_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.onnx._globals import GLOBALS as ONNX_GLOBALS

import onnxruntime  # type: ignore
from onnxruntime.capi import _pybind_state as ORTC

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


def _get_ort_device_type(device_type: str, device_index: int):
    if device_type == "cuda":
        return ORTC.OrtDevice.cuda()  # type: ignore
    if device_type == "cpu":
        return ORTC.OrtDevice.cpu()  # type: ignore
    if device_type == "ort":
        return ORTC.get_ort_device(device_index).device_type()  # type: ignore
    raise ValueError("Unsupported device type: " + device_type)


logger = logging.getLogger(__name__)
# Uncomment the following lines to print out development info.
# logging.basicConfig(level=logging.INFO)
# logger.setLevel(logging.INFO)


def _get_onnx_supported_table() -> Set[str]:
    # TODO(wechi): this entire function should be replaced by a formal a exporter API.

    onnx_supported_ops: Set[str] = set()
    for aten_op_name, schema in torch.onnx._onnx_supported_ops.all_symbolics_schemas().items():
        # TODO(wechi): aten_op_name could be prim::add in addition to aten::add.
        # We should build another dictionary for storing support table for prim ops.
        # Currently, we only consider aten ops as before.
        if not aten_op_name.startswith("aten::"):
            logger.info("Skip %s in support table because it's not in aten domain.", aten_op_name)
            continue
        short_op_name = aten_op_name.split("aten::")[1]
        if not hasattr(torch.ops.aten, short_op_name):  # type: ignore
            # Some aten ops are not in torch.ops.aten. Those are excluded until we
            # figure out why.
            logger.info("Skip %s in support table because it's not found in torch.ops.aten.", aten_op_name)
            continue
        # aten_op_name is aten symbol's name; e.g., "sum" for aten::sum.
        # opsets_string is the ONNX opsets that can express info[0]; e.g., "15 16 17"
        # indicates that opset 15, opset 16, and opset 17 can all express aten_op_name.
        if ONNX_GLOBALS.export_onnx_opset_version in schema.opsets:
            logger.info("Add %s to support table.", aten_op_name)
            onnx_supported_ops.add(aten_op_name)
    return onnx_supported_ops


def _get_support_dictionaries_and_decomposition_tables() -> Tuple[
    Dict[torch._ops.OpOverload, Any],
    Dict[str, Any],
    Dict[torch._ops.OpOverload, Callable],
    Dict[torch._ops.OpOverload, Callable],
]:
    # The keys of this dictionary are OpOverload's which can be
    # exported by ONNX exporter. Type of key is torch._ops.OpOverload.
    # For example, if torch.ops.aten.add.default is a key in support_dict,
    # all torch.fx.Node's with torch.ops.aten.add.default as target will
    # be selected by CapabilityBasedPartitioner and sent to ORT for
    # computation.
    # We choose torch._ops.OpOverload as the key because
    #  1. torch._ops.OpOverload uniquely identifies an op. We don't want
    #     to use OpOverloadPacket because it contains overloads of the same op.
    #     This allows us to select supported ops at the finest grain.
    #  2. torch._ops.OpOverload is what we get from torch.fx.Node.target. Getting
    #     qualified name using _get_qualified_name is not needed.
    support_dictionary: Dict[torch._ops.OpOverload, Any] = {}
    for aten_op_name in _get_onnx_supported_table():
        short_op_name = aten_op_name.split("aten::")[1]
        op_overload_packet = getattr(torch.ops.aten, short_op_name)  # type: ignore
        # Due to the lack of overload name in exporting function's name, assume
        # each exporting function (e.g., torch.onnx.symbolic_opset9.add) support
        # all overloads (e.g., in torch.ops.aten.add).
        # Thus, we register all torch._ops.OpOverload's for the same exporting function.
        # Please manually exclude torch._ops.OpOverload if exporter fails.
        for overload in op_overload_packet.overloads():
            op_overload = getattr(op_overload_packet, overload)
            support_dictionary[op_overload] = None

    # No decomposition table. OpOverload in this table shouldn't be found
    # in aten2aten_decomposition_table.
    # The symbols in this set will be replaced by torch.ops.aten.to.dtype in replace_to_copy_with_to because
    # only aten.to has ONNX exporter.
    # If the replacement fails, ONNX exporter will fail because only aten.to has ONNX exporter.
    # TODO(wechi): For a long-term solution, we need to ensure every op used in op decomposision has
    # an exporter.
    no_decomposition_table: Set[torch._ops.OpOverload] = {
        torch.ops.aten._to_copy.default,  # type: ignore
        torch.ops.aten._to_copy.out,  # type: ignore
    }

    # decomposition_table currently contains both aten2aten and aten2prim decompositions
    # This is a hack to separate them, as ONNX only recognizes aten symbols.
    aten2aten_decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}
    aten2prim_decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}

    for op_overload, decomp_fn in decomposition_table.items():
        if op_overload in support_dictionary:
            # ONNX can express this op, no need to decompose.
            continue
        if "torch._refs" in decomp_fn.__module__:
            aten2prim_decomposition_table[op_overload] = decomp_fn
        else:
            if op_overload in no_decomposition_table:
                continue
            # Assume ONNX can express ops after decomposition.
            # If no, exporter will fail and the user need to
            # remove this decomposition rule.
            aten2aten_decomposition_table[op_overload] = decomp_fn

    # Some torch.fx.Node's are converted to ONNX-compatible ops
    # by torch.jit.script. They don't have direct ONNX exporting
    # functions but still runnable in ORT.
    extra_support_dictionary: Dict[str, Any] = {
        "getattr": None,
        "_operator.getitem": None,
    }

    return support_dictionary, extra_support_dictionary, aten2aten_decomposition_table, aten2prim_decomposition_table


(
    _SUPPORT_DICT,
    _EXTRA_SUPPORT_DICT,
    _ATEN2ATEN_DECOMP,
    _ATEN2PRIM_DECOMP,
) = _get_support_dictionaries_and_decomposition_tables()


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


def _jit_graph_to_onnx_model(graph, operator_export_type):
    r"""
    This function exports torch::jit::Graph object
    to serialized ONNX ModelProto.
    It only keeps the essential parts for IR graph conversions.
    It also does not interact with actual PyTorch modules nor
    PyTorch tensor inputs.
    """
    graph = torch.onnx.utils._optimize_graph(graph, operator_export_type, params_dict={})
    proto, _, _, _ = graph._export_onnx(  # type: ignore
        {},
        ONNX_GLOBALS.export_onnx_opset_version,
        {},
        False,
        operator_export_type,
        False,
        False,
        {},
        True,
        "",
        {},
    )
    return proto


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


def _fx_to_torchscript(
    fx_module: torch.fx.GraphModule,
) -> torch.jit.ScriptModule:
    """Convert torch.fx.Graph to torch.jit.ScriptModule."""

    for node in fx_module.graph.nodes:
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, torch.device):
                v = v.type
            new_kwargs[k] = v
        node.kwargs = new_kwargs
    for node in fx_module.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    fx_module.graph.lint()
    fx_module.recompile()
    return torch.jit.script(fx_module)  # type: ignore


def _decorate_script_module(script_module: torch.jit.ScriptModule, expected_inputs, expected_outputs):
    for i, input_value in enumerate(script_module.graph.inputs()):  # type: ignore
        if input_value.debugName() == "self":
            script_module.graph.eraseInput(i)  # type: ignore
            break
    for input_value, expected_input in zip(script_module.graph.inputs(), expected_inputs):  # type: ignore
        input_value.setType(torch._C.TensorType.create_from_tensor(expected_input))
    for output_value, expected_output in zip(script_module.graph.outputs(), expected_outputs):  # type: ignore
        output_value.setType(torch._C.TensorType.create_from_tensor(expected_output))


def _create_onnx_proto(script_module):
    onnx_proto = _jit_graph_to_onnx_model(script_module.graph, torch.onnx.OperatorExportTypes.ONNX)
    return onnx_proto


def _create_onnx_model(onnx_proto):
    return onnx.ModelProto.FromString(onnx_proto)


def _create_onnx_session(onnx_proto, ep: str):
    # TODO(wechi): Add more EPs per PyTorch device types.
    # TODO(wechi): enable external allocators.
    return onnxruntime.InferenceSession(onnx_proto, providers=[ep])


def _infer_ep_from_device(device):
    if device.type == "cuda":
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def _get_onnx_devices(values: Tuple[torch.Tensor, ...]) -> Tuple[ORTC.OrtDevice, ...]:  # type: ignore
    assert all(value.device == values[0].device for value in values), "All values must be on the same device."

    def _device_id_or_zero(device_id: int) -> int:
        return device_id or 0

    devices: Tuple[ORTC.OrtDevice, ...] = tuple(  # type: ignore
        ORTC.OrtDevice(  # type: ignore
            _get_ort_device_type(value.device.type, _device_id_or_zero(value.device.index)),
            ORTC.OrtDevice.default_memory(),  # type: ignore
            _device_id_or_zero(value.device.index),
        )
        for value in values
    )
    return devices


def _run_onnx_session_with_ortvaluevector(
    sess: onnxruntime.InferenceSession,
    input_names: Tuple[str, ...],
    inputs: Tuple[torch.Tensor, ...],
    input_devices: Tuple[ORTC.OrtDevice, ...],  # type: ignore
    output_names: Tuple[str, ...],
    outputs: Tuple[torch.Tensor, ...],
    output_devices: Tuple[ORTC.OrtDevice, ...],  # type: ignore
) -> Tuple[torch.Tensor, ...]:
    _nvtx_range_push("contiguous")
    inputs = tuple(a.contiguous() for a in inputs)
    _nvtx_range_pop()

    _nvtx_range_push("push_back_batch")
    ort_inputs = ORTC.OrtValueVector()  # type: ignore
    ort_inputs.reserve(len(inputs))
    ort_outputs = ORTC.OrtValueVector()  # type: ignore
    dtypes = []
    shapes = []
    data_ptrs = []

    for value in inputs:
        dtypes.append(_NP_DTYPE[value.dtype])
        shapes.append(value.size())
        data_ptrs.append(value.data_ptr())
    ort_inputs.push_back_batch(inputs, data_ptrs, dtypes, shapes, input_devices)
    _nvtx_range_pop()

    _nvtx_range_push("run_with_ortvaluevector")
    run_options = onnxruntime.RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    sess.run_with_ortvaluevector(run_options, input_names, ort_inputs, output_names, ort_outputs, output_devices)
    _nvtx_range_pop()

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
    allclose = True if real_atol <= atol or real_rtol <= rtol else False
    if not allclose:
        raise RuntimeError(
            "ONNX output doesn't match baseline output with "
            + f"actual rtol={real_rtol} and actual atol={real_atol} "
            + f"but expected rtol={rtol} and expected atol={atol}."
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

    def __init__(self, ep: str = ""):
        self._supported_ops = OrtOperatorSupport()
        # TODO: this is a naive implementation of cache without proper guard
        self._partitioner_cache: Dict[torch.fx.GraphModule, torch.fx.GraphModule] = {}
        # TODO: this is a naive implementation of cache without proper guard, this will only work for identical inputs
        self._ort_execution_info = OrtExecutionInfo()

        self.ep = ep

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
            prim_outputs = FakeTensorProp(graph_module).propagate(*args, **kwargs)
            self._ort_execution_info.example_outputs[graph_module] = prim_outputs
            # Compile the torch.fx.GraphModule into a torch.jit.ScriptModule.
            script_module = _fx_to_torchscript(graph_module)
            # Post-processing step to add expected input and output type information
            # to the graph in torch.jit.ScriptModule. Expected inputs is "args" and "kwargs"
            # while expected outputs is "prim_outputs".
            if isinstance(prim_outputs, tuple):
                _decorate_script_module(script_module, args, prim_outputs)
            else:
                _decorate_script_module(script_module, args, (prim_outputs,))
            # Generate ONNX ModelProto from torch._C.Graph.
            onnx_proto = _create_onnx_proto(script_module)

            # Initialize a ORT session to execute this ONNX model.
            # TorchDynamo assumes all inputs/outputs are on the same device,
            # so we add execution provider only based on the first input's device.
            ep = self.ep if self.ep else _infer_ep_from_device(args[0].device)

            onnx_session = _create_onnx_session(onnx_proto, ep)
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
                onnx_session, input_names, args, input_devices, output_names, prim_outputs, output_devices
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
                onnx_session, input_names, args, input_devices, output_names, (prim_outputs,), output_devices
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
            prim_graph_module = make_fx(graph_module, decomposition_table=_ATEN2ATEN_DECOMP)(*args)
            # TODO(wechi): this is required for removing aten::_to_copy in _replace_to_copy_with_to.
            # We need input and output tensors' devices to decide if aten::_to_copy is just a Cast.
            FakeTensorProp(prim_graph_module).propagate(*args)
            _replace_to_copy_with_to(prim_graph_module)
            partitioner = CapabilityBasedPartitioner(
                prim_graph_module, self._supported_ops, allows_single_node_partition=False
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
