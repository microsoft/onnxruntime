import inspect
import operator
import re
from contextlib import nullcontext
from typing import Any, Callable, Dict, Set, Tuple
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily, _get_current_dispatch_mode

import torch
import torch.fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map


class SymbolicTraceTorchOperatorMode(TorchDispatchMode):
    def __init__(self, root: torch.nn.Module, graph: "torch.fx.Graph", decomposition_table: Dict[Callable, Callable]):
        self.graph: "torch.fx.Graph" = graph
        self.real_value_to_fx_value: Dict[Any, "torch.fx.Node"] = {}
        self.name_pool: Set[str] = set()
        self.root: "torch.nn.Module" = root
        self.decomposition_table = decomposition_table

    def register_name(self, name: str):
        name_counter = 0
        while name in self.name_pool:
            name = re.sub("\d+$", "", name)
            name = f"{name}{name_counter}"
            name_counter += 1
        self.name_pool.add(name)
        return name

    def register_attr(self, name: str, value: Any):
        setattr(self.root, name, value)
        return self.graph.get_attr(name)

    def register_inputs(self, *real_values):
        for value in real_values:
            if value in self.real_value_to_fx_value:
                raise RuntimeError(f"Cannot register graph input {value} multiple times.")
            name = self.register_name("input")
            fx_value = self.graph.placeholder(name)
            self.real_value_to_fx_value[value] = fx_value

    def register_outputs(self, *real_values):
        for value in real_values:
            if isinstance(value, torch.Tensor) and value not in self.real_value_to_fx_value:
                raise RuntimeError("Output value is not in traced variable pool.")
        fx_values = [
            self.real_value_to_fx_value[real_value] if isinstance(real_value, torch.Tensor) else real_value
            for real_value in real_values
        ]
        self.graph.create_node("output", "output", (tuple(fx_values),), {})

    def track(self, func, result, *args, **kwargs):
        def map_to_fx_value(value):
            if value in self.real_value_to_fx_value:
                return self.real_value_to_fx_value[value]
            else:
                # This is an input. Let's register a placeholder for it.
                if isinstance(value, torch.Tensor):
                    name = self.register_name("constant")
                    fx_value = self.register_attr(name, value)
                    self.real_value_to_fx_value[value] = fx_value
                    return fx_value
                else:
                    return value

        fx_args = tree_map(map_to_fx_value, args)
        fx_kwargs = tree_map(map_to_fx_value, kwargs)

        if isinstance(result, (tuple, list)):
            name = self.register_name("result")
            node = self.graph.create_node("call_function", func, fx_args, fx_kwargs, name=name)
            if isinstance(result, torch.Tensor):
                self.real_value_to_fx_value[result] = node
            for i, v in enumerate(result):
                name_i = self.register_name("element")
                getitem_node_i = self.graph.create_node("call_function", operator.getitem, (node, i), {}, name=name_i)
                if isinstance(v, torch.Tensor):
                    self.real_value_to_fx_value[v] = getitem_node_i
        else:
            name = self.register_name("result")
            node = self.graph.create_node("call_function", func, fx_args, fx_kwargs, name=name)
            if isinstance(result, torch.Tensor):
                self.real_value_to_fx_value[result] = node

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in [torch.ops.prim.device.default]:
            return func(*args, **kwargs)

        if func in self.decomposition_table:
            with self:
                result = self.decomposition_table[func](*args, **kwargs)
                if result is not NotImplemented:
                    return result

        with self:
            result = func.decompose(*args, **kwargs)
            if result is not NotImplemented:
                return result

        result = func(*args, **kwargs)
        self.track(func, result, *args, **kwargs)
        return result


from contextlib import contextmanager, nullcontext


@contextmanager
def disable_autocast_cache():
    old_value = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(old_value)


def _trace_through_dispatcher(
    model: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    enable_fake_tensor_mode: bool = False,
    decomposition_table=None,
):
    if decomposition_table is None:
        # No operator decomposition during tracing!
        decomposition_table = {}

    root = torch.nn.Module()
    graph = torch.fx.Graph()
    signature = inspect.signature(model.forward)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    assert len(bound.kwargs) == 0, bound.kwargs

    fake_tensor_context = nullcontext()
    if enable_fake_tensor_mode:
        fake_tensor_context = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_tensor_context, SymbolicTraceTorchOperatorMode(
        root, graph, decomposition_table=decomposition_table
    ) as mode, disable_autocast_cache():
        flat_inputs, _ = tree_flatten(args)
        mode.register_inputs(*flat_inputs)
        outputs = model(*bound.args)
        flat_outputs, _ = tree_flatten(outputs)
        mode.register_outputs(*flat_outputs)
    return (torch.fx.GraphModule(root, graph), bound.args)


def maybe_disable_symbolic_trace_mode():
    mb_trace_mode = _get_current_dispatch_mode()
    if isinstance(mb_trace_mode, SymbolicTraceTorchOperatorMode):
        return _pop_mode_temporarily()
    else:
        return nullcontext()


_WRAPPED_CUSTOM_FUNCTIONS = set()


def wrap_custom_function(custom_function):
    # TorchScript is the intermediate IR in between FX and ONNX.
    # FX graph -> torch.jit.script -> TorchScript graph ->
    # TorchScript-based ONNX exporter -> ONNX graph
    # Without torch.jit.ignore, the torch.jit.script may trace
    # into the custom function, which is not what we want.
    torch.jit.ignore(custom_function)
    _WRAPPED_CUSTOM_FUNCTIONS.add(custom_function)

    def wrapper(*args, **kwargs):
        with maybe_disable_symbolic_trace_mode() as mode:
            result = custom_function(*args, **kwargs)
            if mode:
                mode.track(custom_function, result, *args, **kwargs)
            return result

    return wrapper
