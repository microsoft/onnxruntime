# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Callable

from onnx.onnx_ml_pb2 import GraphProto


class GraphOptimizerRegistry:
    _OPTIMIZER_FUNCS = {}  # noqa: RUF012

    @classmethod
    def register(cls, target_modules: str, devices: str, priority: int, fn: Callable[[GraphProto], None]):
        modules = []
        if target_modules == "all":
            modules.append("all")
        else:
            modules = target_modules.split("|")
        for module in modules:
            if module in cls._OPTIMIZER_FUNCS:
                cls._OPTIMIZER_FUNCS[module].append((fn, devices, priority))
            else:
                cls._OPTIMIZER_FUNCS[module] = [(fn, devices, priority)]

    @classmethod
    def optimize_all(cls, module_name: str, device: str, graph: GraphProto):
        optimizers_to_apply = []
        if "all" in cls._OPTIMIZER_FUNCS:
            optimizers_to_apply.extend(cls._OPTIMIZER_FUNCS["all"])
        if module_name in cls._OPTIMIZER_FUNCS:
            optimizers_to_apply.extend(cls._OPTIMIZER_FUNCS[module_name])
        optimizers_to_apply = [x for x in optimizers_to_apply if x[1] == "all" or device in x[1]]
        optimizers_to_apply.sort(key=lambda x: x[2], reverse=True)
        for fn, _, _ in optimizers_to_apply:
            fn(graph)


# target_modules can be multiple module names separated by "|", or "all" means apply to all modules.
# devices can be multiple device types separated by "|" or "all" means apply to all devices.
def register_graph_optimizer(target_modules: str = "all", devices: str = "all", priority: int = 0):
    def graph_optimizer_wrapper(fn):
        GraphOptimizerRegistry.register(target_modules, devices, priority, fn)
        return fn

    return graph_optimizer_wrapper
