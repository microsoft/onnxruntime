# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Callable

from onnx.onnx_ml_pb2 import GraphProto


class GraphTransformerRegistry:
    _TRANSFORMER_FUNCS = {}  # noqa: RUF012

    @classmethod
    def register(cls, target_modules: str, devices: str, priority: int, fn: Callable[[GraphProto], None]):
        modules = []
        if target_modules == "all":
            modules.append("all")
        else:
            modules = target_modules.split("|")
        for module in modules:
            if module in cls._TRANSFORMER_FUNCS:
                cls._TRANSFORMER_FUNCS[module].append((fn, devices, priority))
            else:
                cls._TRANSFORMER_FUNCS[module] = [(fn, devices, priority)]

    @classmethod
    def transform_all(cls, module_name: str, device: str, graph: GraphProto):
        transformers_to_apply = []
        if "all" in cls._TRANSFORMER_FUNCS:
            transformers_to_apply.extend(cls._TRANSFORMER_FUNCS["all"])
        if module_name in cls._TRANSFORMER_FUNCS:
            transformers_to_apply.extend(cls._TRANSFORMER_FUNCS[module_name])
        transformers_to_apply = [x for x in transformers_to_apply if x[1] == "all" or device in x[1]]
        transformers_to_apply.sort(key=lambda x: x[2], reverse=True)
        for fn, _, _ in transformers_to_apply:
            fn(graph)


# target_modules can be multiple module names separated by "|", or "all" means apply to all modules.
# devices can be multiple device types separated by "|" or "all" means apply to all devices.
def register_graph_transformer(target_modules: str = "all", devices: str = "all", priority: int = 0):
    def graph_transformer_wrapper(fn):
        GraphTransformerRegistry.register(target_modules, devices, priority, fn)
        return fn

    return graph_transformer_wrapper
