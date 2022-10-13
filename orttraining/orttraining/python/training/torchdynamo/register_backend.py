# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from functorch.compile import min_cut_rematerialization_partition
from torchdynamo.optimizations.backends import BACKENDS
from torchdynamo.optimizations.training import AotAutogradStrategy

from .ort_backend import OrtBackend


class AotOrt(AotAutogradStrategy):
    def __init__(self, graph_mmodule: torch.fx.GraphModule, example_inputs):
        super().__init__(graph_mmodule, example_inputs)

        self.ort = OrtBackend()
        self.populate_aten2aten_decomps()

    def populate_aten2aten_decomps(self):
        aten = torch.ops.aten
        default_decompositions = {
            aten.detach,
            aten.gelu_backward,
            aten.leaky_relu_backward,
            aten.sigmoid_backward,
            aten.threshold_backward,
            aten.hardtanh_backward,
            aten.hardsigmoid_backward,
            aten.hardswish_backward,
            aten.tanh_backward,
            aten.silu_backward,
            aten.elu_backward,
            aten.cudnn_batch_norm,
            aten.cudnn_batch_norm_backward,
            aten.masked_fill.Scalar,
            aten.masked_fill.Tensor,
            aten.elu,
            aten.leaky_relu,
            aten.hardtanh,
            aten.hardswish,
            aten.hardsigmoid,
            aten.rsub,
            aten.native_batch_norm_backward,
        }

        self.aten2aten_decompositions = torch._decomp.get_decompositions(default_decompositions)

    def candidate(self):
        return BACKENDS["aot_autograd"](
            self.gm,
            self.example_inputs,
            fw_compiler=self.ort,
            bw_compiler=self.ort,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=self.aten2aten_decompositions,
        )


class AOTAutogradOrtWithContext:
    def __init__(self):
        self.backend_ctx_ctor = lambda: torch.jit.fuser("none")

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        return AotOrt.compile_fn(gm, example_inputs)


aot_ort = AOTAutogradOrtWithContext()
