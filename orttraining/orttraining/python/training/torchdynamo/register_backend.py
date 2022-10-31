# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.optimizations.backends import BACKENDS
from torch._dynamo.optimizations.training import AotAutogradStrategy

from .ort_backend import OrtBackend


class AotOrt(AotAutogradStrategy):
    """Implement compiler interface to plug ORT into TorchDynam.

    Under the hood, AotOrt.compile is called inside functorch. See aot_function
    and aot_module in aot_autograd.py in PyTorch repo for more details. Basically,
    AotOrt.compile is mapped to forward graph compiler, fw_compile, and backward
    graph compiler, bw_compile, in aot_autograd.py.
    """

    def __init__(self, graph_module: torch.fx.GraphModule, example_inputs):
        super().__init__(graph_module, example_inputs)

        self.ort = OrtBackend()

    def candidate(self):
        return BACKENDS["aot_autograd"](
            # Graph to compile.
            self.gm,
            # Example inputs that self.gm can execute on.
            # That it, self.gm(*example_inputs) will run.
            self.example_inputs,
            # Forward graph's compiler.
            fw_compiler=self.ort,
            # Backward graph's compiler.
            bw_compiler=self.ort,
            # partition_fn splits training graph into forward and backward graphs.
            partition_fn=min_cut_rematerialization_partition,
        )


# Call stack:
# AotAutogradStrategy.compile_fn
#  AotAutogradStrategy.verified_candidate
#   AotOrt.candidate
aot_ort = AotOrt.compile_fn
