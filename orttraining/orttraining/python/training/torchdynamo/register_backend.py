# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.optimizations.training import aot_autograd
from .ort_backend import OrtBackend


# This is the underlying compiler for ALL training graphs if the user uses ORT
# to optimize training graphs.
# A global compiler is used here, so that cached compilation results can be reused
# across different graphs (each graph calls this compiler once).
DEFAULT_BACKEND = OrtBackend()

# Wrap ORT as a compiler in Dynamo.
#
# Under the hood, AotOrt.compile is called inside functorch. See aot_function
# and aot_module in aot_autograd.py in PyTorch repo for more details. Basically,
# AotOrt.compile is mapped to forward graph compiler, fw_compile, and backward
# graph compiler, bw_compile, in aot_autograd.py.
aot_ort = aot_autograd(fw_compiler=DEFAULT_BACKEND, partition_fn=min_cut_rematerialization_partition)
