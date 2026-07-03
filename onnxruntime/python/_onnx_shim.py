# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Compatibility shim to import ``onnx`` or ``onnx_light.onnx``.

When the environment variable ``USE_OPTIM_ONNX`` is set to ``1`` this module
re-exports ``onnx_light.onnx`` (and its submodules) so the rest of the
onnxruntime Python package never imports the upstream ``onnx`` package
directly.  Otherwise it falls back to the standard ``onnx`` package.  This
mirrors the ``-Donnxruntime_USE_ONNX_LIGHT=ON`` build option on the C++ side.

The selected module (and a curated set of submodules) is registered under this
module's own ``onnx`` namespace in :data:`sys.modules`, so every import style
used across the code base resolves through the shim::

    from onnxruntime._onnx_shim import onnx                 # onnx.helper.make_node(...)
    from onnxruntime._onnx_shim.onnx import TensorProto, helper
    from onnxruntime._onnx_shim.onnx.helper import make_node

Direct ``import onnx`` / ``from onnx import ...`` statements are banned by the
``onnxruntime/python`` ruff configuration (flake8-tidy-imports, ``TID251``); use
this shim instead.  The imports below intentionally go through
:func:`importlib.import_module` (a runtime string, not an ``import`` statement)
so this file is the single place that references the backend package by name.
"""

from __future__ import annotations

import importlib
import os
import sys

_USE_OPTIM_ONNX = os.environ.get("USE_OPTIM_ONNX", "0") == "1"

# Base package providing the ONNX Python API: onnx-light when opted in, the
# upstream onnx package otherwise.
_BASE = "onnx_light.onnx" if _USE_OPTIM_ONNX else "onnx"

onnx = importlib.import_module(_BASE)

# Submodules referenced across the onnxruntime Python package. They are imported
# eagerly so ``onnx.<sub>`` attribute access works after
# ``from onnxruntime._onnx_shim import onnx`` and registered under this shim's
# ``onnx`` namespace so ``from onnxruntime._onnx_shim.onnx.<sub> import ...``
# resolves. Missing submodules are tolerated: a backend that does not ship one
# simply does not register it, and code paths that need it fail only when they
# are actually exercised.
_SUBMODULES = (
    "helper",
    "numpy_helper",
    "external_data_helper",
    "shape_inference",
    "checker",
    "defs",
    "parser",
    "mapping",
    "version",
    "onnx_pb",
    "backend",
    "backend.base",
    "reference",
    "reference.op_run",
)


def _register_shim_namespace() -> None:
    """Registers ``onnx`` and its submodules under this shim's namespace."""
    shim_onnx_name = f"{__name__}.onnx"
    sys.modules[shim_onnx_name] = onnx
    for sub in _SUBMODULES:
        try:
            module = importlib.import_module(f"{_BASE}.{sub}")
        except ImportError:
            continue
        sys.modules[f"{shim_onnx_name}.{sub}"] = module


_register_shim_namespace()

__all__ = ["onnx"]
