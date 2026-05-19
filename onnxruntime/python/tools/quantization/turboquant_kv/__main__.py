# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Module entry point: `python -m onnxruntime.quantization.turboquant_kv ...`"""

from .onnx_rewriter import main

if __name__ == "__main__":
    raise SystemExit(main())
