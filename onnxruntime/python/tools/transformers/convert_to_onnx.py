# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxruntime.transformers.models.gpt2.convert_to_onnx import *

# This file is for backward compatible.
# For other models like longformer or T5, please look at ./models/*/convert_to_onnx.py

if __name__ == '__main__':
    main()