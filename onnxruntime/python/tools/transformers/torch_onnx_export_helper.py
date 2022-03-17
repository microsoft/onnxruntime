#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import torch
from packaging.version import Version

def torch_onnx_export(
    model,
    args,
    f,
    input_names,
    output_names,
    example_outputs,
    dynamic_axes,
    do_constant_folding,
    opset_version,
    use_external_data_format):
    if Version(torch.__version__) >= Version("1.11.0"):
        torch.onnx.export(model=model,
                          args=args,
                          f=f,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=do_constant_folding,
                          opset_version=opset_version)
    else:
        torch.onnx.export(model=model,
                  args=args,
                  f=f,
                  input_names=input_names,
                  output_names=output_names,
                  example_outputs=example_outputs,
                  dynamic_axes=dynamic_axes,
                  do_constant_folding=do_constant_folding,
                  opset_version=opset_version,
                  use_external_data_format=use_external_data_format)
