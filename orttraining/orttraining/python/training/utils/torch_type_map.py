# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import torch

# Mapping from pytorch scalar type to onnx scalar type.
_CAST_PYTORCH_TO_ONNX = {
    "Byte": [torch.onnx.TensorProtoDataType.UINT8, torch.uint8],
    "Char": [torch.onnx.TensorProtoDataType.INT8, torch.int8],
    "Double": [torch.onnx.TensorProtoDataType.DOUBLE, torch.double],
    "Float": [torch.onnx.TensorProtoDataType.FLOAT, torch.float],
    "Half": [torch.onnx.TensorProtoDataType.FLOAT16, torch.half],
    "Int": [torch.onnx.TensorProtoDataType.INT32, torch.int],
    "Long": [torch.onnx.TensorProtoDataType.INT64, torch.int64],
    "Short": [torch.onnx.TensorProtoDataType.INT16, torch.short],
    "Bool": [torch.onnx.TensorProtoDataType.BOOL, torch.bool],
    "ComplexFloat": [torch.onnx.TensorProtoDataType.COMPLEX64, torch.complex64],
    "ComplexDouble": [torch.onnx.TensorProtoDataType.COMPLEX128, torch.complex128],
    "BFloat16": [torch.onnx.TensorProtoDataType.BFLOAT16, torch.bfloat16],
    # Not yet defined in torch.
    # "Float8E4M3FN": torch.onnx.TensorProtoDataType.FLOAT8E4M3FN,
    # "Float8E4M3FNUZ": torch.onnx.TensorProtoDataType.FLOAT8E4M3FNUZ,
    # "Float8E5M2": torch.onnx.TensorProtoDataType.FLOAT8E5M2,
    # "Float8E5M2FNUZ": torch.onnx.TensorProtoDataType.FLOAT8E5M2FNUZ,
    "Undefined": [torch.onnx.TensorProtoDataType.UNDEFINED, None],
}


_DTYPE_TO_ONNX = {torch_dtype: onnx_dtype for k, (onnx_dtype, torch_dtype) in _CAST_PYTORCH_TO_ONNX.items()}


def pytorch_dtype_to_onnx(dtype: torch.dtype) -> torch.onnx.TensorProtoDataType:
    """Converts a pytorch dtype to an onnx dtype."""
    return _DTYPE_TO_ONNX[dtype]


def pytorch_scalar_type_str_to_onnx(scalar_type: str) -> torch.onnx.TensorProtoDataType:
    """Converts a pytorch scalar type string to an onnx dtype."""
    try:
        return torch.onnx.JitScalarType.from_name(scalar_type).onnx_type()
    except AttributeError:
        return _CAST_PYTORCH_TO_ONNX[scalar_type][0]
