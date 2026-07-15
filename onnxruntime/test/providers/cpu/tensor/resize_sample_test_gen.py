# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This code is used to generate test data for the test case:
# TEST(ResizeOpTest, NoAntialias_AlignCorners_Cubic_Floor_NHWC)
# located in onnxruntime/test/providers/cpu/tensor/resize_op_test.cc

import torch
import torch.nn.functional as F

# Define input tensor
input_tensor = (
    torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[9, 10], [11, 12], [13, 14], [15, 16]],
            [[17, 18], [19, 20], [21, 22], [23, 24]],
        ]
    )
    .unsqueeze(0)
    .to(torch.float32)
)

output_shape = (1, 6, 8, 2)

# Apply resize operation
output_tensor = F.interpolate(
    input_tensor.permute(0, 3, 1, 2),  # Convert to NHWC to NCHW for PyTorch
    size=(6, 8),  # Note: PyTorch expects size in (height, width) format
    mode="bicubic",  # bicubic is the same as cubic in ONNX
    align_corners=True,
).permute(0, 2, 3, 1)  # Convert back to NHWC

print(output_tensor)
