// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace op_kernel_type_control {

// specify allowed types here

// Examples:
// Specify allowed types per Op kernel arg:
//   ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES(kOnnxDomain, Cast, Input, 0, float, int64_t);
//   ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES(kOnnxDomain, Cast, Output, 0, float, int64_t);
// Specify allowed types globally:
//   ORT_SPECIFY_OP_KERNEL_GLOBAL_ALLOWED_TYPES(float, double, int32_t)

// @@insertion_point_begin(allowed_types)@@
// @@insertion_point_end(allowed_types)@@

}  // namespace op_kernel_type_control
}  // namespace onnxruntime
