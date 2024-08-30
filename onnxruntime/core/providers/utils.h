// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"
#endif

namespace onnxruntime {
namespace utils {

#if !defined(DISABLE_OPTIONAL_TYPE)
common::Status OutputOptionalWithoutDataHelper(const ONNX_NAMESPACE::TypeProto& input_type_proto,
                                               OpKernelContext* context, int output_index);
#endif

/// <summary>
/// Check if the reciprocal of 'scale' is a factor of 'n'.
///   e.g. a scale of 0.5 is 1/2, the reciprocal is 2, and 2 is a factor of any even number.
/// </summary>
bool ReciprocalIsAFactorOfN(int64_t n, float scale);

/// <summary>
/// Compute the output shape for broadcasting the given input shapes of lhs and rhs.
/// </summary>
common::Status ComputeBroadcastOutputShape(const std::string& node_name,
                                           const TensorShape& lhs_shape,
                                           const TensorShape& rhs_shape,
                                           TensorShape& out_shape);

}  // namespace utils
}  // namespace onnxruntime
