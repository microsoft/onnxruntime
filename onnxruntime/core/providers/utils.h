// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace utils {

common::Status OutputOptionalWithoutDataHelper(const ONNX_NAMESPACE::TypeProto& input_type_proto,
                                               OpKernelContext* context, int output_index);

}  // namespace utils
}  // namespace onnxruntime
