// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_concat.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/common/common.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    StringConcat,
    20,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    StringConcat);

Status StringConcat::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs broadcast_funcs{
      [](BroadcastHelper& broadcast_helper) {
        auto x = broadcast_helper.ScalarInput0<std::string>();
        auto Y = broadcast_helper.SpanInput1<std::string>();
        auto output = broadcast_helper.OutputSpan<std::string>();
        const auto x_size = x.length();
        for (size_t i = 0; i < Y.size(); ++i) {
          output[i].reserve(x_size + Y[i].length());
          output[i].append(x);
          output[i].append(Y[i]);
        }
      },
      [](BroadcastHelper& broadcast_helper) {
        auto X = broadcast_helper.SpanInput0<std::string>();
        auto y = broadcast_helper.ScalarInput1<std::string>();
        auto output = broadcast_helper.OutputSpan<std::string>();
        const auto y_size = y.length();
        for (size_t i = 0; i < X.size(); ++i) {
          output[i].reserve(y_size + X[i].length());
          output[i].append(X[i]);
          output[i].append(y);
        }
      },
      [](BroadcastHelper& broadcast_helper) {
        auto X = broadcast_helper.SpanInput0<std::string>();
        auto Y = broadcast_helper.SpanInput1<std::string>();
        auto output = broadcast_helper.OutputSpan<std::string>();
        for (size_t i = 0; i < X.size(); ++i) {
          output[i].reserve(X[i].length() + Y[i].length());
          output[i].append(X[i]);
          output[i].append(Y[i]);
        }
      }};
  UntypedBroadcastTwo(*context, broadcast_funcs);
  return Status::OK();
}

}  // namespace onnxruntime
