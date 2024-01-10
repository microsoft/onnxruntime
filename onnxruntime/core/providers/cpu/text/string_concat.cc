// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_concat.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/common/common.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(StringConcat, 20,
                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
                         StringConcat);

Status StringConcat::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs broadcast_funcs{[](BroadcastHelper& broadcast_helper) {
                                              auto x = broadcast_helper.ScalarInput0<std::string>();
                                              auto y = broadcast_helper.SpanInput1<std::string>();
                                              auto y_iter = y.begin();
                                              auto output_iter = broadcast_helper.OutputSpan<std::string>().begin();
                                              const auto x_size = x.length();
                                              while (y_iter != y.end()) {
                                                output_iter->reserve(x_size + y_iter->length());
                                                output_iter->append(x);
                                                output_iter->append(*y_iter);
                                                y_iter++;
                                                output_iter++;
                                              }
                                            },
                                            [](BroadcastHelper& broadcast_helper) {
                                              auto x = broadcast_helper.SpanInput0<std::string>();
                                              auto x_iter = x.begin();
                                              auto y = broadcast_helper.ScalarInput1<std::string>();
                                              auto output_iter = broadcast_helper.OutputSpan<std::string>().begin();
                                              const auto y_size = y.length();
                                              while (x_iter != x.end()) {
                                                output_iter->reserve(y_size + x_iter->length());
                                                output_iter->append(*x_iter);
                                                output_iter->append(y);
                                                x_iter++;
                                                output_iter++;
                                              }
                                            },
                                            [](BroadcastHelper& broadcast_helper) {
                                              auto x_iter = broadcast_helper.SpanInput0<std::string>().begin();
                                              auto y_iter = broadcast_helper.SpanInput1<std::string>().begin();
                                              auto output = broadcast_helper.OutputSpan<std::string>();
                                              auto output_iter = output.begin();
                                              while (output_iter != output.end()) {
                                                output_iter->reserve(x_iter->length() + y_iter->length());
                                                output_iter->append(*x_iter);
                                                output_iter->append(*y_iter);
                                                x_iter++;
                                                y_iter++;
                                                output_iter++;
                                              }
                                            }};
  UntypedBroadcastTwo(*context, broadcast_funcs);
  return Status::OK();
}

}  // namespace onnxruntime
