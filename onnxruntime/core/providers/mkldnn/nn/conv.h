// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"
#include "../mkldnn_execution_provider.h"

namespace onnxruntime {
namespace mkl_dnn {
template <typename T>
class Conv final : public onnxruntime::Conv<T> {
 public:
  explicit Conv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info) {
      provider_ = (const_cast<MKLDNNExecutionProvider*>(
        dynamic_cast<const MKLDNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  Status Compute(OpKernelContext* context) const override;

std::shared_ptr<mkldnn::memory> Reorder_weights(OpKernelContext* context,
	const mkldnn::memory::dims& src_dims, const mkldnn::memory::dims& filter_dims,
	const mkldnn::memory::dims& bias_dims, const mkldnn::memory::dims& dst_dims,
	const mkldnn::memory::dims& strides, const mkldnn::memory::dims& dilations,
	const mkldnn::memory::dims& padding_left, const mkldnn::memory::dims& padding_right,
	mkldnn::memory::format& filter_format,
	mkldnn::memory::dims& filter_dims_mkl) const;
 private:
   MKLDNNExecutionProvider * provider_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
