// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "../mkldnn_execution_provider.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class Sum final : public onnxruntime::Sum_6<T> {
 public:
   explicit Sum(const OpKernelInfo& info) : onnxruntime::Sum_6<T>(info) {}

  Status Compute(OpKernelContext* context) const override;

private:
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
