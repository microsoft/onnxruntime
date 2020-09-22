// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/matmul.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class FusedMatMul final : public MatMul<float> {
 public:
  FusedMatMul(const OpKernelInfo& info) : MatMul<T>(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("alpha", &alpha_attr_));
    ORT_THROW_IF_ERROR(info.GetAttr("transA", &trans_a_attr_));
    ORT_THROW_IF_ERROR(info.GetAttr("transB", &trans_b_attr_));
  }
};

}  // namespace contrib
}  // namespace onnxruntime
