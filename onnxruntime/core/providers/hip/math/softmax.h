// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

template <typename T>
struct AccumulateType {};
template <>
struct AccumulateType<float> { using type = float; };
template <>
struct AccumulateType<MLFloat16> { using type = float; };
template <>
struct AccumulateType<double> { using type = double; };
template <typename T>
using AccType = typename AccumulateType<T>::type;

template <typename T>
Status SoftMaxComputeHelper(
    const T* input,
    const TensorShape& shape,
    T* Y,
    //hipdnnHandle_t handle,
    int64_t axis);

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_forward(output_t* dst, const input_t* src, int softmax_elements, int softmax_elements_stride, int batch_count);

template <typename T>
class Softmax final : public HipKernel {
 public:
  Softmax(const OpKernelInfo& info) : HipKernel{info} {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(1));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace hip
}  // namespace onnxruntime
