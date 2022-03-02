// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

template <typename ScaleT>
struct GetScaleValueImpl {
  void operator()(const Tensor* scale, float& scale_value) const {
    ORT_ENFORCE(scale->Shape().Size() == 1, "Scale input should have a single value.");
    scale_value = static_cast<float>(*(scale->template Data<ScaleT>()));
    ORT_ENFORCE(scale_value != 0.0f, "Scale value must not be 0.");
  }
};

template <typename T>
void Impl_Scale(
    cudaStream_t stream,
    const T* input_data,
    const float scale_value,
    T* output_data,
    size_t count);
 }
}  // namespace onnxruntime
