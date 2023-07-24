// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/mlas/inc/mlas.h"
#include "core/framework/float16.h"
#include "core/providers/cpu/activation/activations.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

namespace onnxruntime {
namespace functors {

template <>
struct Relu<MLFloat16> : public ElementWiseRangedTransform<MLFloat16> {
  MLAS_ACTIVATION Activation;
  Status Init(const onnxruntime::NodeAttributes&) {
    Activation.ActivationKind = MlasReluActivation;
    return Status::OK();
  }
  GSL_SUPPRESS(r .11)
  ElementWiseRangedTransform<MLFloat16>* Copy() const final {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;  // redundant?
    return new T2(*this);
  }
  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    MLFloat16* output_ptr = this->output + first;
    const MLFloat16* input_ptr = this->input + first;

    // Linux compilation pipeline complained memcpy_s does not exists?!
    // memcpy_s(output_ptr, len * sizeof(MLFloat16), input_ptr, len * sizeof(MLFloat16));
    memcpy(output_ptr, input_ptr, len * sizeof(MLFloat16));

    MlasFp16Activation(&Activation, output_ptr, 1, len, len);
  }
};

template <>
struct LeakyRelu<MLFloat16> : public ElementWiseRangedTransform<MLFloat16> {
  MLAS_ACTIVATION Activation;
  Status Init(const onnxruntime::NodeAttributes& attributes) {
    Activation.ActivationKind = MlasLeakyReluActivation;
    return (GetFloatParam("alpha", attributes, Activation.Parameters.LeakyRelu.alpha));
  }
  GSL_SUPPRESS(r .11)
  ElementWiseRangedTransform<MLFloat16>* Copy() const final {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  };

  float Cost() const final {
    return 2.0f;
  }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    MLFloat16* output_ptr = this->output + first;
    const MLFloat16* input_ptr = this->input + first;
    // Linux compilation pipeline complained memcpy_s does not exists?!
    // memcpy_s(output_ptr, len * sizeof(MLFloat16), input_ptr, len * sizeof(MLFloat16));
    memcpy(output_ptr, input_ptr, len * sizeof(MLFloat16));

    MlasFp16Activation(&Activation, output_ptr, 1, len, len);
  }
};

// TODO Add the following activations:
//    MlasTanhActivation,
//    MlasLogisticActivation,
//    MlasClipActivation,
//    MlasHardSigmoidActivation,

}  // namespace functors
}  // namespace onnxruntime

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
