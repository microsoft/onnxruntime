// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/random_generator.h"
#include <chrono>
#include <random>
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T1, typename T2>
class Dropout final: public OpKernel {
 public:
  Dropout(const OpKernelInfo& info) : OpKernel{info} {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<RandomGenerator>(seed);
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<RandomGenerator> generator_;
};

namespace {
constexpr float k_default_ratio{0.5f};

template <typename T2>
float GetRatioOrDefault(const Tensor* ratio_tensor) {
  if (ratio_tensor) {
    ORT_ENFORCE(ratio_tensor->Shape().Size() == 1, "ratio input should have a single value.");
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
    const float ratio_value = *ratio_tensor->Data<T2>();
    ORT_ENFORCE(0.0f <= ratio_value && ratio_value < 1.0f, "ratio must be in the range [0, 1)");
    return ratio_value;
  }
  return k_default_ratio;
}
}  // namespace

template <typename T1, typename T2>
Status Dropout<T1, T2>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  auto X_span = X->DataAsSpan<T1>();
  const Tensor* ratio = context->Input<Tensor>(1);  // optional
  const float ratio_value = GetRatioOrDefault<T2>(ratio);
  const auto& X_shape = X->Shape();
  Tensor* Y = context->Output(0, X_shape);
  auto Y_span = Y->MutableDataAsSpan<T1>();
  Tensor* mask = context->Output(1, X_shape);  // optional
  std::unique_ptr<bool[]> temp_mask_buffer{};  // temporary buffer to use if mask input is not provided
  auto mask_span = [&X_shape, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableDataAsSpan<bool>();
    temp_mask_buffer = std::make_unique<bool[]>(X_shape.Size());
    return gsl::make_span(temp_mask_buffer.get(), X_shape.Size());
  }();

  ORT_ENFORCE(!mask || mask->Shape() == X_shape, "X and mask should have the same shape");

  const Tensor* training_mode = context->Input<Tensor>(2);
  if ((0 == ratio_value) || (training_mode == nullptr || *(training_mode->Data<bool>()) == false)) {
    // drop none
    if (X_span.data() != Y_span.data()) {
      std::copy(X_span.begin(), X_span.end(), Y_span.begin());
    }

    if (mask != nullptr) {
      std::fill(mask_span.begin(), mask_span.end(), true);
    }

  } else {
    // drop some
    ConstEigenVectorArrayMap<T1> X_arr(X_span.data(), X_span.size());
    EigenVectorArrayMap<T1> Y_arr(Y_span.data(), Y_span.size());
    EigenVectorArrayMap<bool> mask_arr(mask_span.data(), mask_span.size());

    // generate mask
    {
      RandomGenerator& generator = generator_ != nullptr ? *generator_.get() : RandomGenerator::Default();
      std::default_random_engine rng(generator.NextSeed());
      std::uniform_real_distribution<float> dist{0.0f, 1.0f};
      mask_arr = Eigen::ArrayX<bool>::NullaryExpr(
          mask_arr.size(),
          [ratio_value, &dist, &rng]() { return dist(rng) >= ratio_value; });
    }

    Y_arr = mask_arr.cast<T1>() * X_arr / (1.0f - ratio_value);
  }

  return Status::OK();
}

}  // namespace onnxruntime
