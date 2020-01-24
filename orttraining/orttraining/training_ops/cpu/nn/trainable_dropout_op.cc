// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include "orttraining/training_ops/cpu/nn/trainable_dropout_op.h"

#include <chrono>

#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

namespace {
constexpr float k_default_ratio{0.5f};

float GetRatioOrDefault(const Tensor* ratio_tensor) {
  if (ratio_tensor) {
    ORT_ENFORCE(ratio_tensor->Shape().Size() == 1, "ratio input should have a single value.");
    const float ratio_value = *ratio_tensor->Data<float>();
    ORT_ENFORCE(0.0f <= ratio_value && ratio_value < 1.0f, "ratio must be in the range [0, 1)");
    return ratio_value;
  }
  return k_default_ratio;
}
}  // namespace

// TrainableDropout

ONNX_OPERATOR_KERNEL_EX(
    TrainableDropout,
    kOnnxDomain,
    9,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    TrainableDropout);

TrainableDropout::TrainableDropout(const OpKernelInfo& info)
    : OpKernel{info},
      random_seed_{
          info.GetAttrOrDefault<int64_t>(
              "seed", std::chrono::steady_clock::now().time_since_epoch().count())},
      rng_{static_cast<typename decltype(rng_)::result_type>(random_seed_)} {
}

Status TrainableDropout::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  if (X->DataType() == DataTypeImpl::GetType<float>()) {
    ORT_RETURN_IF_ERROR(ComputeImpl<float>(context));
  } else if (X->DataType() == DataTypeImpl::GetType<double>()) {
    ORT_RETURN_IF_ERROR(ComputeImpl<double>(context));
  } else {
    ORT_NOT_IMPLEMENTED("unsupported data type: ", X->DataType());
  }
  return Status::OK();
}

template <typename T>
Status TrainableDropout::ComputeImpl(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  auto X_span = X->DataAsSpan<T>();

  const Tensor* ratio = context->Input<Tensor>(1);  // optional
  const float ratio_value = GetRatioOrDefault(ratio);

  const auto& X_shape = X->Shape();

  Tensor* Y = context->Output(0, X_shape);
  auto Y_span = Y->MutableDataAsSpan<T>();

  Tensor* mask = context->Output(1, X_shape);  // optional
  std::unique_ptr<bool[]> temp_mask_buffer{};  // temporary buffer to use if mask input is not provided
  auto mask_span = [&X_shape, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableDataAsSpan<bool>();
    temp_mask_buffer = onnxruntime::make_unique<bool[]>(X_shape.Size());
    return gsl::make_span(temp_mask_buffer.get(), X_shape.Size());
  }();

  ORT_ENFORCE(Y->Shape() == X_shape, "X and Y should have the same shape");
  ORT_ENFORCE(!mask || mask->Shape() == X_shape, "X and mask should have the same shape");

  if (ratio_value == 0.0f) {
    // drop none
    if (X_span.data() != Y_span.data()) {
      std::copy(X_span.begin(), X_span.end(), Y_span.begin());
    }
    std::fill(mask_span.begin(), mask_span.end(), true);
  } else if (ratio_value < 1.0f) {
    // drop some
    ConstEigenVectorArrayMap<T> X_arr(X_span.data(), X_span.size());
    EigenVectorArrayMap<T> Y_arr(Y_span.data(), Y_span.size());
    EigenVectorArrayMap<bool> mask_arr(mask_span.data(), mask_span.size());

    // generate mask
    {
      std::uniform_real_distribution<float> dist{0.0f, 1.0f};
      std::lock_guard<std::mutex> rng_guard{rng_mutex_};
      mask_arr = Eigen::ArrayX<bool>::NullaryExpr(
          mask_arr.size(),
          [this, ratio_value, &dist]() { return dist(rng_) >= ratio_value; });
    }

    Y_arr = mask_arr.cast<T>() * X_arr / (1.0f - ratio_value);
  }

  return Status::OK();
}

// TrainableDropoutGrad

ONNX_OPERATOR_KERNEL_EX(
    TrainableDropoutGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    TrainableDropoutGrad);

Status TrainableDropoutGrad::Compute(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  if (dY->DataType() == DataTypeImpl::GetType<float>()) {
    ORT_RETURN_IF_ERROR(ComputeImpl<float>(context));
  } else if (dY->DataType() == DataTypeImpl::GetType<double>()) {
    ORT_RETURN_IF_ERROR(ComputeImpl<double>(context));
  } else {
    ORT_NOT_IMPLEMENTED("unsupported data type: ", dY->DataType());
  }
  return Status::OK();
}

template <typename T>
Status TrainableDropoutGrad::ComputeImpl(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  auto dY_span = dY->DataAsSpan<T>();

  const Tensor* mask = context->Input<Tensor>(1);
  auto mask_span = mask->DataAsSpan<bool>();

  const Tensor* ratio = context->Input<Tensor>(2);  // optional
  const float ratio_value = GetRatioOrDefault(ratio);

  const auto& dY_shape = dY->Shape();

  Tensor* dX = context->Output(0, dY_shape);
  auto dX_span = dX->MutableDataAsSpan<T>();

  ORT_ENFORCE(mask->Shape() == dY_shape, "dY and mask should have the same shape");
  ORT_ENFORCE(dX->Shape() == dY_shape, "dY and dX should have the same shape");

  if (ratio_value == 0.0f) {
    if (dY_span.data() != dX_span.data()) {
      std::copy(dY_span.begin(), dY_span.end(), dX_span.begin());
    }
  } else if (ratio_value < 1.0f) {
    ConstEigenVectorArrayMap<T> dY_arr(dY_span.data(), dY_span.size());
    ConstEigenVectorArrayMap<bool> mask_arr(mask_span.data(), mask_span.size());
    EigenVectorArrayMap<T> dX_arr(dX_span.data(), dX_span.size());

    dX_arr = mask_arr.cast<T>() * dY_arr / (1.0f - ratio_value);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
