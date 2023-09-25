// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/svmregressor.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    SVMRegressor,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SVMRegressor<float>);

template <typename T>
SVMRegressor<T>::SVMRegressor(const OpKernelInfo& info)
    : OpKernel(info),
      SVMCommon(info),
      support_vectors_(info.GetAttrsOrDefault<float>("support_vectors")),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  int64_t vector_count = 0;
  ORT_ENFORCE(info.GetAttr<int64_t>("n_supports", &vector_count).IsOK());
  vector_count_ = narrow<ptrdiff_t>(vector_count);
  ORT_ENFORCE(info.GetAttrs<float>("rho", rho_).IsOK());
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", coefficients_).IsOK());
  ORT_ENFORCE(!coefficients_.empty());

  auto onec = info.GetAttrOrDefault<int64_t>("one_class", 0);
  one_class_ = (onec != 0);

  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  // length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size();
    mode_ = SVM_TYPE::SVM_LINEAR;
    set_kernel_type(KERNEL::LINEAR);
  }
}

template <typename T>
Status SVMRegressor<T>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);

  ptrdiff_t num_features = X->Shape().NumDimensions() == 1 ? narrow<ptrdiff_t>(X->Shape()[0]) : narrow<ptrdiff_t>(X->Shape()[1]);
  ptrdiff_t num_batches = X->Shape().NumDimensions() == 1 ? 1 : narrow<ptrdiff_t>(X->Shape()[0]);
  ORT_RETURN_IF_NOT(num_features == feature_count_ && num_features >= 0 && num_batches >= 0, "Invalid argument");

  // X: [num_batches, feature_count_] where features could be coefficients or support vectors
  // coefficients_: [vector_count_]
  // support_vectors_ : [vector_count_, feature_count_]

  // Y: [num_batches, 1]
  Tensor* Y = ctx->Output(0, {num_batches, 1});  // this op outputs for one target only
  const auto x_data = X->template DataAsSpan<T>();
  auto out = Y->MutableDataAsSpan<T>();

  concurrency::ThreadPool* threadpool = ctx->GetOperatorThreadPool();

  if (mode_ == SVM_TYPE::SVM_SVC) {
    AllocatorPtr allocator;
    auto status = ctx->GetTempSpaceAllocator(&allocator);
    ORT_RETURN_IF_ERROR(status);

    auto tmp_data = IAllocator::MakeUniquePtr<T>(allocator, num_batches * SafeInt<size_t>(vector_count_));
    auto tmp_data_span = gsl::make_span<T>(tmp_data.get(), num_batches * SafeInt<size_t>(vector_count_));

    // combine the input data with the support vectors and apply the kernel type
    // output is {num_batches, vector_count_}
    batched_kernel_dot<float>(x_data, support_vectors_, num_batches, vector_count_, feature_count_, 0.f, tmp_data_span,
                              threadpool);

    static const TensorShape rho_shape({1});

    // combine with coefficients and add rho_[0]
    onnxruntime::Gemm<T>::ComputeGemm(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
                                      num_batches, 1, vector_count_,
                                      1.f, tmp_data_span.data(), coefficients_.data(), 1.f,
                                      rho_.data(), &rho_shape,
                                      out.data(),
                                      threadpool);
  } else if (mode_ == SVM_TYPE::SVM_LINEAR) {
    // combine the coefficients with the input data and apply the kernel type
    batched_kernel_dot<float>(x_data, coefficients_, num_batches, 1, feature_count_, rho_[0], out, threadpool);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected mode:", static_cast<int>(mode_));
  }

  if (one_class_) {
    float* y = out.data();
    float* y_end = y + out.size();

    while (y < y_end) {
      *y = (*y > 0.f ? 1.f : -1.f);
      ++y;
    }
  }

  return Status::OK();
}
}  // namespace ml
}  // namespace onnxruntime
