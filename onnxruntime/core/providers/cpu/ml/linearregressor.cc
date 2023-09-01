// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/linearregressor.h"
#include "core/common/narrow.h"
#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    LinearRegressor,
    1,
    // KernelDefBuilder().TypeConstraint("T", std::vector<MLDataType>{
    //                                            DataTypeImpl::GetTensorType<float>(),
    //                                            DataTypeImpl::GetTensorType<double>()}),
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    LinearRegressor);

LinearRegressor::LinearRegressor(const OpKernelInfo& info)
    : OpKernel(info),
      intercepts_(info.GetAttrsOrDefault<float>("intercepts")),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  ORT_ENFORCE(info.GetAttr<int64_t>("targets", &num_targets_).IsOK());
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", coefficients_).IsOK());

  // use the intercepts_ if they're valid
  use_intercepts_ = intercepts_.size() == static_cast<size_t>(num_targets_);
}

// Use GEMM for the calculations, with broadcasting of intercepts
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
//
// X: [num_batches, num_features]
// coefficients_: [num_targets, num_features]
// intercepts_: optional [num_targets].
// Output: X * coefficients_^T + intercepts_: [num_batches, num_targets]
template <typename T>
static Status ComputeImpl(const Tensor& input, ptrdiff_t num_batches, ptrdiff_t num_features, ptrdiff_t num_targets,
                          const std::vector<float>& coefficients,
                          const std::vector<float>* intercepts, Tensor& output,
                          POST_EVAL_TRANSFORM post_transform,
                          concurrency::ThreadPool* threadpool) {
  const T* input_data = input.Data<T>();
  T* output_data = output.MutableData<T>();

  if (intercepts != nullptr) {
    TensorShape intercepts_shape({num_targets});
    onnxruntime::Gemm<T>::ComputeGemm(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
                                      num_batches, num_targets, num_features,
                                      1.f, input_data, coefficients.data(), 1.f,
                                      intercepts->data(), &intercepts_shape,
                                      output_data,
                                      threadpool);
  } else {
    onnxruntime::Gemm<T>::ComputeGemm(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
                                      num_batches, num_targets, num_features,
                                      1.f, input_data, coefficients.data(), 1.f,
                                      nullptr, nullptr,
                                      output_data,
                                      threadpool);
  }

  if (post_transform != POST_EVAL_TRANSFORM::NONE) {
    ml::batched_update_scores_inplace(gsl::make_span(output_data, SafeInt<size_t>(num_batches) * num_targets),
                                      num_batches, num_targets, post_transform, -1, false, threadpool);
  }

  return Status::OK();
}

Status LinearRegressor::Compute(OpKernelContext* ctx) const {
  Status status = Status::OK();

  const auto& X = *ctx->Input<Tensor>(0);
  const auto& input_shape = X.Shape();

  if (input_shape.NumDimensions() > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input shape had more than 2 dimension. Dims=",
                           input_shape.NumDimensions());
  }

  ptrdiff_t num_batches = input_shape.NumDimensions() <= 1 ? 1 : narrow<ptrdiff_t>(input_shape[0]);
  ptrdiff_t num_features = input_shape.NumDimensions() <= 1 ? narrow<ptrdiff_t>(input_shape.Size())
                                                            : narrow<ptrdiff_t>(input_shape[1]);
  Tensor& Y = *ctx->Output(0, {num_batches, num_targets_});
  concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool();

  auto element_type = X.GetElementType();

  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      status = ComputeImpl<float>(X, num_batches, num_features, narrow<ptrdiff_t>(num_targets_), coefficients_,
                                  use_intercepts_ ? &intercepts_ : nullptr,
                                  Y, post_transform_, tp);

      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      // TODO: Add support for 'double' to the scoring functions in ml_common.h
      // once that is done we can just call ComputeImpl<double>...
      // Alternatively we could cast the input to float.
    }
    default:
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported data type of ", element_type);
  }

  return status;
}

}  // namespace ml
}  // namespace onnxruntime
