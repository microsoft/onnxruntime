/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// NOTE: License applies to the multinomial implementation only.
// Portions Copyright (c) Microsoft Corporation

#include "core/providers/cpu/generator/random.h"

#ifdef _WIN32
#pragma warning(disable : 28020)
#endif

#include <algorithm>
#include <chrono>
#include <random>

#include "gsl/gsl"

#include "core/common/eigen_common_wrapper.h"
#include "core/common/safeint.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"
#include "core/util/math_cpuonly.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomNormal, Output, 0,
    float, double);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomUniform, Output, 0,
    float, double);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomNormalLike, Output, 0,
    float, double);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomUniformLike, Output, 0,
    float, double);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Multinomial, Output, 0,
    int32_t, int64_t);
}

using RandomNormalOutputTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomNormal, Output, 0);
using EnabledRandomNormalOutputTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomNormal, Output, 0);

using RandomUniformOutputTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomUniform, Output, 0);
using EnabledRandomUniformOutputTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomUniform, Output, 0);

using RandomNormalLikeOutputTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomNormalLike, Output, 0);
using EnabledRandomNormalLikeOutputTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomNormalLike, Output, 0);

using RandomUniformLikeOutputTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomUniformLike, Output, 0);
using EnabledRandomUniformLikeOutputTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, RandomUniformLike, Output, 0);

using MultinomialOutputTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Multinomial, Output, 0);
using EnabledMultinomialOutputTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Multinomial, Output, 0);

using EnabledRandomUniformComputeOutputTypes =
    utils::TypeSetUnion<
        EnabledRandomUniformOutputTypes,
        EnabledRandomUniformLikeOutputTypes>;

using EnabledRandomNormalComputeOutputTypes =
    utils::TypeSetUnion<
        EnabledRandomNormalOutputTypes,
        EnabledRandomNormalLikeOutputTypes>;

ONNX_CPU_OPERATOR_KERNEL(
    RandomNormal,
    1,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<RandomNormalOutputTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledRandomNormalOutputTypes>()),
    RandomNormal);

ONNX_CPU_OPERATOR_KERNEL(
    RandomUniform,
    1,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<RandomUniformOutputTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledRandomUniformOutputTypes>()),
    RandomUniform);

ONNX_CPU_OPERATOR_KERNEL(
    RandomNormalLike,
    1,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<RandomNormalLikeOutputTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledRandomNormalLikeOutputTypes>()),
    RandomNormalLike);

ONNX_CPU_OPERATOR_KERNEL(
    RandomUniformLike,
    1,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<RandomUniformLikeOutputTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledRandomUniformLikeOutputTypes>()),
    RandomUniformLike);

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#multinomial
ONNX_CPU_OPERATOR_KERNEL(
    Multinomial,
    7,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<MultinomialOutputTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledMultinomialOutputTypes>()),
    Multinomial);

template <typename T, typename TDistribution>
void GenerateData(std::default_random_engine& generator, TDistribution distribution, Tensor& tensor);

static Status RandomNormalCompute(float mean, float scale, std::default_random_engine& generator, TensorProto::DataType dtype, Tensor& Y);
static Status RandomUniformCompute(float high, float low, std::default_random_engine& generator, TensorProto::DataType dtype, Tensor& Y);

static Status CreateOutputTensorFromTensorShape(OpKernelContext* ctx, const Tensor& X, Tensor** Y);
static TensorProto::DataType InferDataType(const Tensor& tensor);

Status RandomNormal::Compute(OpKernelContext* ctx) const {
  Tensor& Y = *ctx->Output(0, shape_);

  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  auto status = RandomNormalCompute(mean_, scale_, generator_, dtype_, Y);

  return status;
}

Status RandomUniform::Compute(OpKernelContext* ctx) const {
  Tensor& Y = *ctx->Output(0, shape_);

  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  auto status = RandomUniformCompute(low_, high_, generator_, dtype_, Y);

  return status;
}

Status RandomNormalLike::Compute(OpKernelContext* ctx) const {
  const auto* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  Tensor* Y = nullptr;

  auto status = CreateOutputTensorFromTensorShape(ctx, X, &Y);
  ORT_RETURN_IF_ERROR(status);

  auto dtype = dtype_ != TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(X);

  if (dtype == TensorProto_DataType_UNDEFINED)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Could not infer data type from input tensor with data type ",
                           X.DataType());

  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  status = RandomNormalCompute(mean_, scale_, generator_, dtype, *Y);

  return status;
}

Status RandomUniformLike::Compute(OpKernelContext* ctx) const {
  const auto* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  Tensor* Y = nullptr;

  auto status = CreateOutputTensorFromTensorShape(ctx, X, &Y);
  ORT_RETURN_IF_ERROR(status);

  auto dtype = dtype_ != TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(X);

  if (dtype == TensorProto_DataType_UNDEFINED)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Could not infer data type from input tensor with data type ",
                           X.DataType());
  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  status = RandomUniformCompute(low_, high_, generator_, dtype, *Y);

  return status;
}

// Rank-2 tensor (matrix) of scalar type T.
template <typename T, typename IndexType = int64_t>
using Matrix = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>>;

template <typename T, typename IndexType = int64_t>
using ConstMatrix = Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>>;

template <typename T, typename IndexType = int64_t>
using EigenVector = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>>;

template <typename OutputType>
static Status MultinomialCompute(OpKernelContext* ctx,
                                 const Tensor& X,
                                 const int64_t batch_size,
                                 const int64_t num_classes,
                                 const int64_t num_samples,
                                 std::default_random_engine& generator,
                                 Tensor& Y) {
  if (!utils::HasType<EnabledMultinomialOutputTypes, OutputType>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output type not supported in this build.");
  }

  // implementation copied from Tensorflow with some changes such as using the std::uniform_real_distribution
  // instead of the Philox RNG.
  Eigen::array<int64_t, 2> X_dims = {{batch_size, num_classes}};
  ConstMatrix<float> logits = ConstMatrix<float>(X.template Data<float>(), X_dims);

  Eigen::array<int64_t, 2> Y_dims = {{batch_size, num_samples}};
  Matrix<OutputType> output = Matrix<OutputType>(Y.template MutableData<OutputType>(), Y_dims);

  // BEGIN create temporary tensor
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
  auto cdf_data = static_cast<double*>(alloc->Alloc(SafeInt<size_t>(sizeof(double)) * num_classes));
  BufferUniquePtr cdf_buffer(cdf_data, BufferDeleter(alloc));
  Eigen::array<int64_t, 1> cdf_dims = {{num_classes}};
  auto cdf = EigenVector<double>(cdf_data, cdf_dims);
  // END create temporary tensor

  std::uniform_real_distribution<double> dist(0.0, 1.0);  // TODO: should this be initialized per batch?

  for (int64_t b = 0; b < batch_size; ++b) {
    const float* logits_row = &(logits(b, 0));
    // Takes an along-class maximum (for numerical stability).
    float maxx = std::numeric_limits<float>::lowest();
    for (int64_t j = 0; j < num_classes; ++j) {
      if (Eigen::numext::isfinite(logits_row[j])) {
        maxx = std::max(maxx, logits_row[j]);
      }
    }
    const auto max_logit = static_cast<double>(maxx);

    // Precompute cumulative probability distribution across classes.
    // Note: This isn't normalized.
    cdf = (logits.chip<0>(b).cast<double>() - max_logit).exp();
    double running_total = 0;
    for (int64_t j = 0; j < num_classes; ++j) {
      if (Eigen::numext::isfinite(logits_row[j])) {
        running_total += cdf(j);
      }
      cdf(j) = running_total;
    }
    // Generate each sample.
    const double* cdf_begin = cdf.data();
    const double* cdf_end = cdf.data() + num_classes;
    for (int64_t j = 0; j < num_samples; ++j) {
      const double to_find = dist(generator) * running_total;
      auto found_iter = std::upper_bound(cdf_begin, cdf_end, to_find);
      output(b, j) = static_cast<OutputType>(std::distance(cdf_begin, found_iter));
    }
  }

  return Status::OK();
}

Status Multinomial::Compute(OpKernelContext* ctx) const {
  const auto* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  auto& X_dims = X.Shape().GetDims();

  if (X_dims.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Empty dimensions for input tensor");
  }

  const auto batch_size = X_dims[0];
  const auto num_classes = X_dims[1];

  // validate inputs
  if (batch_size < 1) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "batch_size is < 1");
  }
  if (num_classes < 1) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "num_classes is < 1");
  }
  if (num_samples_ < 1) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "num_samples is < 1");
  }

  Tensor* Y = ctx->Output(0, {batch_size, num_samples_});

  Status status = Status::OK();
  std::lock_guard<onnxruntime::OrtMutex> l(generator_mutex_);
  switch (output_dtype_) {
    case TensorProto::INT32: {
      status = MultinomialCompute<int32_t>(ctx, X, batch_size, num_classes, num_samples_, generator_, *Y);
      break;
    }
    case TensorProto::INT64: {
      status = MultinomialCompute<int64_t>(ctx, X, batch_size, num_classes, num_samples_, generator_, *Y);
      break;
    }
    default:
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid data type of ", output_dtype_);
  }

  return status;
}

// create output tensor using shape of input tensor
static Status CreateOutputTensorFromTensorShape(OpKernelContext* ctx, const Tensor& X, Tensor** Y) {
  *Y = ctx->Output(0, X.Shape());

  return Status::OK();
}

static TensorProto::DataType InferDataType(const Tensor& tensor) {
  auto elem_type = tensor.GetElementType();
  int dtype = TensorProto_DataType_UNDEFINED;

  if (TensorProto_DataType_FLOAT == elem_type || TensorProto_DataType_DOUBLE == elem_type) {
    dtype = elem_type;
  } else {
    // unsupported. return UNDEFINED
  }
  return static_cast<TensorProto::DataType>(dtype);
}

static Status RandomNormalCompute(float mean, float scale,
                                  std::default_random_engine& generator,
                                  TensorProto::DataType dtype, Tensor& Y) {
  bool handled = false;
  switch (dtype) {
    case TensorProto::FLOAT: {
      if (utils::HasType<EnabledRandomNormalComputeOutputTypes, float>()) {
        GenerateData<float, std::normal_distribution<float>>(
            generator, std::normal_distribution<float>{mean, scale}, Y);
        handled = true;
      }
      break;
    }
    case TensorProto::DOUBLE: {
      if (utils::HasType<EnabledRandomNormalComputeOutputTypes, double>()) {
        GenerateData<double, std::normal_distribution<double>>(
            generator, std::normal_distribution<double>{mean, scale}, Y);
        handled = true;
      }
      break;
    }
    default:
      break;
  }

  if (!handled) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output type not supported in this build: ", dtype);
  }

  return Status::OK();
}

static Status RandomUniformCompute(float low, float high,
                                   std::default_random_engine& generator,
                                   TensorProto::DataType dtype,
                                   Tensor& Y) {
  bool handled = false;
  switch (dtype) {
    case TensorProto::FLOAT: {
      if (utils::HasType<EnabledRandomUniformComputeOutputTypes, float>()) {
        GenerateData<float, std::uniform_real_distribution<float>>(
            generator, std::uniform_real_distribution<float>{low, high}, Y);
        handled = true;
      }
      break;
    }
    case TensorProto::DOUBLE: {
      if (utils::HasType<EnabledRandomUniformComputeOutputTypes, double>()) {
        GenerateData<double, std::uniform_real_distribution<double>>(
            generator, std::uniform_real_distribution<double>{low, high}, Y);
        handled = true;
      }
      break;
    }
    default:
      break;
  }

  if (!handled) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output type not supported in this build: ", dtype);
  }

  return Status::OK();
}

template <typename T, typename TDistribution>
void GenerateData(std::default_random_engine& generator, TDistribution distribution, Tensor& tensor) {
  T* out = tensor.MutableData<T>();
  for (int64_t i = 0, end = tensor.Shape().Size(); i < end; ++i) {
    *out = distribution(generator);
    ++out;
  }
}

}  // namespace onnxruntime
