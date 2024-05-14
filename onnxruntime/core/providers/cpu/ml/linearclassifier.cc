// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/linearclassifier.h"
#include "core/common/narrow.h"
#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    LinearClassifier,
    1,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<float>(),
                                  DataTypeImpl::GetTensorType<double>(),
                                  DataTypeImpl::GetTensorType<int32_t>(),
                                  DataTypeImpl::GetTensorType<int64_t>(),
                              })
        .TypeConstraint("T2", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<std::string>(),
                                  DataTypeImpl::GetTensorType<int64_t>(),
                              }),
    LinearClassifier);

LinearClassifier::LinearClassifier(const OpKernelInfo& info)
    : OpKernel(info),
      multi_class_(info.GetAttrOrDefault<int64_t>("multi_class", 0)),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))),
      intercepts_(info.GetAttrsOrDefault<float>("intercepts")),
      classlabels_strings_(info.GetAttrsOrDefault<std::string>("classlabels_strings")),
      classlabels_ints_(info.GetAttrsOrDefault<int64_t>("classlabels_ints")) {
  if (!info.GetAttrs<float>("coefficients", coefficients_).IsOK())
    ORT_ENFORCE(!coefficients_.empty());

  using_strings_ = !classlabels_strings_.empty();
  class_count_ = static_cast<ptrdiff_t>(intercepts_.size());
}

// Use GEMM for the calculations, with broadcasting of intercepts
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
//
// X: [num_batches, num_features]
// coefficients_: [num_targets, num_features]
// intercepts_: [num_targets]
// scores: X * coefficients_^T + intercepts_: [num_batches, num_targets]
void LinearClassifier::ComputeImpl(const gsl::span<const float> input,
                                   ptrdiff_t num_batches, ptrdiff_t num_features, ptrdiff_t num_targets,
                                   const std::vector<float>& coefficients,
                                   const std::vector<float>& intercepts,
                                   Tensor& labels_output, Tensor& scores_output,
                                   POST_EVAL_TRANSFORM post_transform,
                                   bool add_second_class,
                                   concurrency::ThreadPool* threadpool) const {
  const float* input_data = input.data();
  auto scores_output_data = scores_output.MutableDataAsSpan<float>();
  size_t scores_output_size = SafeInt<size_t>(num_batches) * num_targets * (add_second_class ? 2 : 1);
  ORT_ENFORCE(scores_output_data.size() >= scores_output_size,
              "Scores output is incorrect size. Expected:", scores_output_size,
              " Found:", scores_output_data.size());

  TensorShape intercepts_shape({num_targets});
  onnxruntime::Gemm<float>::ComputeGemm(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
                                        num_batches, num_targets, num_features,
                                        1.f, input_data, coefficients.data(), 1.f,
                                        intercepts.data(), &intercepts_shape,
                                        scores_output_data.data(),
                                        threadpool);

  float* score = scores_output_data.data();
  float* end_scores = score + (num_batches * num_targets);  // we haven't added extra targets yet so iterate the original scores

  if (num_targets == 1) {
    if (using_strings_) {
      std::string* y_out = labels_output.MutableData<std::string>();
      bool use_class_labels = classlabels_strings_.size() == 2;
      std::string positive_label = use_class_labels ? classlabels_strings_[1] : "1";
      std::string negative_label = use_class_labels ? classlabels_strings_[0] : "0";

      while (score < end_scores) {
        *y_out++ = (*score++ > 0) ? positive_label
                                  : negative_label;
      }
    } else {
      int64_t* y_out = labels_output.MutableData<int64_t>();
      bool use_class_labels = classlabels_ints_.size() == 2;
      int64_t positive_label = use_class_labels ? classlabels_ints_[1] : 1;
      int64_t negative_label = use_class_labels ? classlabels_ints_[0] : 0;

      while (score < end_scores) {
        *y_out++ = (*score++ > 0) ? positive_label
                                  : negative_label;
      }
    }
  } else {
    for (int64_t i = 0; i < num_batches; ++i) {
      int maxclass = 0;
      float maxweight = *score++;

      for (int j = 1; j < num_targets; ++j, ++score) {
        if (*score > maxweight) {
          maxweight = *score;
          maxclass = j;
        }
      }

      if (using_strings_) {
        labels_output.MutableData<std::string>()[i] = classlabels_strings_[maxclass];
      } else {
        labels_output.MutableData<int64_t>()[i] = classlabels_ints_[maxclass];
      }
    }
  }

  if (post_transform != POST_EVAL_TRANSFORM::NONE || add_second_class) {
    ml::batched_update_scores_inplace(scores_output_data, num_batches, num_targets, post_transform,
                                      add_second_class ? 1 : -1, false,
                                      threadpool);
  }
}

template <typename SrcType>
static void CastInputToFloat(const Tensor& in, gsl::span<float>& out) {
  size_t shape_size = static_cast<size_t>(in.Shape().Size());
  ORT_ENFORCE(shape_size == out.size());

  const SrcType* in_data = in.Data<SrcType>();
  float* out_data = out.data();
  for (size_t i = 0; i < shape_size; ++i) {
    *out_data++ = static_cast<float>(*in_data++);
  }
}

Status LinearClassifier::Compute(OpKernelContext* ctx) const {
  const auto& X = *ctx->Input<Tensor>(0);
  const TensorShape& input_shape = X.Shape();
  if (input_shape.NumDimensions() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input shape needs to be at least a single dimension.");
  }

  ptrdiff_t num_batches = input_shape.NumDimensions() == 1 ? 1 : narrow<ptrdiff_t>(input_shape[0]);
  ptrdiff_t num_features = input_shape.NumDimensions() == 1 ? narrow<ptrdiff_t>(
                                                                  input_shape[0])
                                                            : narrow<ptrdiff_t>(input_shape[1]);

  Tensor* Y = ctx->Output(0, {num_batches});

  int64_t output_classes = class_count_;
  bool add_second_class = false;
  if (class_count_ == 1 &&
      ((using_strings_ && classlabels_strings_.size() == 2) ||
       (!using_strings_ && classlabels_ints_.size() == 2))) {
    output_classes = 2;
    add_second_class = true;
  }

  Tensor* Z = ctx->Output(1, {num_batches, output_classes});

  concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool();

  gsl::span<const float> input;
  float* cast_buffer = nullptr;

  auto element_type = X.GetElementType();
  AllocatorPtr alloc;

  if (element_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    input = X.DataAsSpan<float>();
  } else {
    // at some point we need to convert to float as output Z has type 'tensor(float)'.
    // we have a fast GEMM implementation for float, so convert the input to float so we can use that.
    auto status = ctx->GetTempSpaceAllocator(&alloc);
    size_t num_elements = onnxruntime::narrow<size_t>(input_shape.Size());
    cast_buffer = reinterpret_cast<float*>(alloc->AllocArray(num_elements, sizeof(float)));
    auto cast_span = gsl::make_span<float>(cast_buffer, num_elements);

    switch (element_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        CastInputToFloat<int32_t>(X, cast_span);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        CastInputToFloat<int64_t>(X, cast_span);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        CastInputToFloat<double>(X, cast_span);
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input element type of ", element_type);
    }

    input = cast_span;
  }

  ComputeImpl(input, num_batches, num_features, class_count_, coefficients_, intercepts_,
              *Y, *Z, post_transform_, add_second_class, tp);

  if (cast_buffer != nullptr) {
    alloc->Free(cast_buffer);
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
