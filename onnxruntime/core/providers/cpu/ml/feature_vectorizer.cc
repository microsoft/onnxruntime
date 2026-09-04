// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/feature_vectorizer.h"

#include <gsl/gsl>

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    FeatureVectorizer,
    1,
    KernelDefBuilder().TypeConstraint("T1", std::vector<MLDataType>{
                                                DataTypeImpl::GetTensorType<int32_t>(),
                                                DataTypeImpl::GetTensorType<int64_t>(),
                                                DataTypeImpl::GetTensorType<float>(),
                                                DataTypeImpl::GetTensorType<double>()}),
    FeatureVectorizer);

template <typename T>
static void VectorizeTensor(const Tensor& input_tensor, int64_t feature_size, int64_t sum_input_dimensions,
                            typename gsl::span<float>::iterator out_iter);

template <typename T>
static void CopyWithCast(typename gsl::span<const T>::iterator begin,
                         typename gsl::span<const T>::iterator end,
                         gsl::span<float>::iterator out_iter);

Status FeatureVectorizer::Compute(OpKernelContext* context) const {
  int input_count = context->NumVariadicInputs(0);
  ORT_ENFORCE(input_count >= 0 && static_cast<size_t>(input_count) == input_dimensions_.size(), "Number of inputs (",
              input_count, ") does not match number of inputdimensions values (", input_dimensions_.size(), ").");

  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const auto get_input_batch_size = [](const Tensor& input_tensor, int index, int64_t& batch_size) -> Status {
    const auto& input_dims = input_tensor.Shape().GetDims();
    ORT_RETURN_IF_NOT(!input_dims.empty(), "FeatureVectorizer input ", index,
                      " must have at least 1 dimension.");
    batch_size = input_dims.size() == 1 ? 1 : input_dims[0];
    return Status::OK();
  };

  const Tensor& X = *tensor_pointer;
  // all inputs must have the same batch size
  int64_t N = 0;
  ORT_RETURN_IF_ERROR(get_input_batch_size(X, 0, N));

  for (int index = 1; index < input_count; ++index) {
    const auto* input_tensor_ptr = context->Input<Tensor>(index);
    ORT_RETURN_IF(input_tensor_ptr == nullptr, "FeatureVectorizer input ", index, " is missing.");
    int64_t input_rows = 0;
    ORT_RETURN_IF_ERROR(get_input_batch_size(*input_tensor_ptr, index, input_rows));
    ORT_RETURN_IF_NOT(input_rows == N,
                      "All inputs to FeatureVectorizer must have the same batch size. "
                      "Input 0 batch size: ",
                      N, ", input ", index, " batch size: ", input_rows, ".");
  }

  // initialize all the output to 0.f
  Tensor* Y = context->Output(0, {N, total_dimensions_});
  auto Y_data = Y->MutableData<float>();

  auto out = gsl::make_span(Y_data, onnxruntime::narrow<size_t>(Y->Shape().Size()));

  // init all to 0.f so we don't need to do that each loop if we have to add padding
  std::fill_n(out.data(), out.size(), 0.f);

  int64_t feature_offset = 0;

  // for each feature, write out its data in one pass
  for (int index = 0; index < input_count; ++index) {
    const auto* input_tensor_ptr = context->Input<Tensor>(index);
    ORT_RETURN_IF(input_tensor_ptr == nullptr, "FeatureVectorizer input ", index, " is missing.");
    auto& input_tensor = *input_tensor_ptr;

    auto feature_size = input_dimensions_[index];

    auto cur_out = out.begin() + onnxruntime::narrow<size_t>(feature_offset);

    if (input_tensor.IsDataType<float>()) {
      // straight copy for float to float
      VectorizeTensor<float>(input_tensor, feature_size, total_dimensions_, cur_out);
    } else if (input_tensor.IsDataType<int32_t>()) {
      VectorizeTensor<int32_t>(input_tensor, feature_size, total_dimensions_, cur_out);
    } else if (input_tensor.IsDataType<int64_t>()) {
      VectorizeTensor<int64_t>(input_tensor, feature_size, total_dimensions_, cur_out);
    } else if (input_tensor.IsDataType<double>()) {
      VectorizeTensor<double>(input_tensor, feature_size, total_dimensions_, cur_out);
    } else {
      // should never happen. graph validation should have failed
      ORT_THROW("Invalid input type:", input_tensor.DataType());
    }

    // move to start of next feature
    feature_offset += feature_size;
  }

  return Status::OK();
}  // namespace ml

template <typename T>
static void VectorizeTensor(const Tensor& input_tensor, int64_t feature_size, int64_t sum_input_dimensions,
                            typename gsl::span<float>::iterator out_iter) {
  auto& shape = input_tensor.Shape();
  auto input_dims = shape.GetDims();

  auto input_size = input_dims.size() == 1 ? input_dims[0] : input_tensor.Shape().SizeFromDimension(1);
  auto N = input_dims.size() == 1 ? 1 : input_dims[0];

  // if there's extra data, ignore it
  auto stride = input_size;
  if (input_size > feature_size) {
    stride = feature_size;
  }

  auto data = input_tensor.Data<T>();
  auto input = gsl::make_span(data, onnxruntime::narrow<size_t>(shape.Size()));
  auto input_iter = input.begin();

  for (int i = 0; i < N;) {
    // copy each row to the output. iters are passed by value
    CopyWithCast<T>(input_iter, input_iter + onnxruntime::narrow<size_t>(stride), out_iter);

    // skip to start of next input row, and start of next output
    // if we have more input. otherwise we go past then end of the input and the bounds checking errors out
    if (++i < N) {
      input_iter += onnxruntime::narrow<size_t>(input_size);
      out_iter += onnxruntime::narrow<size_t>(sum_input_dimensions);
    }
  }
}

template <typename T>
static void CopyWithCast(typename gsl::span<const T>::iterator begin,
                         typename gsl::span<const T>::iterator end,
                         gsl::span<float>::iterator out_iter) {
  std::for_each(begin, end,
                [&out_iter](const typename gsl::span<const T>::const_reference value) {
                  *out_iter = static_cast<float>(value);
                  ++out_iter;
                });
}

}  // namespace ml
}  // namespace onnxruntime
