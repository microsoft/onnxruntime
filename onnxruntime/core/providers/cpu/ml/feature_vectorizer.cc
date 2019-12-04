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
static void CopyWithCast(typename gsl::span<const T>::const_iterator begin,
                         typename gsl::span<const T>::const_iterator end,
                         gsl::span<float>::iterator out_iter);

Status FeatureVectorizer::Compute(OpKernelContext* context) const {
  int input_count = context->NumVariadicInputs(0);
  ORT_ENFORCE(input_count >= 0 && static_cast<size_t>(input_count) == input_dimensions_.size(), "Number of inputs (",
              input_count, ") does not match number of inputdimensions values (", input_dimensions_.size(), ").");

  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  const auto& x_dims = X.Shape().GetDims();

  // assumes all inputs have the same batch size
  int64_t N = X.Shape().NumDimensions() == 1 ? 1 : x_dims[0];

  // initialize all the output to 0.f
  Tensor* Y = context->Output(0, TensorShape({N, total_dimensions_}));
  auto Y_data = Y->template MutableData<float>();

  auto out = gsl::make_span(Y_data, Y->Shape().Size());

  // init all to 0.f so we don't need to do that each loop if we have to add padding
  std::fill_n(out.data(), out.size(), 0.f);

  int64_t feature_offset = 0;

  // for each feature, write out its data in one pass
  for (int index = 0; index < input_count; ++index) {
    const auto* input_tensor_ptr = context->Input<Tensor>(index);
    ORT_ENFORCE(input_tensor_ptr != nullptr);
    auto& input_tensor = *input_tensor_ptr;

    auto feature_size = input_dimensions_[index];

    auto data_type = input_tensor.DataType();
    auto cur_out = out.begin() + feature_offset;

    if (utils::IsPrimitiveDataType<float>(data_type)) {
      // straight copy for float to float
      VectorizeTensor<float>(input_tensor, feature_size, total_dimensions_, cur_out);
    } else if (utils::IsPrimitiveDataType<int32_t>(data_type)) {
      VectorizeTensor<int32_t>(input_tensor, feature_size, total_dimensions_, cur_out);
    } else if (utils::IsPrimitiveDataType<int64_t>(data_type)) {
      VectorizeTensor<int64_t>(input_tensor, feature_size, total_dimensions_, cur_out);
    } else if (utils::IsPrimitiveDataType<double>(data_type)) {
      VectorizeTensor<double>(input_tensor, feature_size, total_dimensions_, cur_out);
    } else {
      // should never happen. graph validation should have failed
      ORT_THROW("Invalid input type:", data_type);
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
  auto& input_dims = shape.GetDims();

  auto input_size = input_dims.size() == 1 ? input_dims[0] : input_tensor.Shape().SizeFromDimension(1);
  auto N = input_dims.size() == 1 ? 1 : input_dims[0];

  // if there's extra data, ignore it
  auto stride = input_size;
  if (input_size > feature_size) {
    stride = feature_size;
  }

  auto data = input_tensor.template Data<T>();
  auto input = gsl::make_span(data, shape.Size());
  auto input_iter = input.cbegin();

  for (int i = 0; i < N;) {
    // copy each row to the output. iters are passed by value
    CopyWithCast<T>(input_iter, input_iter + stride, out_iter);

    // skip to start of next input row, and start of next output
    // if we have more input. otherwise we go past then end of the input and the bounds checking errors out
    if (++i < N) {
      input_iter += input_size;
      out_iter += sum_input_dimensions;
    }
  }
}

template <typename T>
static void CopyWithCast(typename gsl::span<const T>::const_iterator begin,
                         typename gsl::span<const T>::const_iterator end,
                         gsl::span<float>::iterator out_iter) {
  std::for_each(begin, end,
                [&out_iter](const typename gsl::span<T>::const_reference value) {
                  *out_iter = static_cast<float>(value);
                  ++out_iter;
                });
}

}  // namespace ml
}  // namespace onnxruntime
