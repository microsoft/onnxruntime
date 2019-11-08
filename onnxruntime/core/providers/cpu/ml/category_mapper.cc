// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/category_mapper.h"
#include <algorithm>
#include <gsl/gsl>
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    CategoryMapper,
    1,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>(),
                                                              DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>(),
                                                DataTypeImpl::GetTensorType<int64_t>()}),
    CategoryMapper);

Status CategoryMapper::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  const TensorShape& shape = X.Shape();
  Tensor& Y = *context->Output(0, TensorShape(shape));

  auto input_type = X.DataType();

  if (utils::IsDataTypeString(input_type)) {
    if (!utils::IsPrimitiveDataType<int64_t>(Y.DataType()))
      return Status(ONNXRUNTIME, FAIL, "Input of string must have output of int64");

    auto input = gsl::make_span(X.template Data<std::string>(), shape.Size());
    auto output = gsl::make_span(Y.template MutableData<int64_t>(), shape.Size());
    auto out = output.begin();

    // map isn't going to change so get end() once instead of calling inside the for_each loop
    const auto map_end = string_to_int_map_.end();

    std::for_each(input.cbegin(), input.cend(),
                  [&out, &map_end, this](const std::string& value) {
                    auto map_to = string_to_int_map_.find(value);
                    *out = map_to == map_end ? default_int_ : map_to->second;
                    ++out;
                  });
  } else {
    if (!utils::IsDataTypeString(Y.DataType()))
      return Status(ONNXRUNTIME, FAIL, "Input of int64 must have output of string ");

    auto input = gsl::make_span(X.template Data<int64_t>(), shape.Size());
    auto output = gsl::make_span(Y.template MutableData<std::string>(), shape.Size());
    auto out = output.begin();

    const auto map_end = int_to_string_map_.end();

    std::for_each(input.cbegin(), input.cend(),
                  [&out, &map_end, this](const int64_t& value) {
                    auto map_to = int_to_string_map_.find(value);
                    *out = map_to == map_end ? default_string_ : map_to->second;
                    ++out;
                  });
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
