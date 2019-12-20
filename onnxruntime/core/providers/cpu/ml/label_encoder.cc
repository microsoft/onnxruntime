// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/label_encoder.h"
#include <algorithm>
#include <gsl/gsl>
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_VERSIONED_ML_KERNEL(
    LabelEncoder,
    1, 1,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>(),
                                                              DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>(),
                                                DataTypeImpl::GetTensorType<int64_t>()})
        .SinceVersion(1, 2),
    LabelEncoder);

Status LabelEncoder::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  const TensorShape& shape = X.Shape();
  Tensor& Y = *context->Output(0, TensorShape(shape));

  auto input_type = X.DataType();

  if (utils::IsDataTypeString(input_type)) {
    if (!utils::IsPrimitiveDataType<int64_t>(Y.DataType()))
      return Status(ONNXRUNTIME, FAIL, "Input of tensor(string) must have output of tensor(int64)");

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
      return Status(ONNXRUNTIME, FAIL, "Input of tensor(int64) must have output of tensor(string)");

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

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder,
    2,
    float_string,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_2<float, std::string>);

template <>
void LabelEncoder_2<float, std::string>::InitializeSomeFields(const OpKernelInfo& info) {
  _key_field_name = "keys_floats";
  _value_field_name = "values_strings";
  info.GetAttrOrDefault<std::string>("default_string", &_default_value, std::string("_Unused"));
};

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder,
    2,
    string_float,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()}),
    LabelEncoder_2<std::string, float>);

template <>
void LabelEncoder_2<std::string, float>::InitializeSomeFields(const OpKernelInfo& info) {
  _key_field_name = "keys_strings";
  _value_field_name = "values_floats";
  info.GetAttrOrDefault<float>("default_float", &_default_value, -0.0f);
};

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder,
    2,
    int64_float,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()}),
    LabelEncoder_2<std::int64_t, float>);

template <>
void LabelEncoder_2<std::int64_t, float>::InitializeSomeFields(const OpKernelInfo& info) {
  _key_field_name = "keys_int64s";
  _value_field_name = "values_floats";
  info.GetAttrOrDefault<float>("default_float", &_default_value, -0.0f);
};

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder,
    2,
    float_int64,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_2<float, std::int64_t>);

template <>
void LabelEncoder_2<float, std::int64_t>::InitializeSomeFields(const OpKernelInfo& info) {
  _key_field_name = "keys_floats";
  _value_field_name = "values_int64s";
  info.GetAttrOrDefault<std::int64_t>("default_int64", &_default_value, (std::int64_t)-1);
};

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder,
    2,
    int64_string,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_2<std::int64_t, std::string>)

template <>
void LabelEncoder_2<std::int64_t, std::string>::InitializeSomeFields(const OpKernelInfo& info) {
  _key_field_name = "keys_int64s";
  _value_field_name = "values_strings";
  info.GetAttrOrDefault<std::string>("default_string", &_default_value, std::string("_Unused"));
};

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder,
    2,
    string_int64,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_2<std::string, std::int64_t>)

template <>
void LabelEncoder_2<std::string, std::int64_t>::InitializeSomeFields(const OpKernelInfo& info) {
  _key_field_name = "keys_strings";
  _value_field_name = "values_int64s";
  info.GetAttrOrDefault<std::int64_t>("default_int64", &_default_value, (std::int64_t)-1);
};

}  // namespace ml
}  // namespace onnxruntime
