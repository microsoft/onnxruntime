// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/label_encoder.h"
#include <algorithm>
#include "core/common/gsl.h"
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_VERSIONED_ML_KERNEL(
    LabelEncoder, 1, 1,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()})
        .SinceVersion(1, 2),
    LabelEncoder);

Status LabelEncoder::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  const TensorShape& shape = X.Shape();
  Tensor& Y = *context->Output(0, shape);

  if (X.IsDataTypeString()) {
    if (!Y.IsDataType<int64_t>())
      return Status(ONNXRUNTIME, FAIL, "Input of tensor(string) must have output of tensor(int64)");

    auto input = gsl::make_span(X.Data<std::string>(), onnxruntime::narrow<size_t>(shape.Size()));
    auto output = gsl::make_span(Y.MutableData<int64_t>(), onnxruntime::narrow<size_t>(shape.Size()));
    auto out = output.begin();

    // map isn't going to change so get end() once instead of calling inside the for_each loop
    const auto map_end = string_to_int_map_.end();

    std::for_each(input.begin(), input.end(), [&out, &map_end, this](const std::string& value) {
      auto map_to = string_to_int_map_.find(value);
      *out = map_to == map_end ? default_int_ : map_to->second;
      ++out;
    });
  } else {
    if (!Y.IsDataTypeString())
      return Status(ONNXRUNTIME, FAIL, "Input of tensor(int64) must have output of tensor(string)");

    auto input = gsl::make_span(X.Data<int64_t>(), onnxruntime::narrow<size_t>(shape.Size()));
    auto output = gsl::make_span(Y.MutableData<std::string>(), onnxruntime::narrow<size_t>(shape.Size()));
    auto out = output.begin();

    const auto map_end = int_to_string_map_.end();

    std::for_each(input.begin(), input.end(), [&out, &map_end, this](const int64_t& value) {
      auto map_to = int_to_string_map_.find(value);
      *out = map_to == map_end ? default_string_ : map_to->second;
      ++out;
    });
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, float_string,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_2<float, std::string>);

template <>
void LabelEncoder_2<float, std::string>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_strings";
  info.GetAttrOrDefault<std::string>("default_string", &default_value_, std::string("_Unused"));
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, string_float,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()}),
    LabelEncoder_2<std::string, float>);

template <>
void LabelEncoder_2<std::string, float>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_strings";
  value_field_name_ = "values_floats";
  info.GetAttrOrDefault<float>("default_float", &default_value_, -0.0f);
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, int64_float,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()}),
    LabelEncoder_2<std::int64_t, float>);

template <>
void LabelEncoder_2<std::int64_t, float>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_floats";
  info.GetAttrOrDefault<float>("default_float", &default_value_, -0.0f);
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, float_int64,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_2<float, std::int64_t>);

template <>
void LabelEncoder_2<float, std::int64_t>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_int64s";
  info.GetAttrOrDefault<std::int64_t>("default_int64", &default_value_, (std::int64_t)-1);
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, string_string,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_2<std::string, std::string>)

template <>
void LabelEncoder_2<std::string, std::string>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_strings";
  value_field_name_ = "values_strings";
  info.GetAttrOrDefault<std::string>("default_string", &default_value_, std::string("_Unused"));
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, float_float,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()}),
    LabelEncoder_2<float, float>)

template <>
void LabelEncoder_2<float, float>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_floats";
  info.GetAttrOrDefault<float>("default_float", &default_value_, -0.0f);
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, int64_string,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_2<std::int64_t, std::string>)

template <>
void LabelEncoder_2<std::int64_t, std::string>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_strings";
  info.GetAttrOrDefault<std::string>("default_string", &default_value_, std::string("_Unused"));
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, string_int64,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_2<std::string, std::int64_t>)

template <>
void LabelEncoder_2<std::string, std::int64_t>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_strings";
  value_field_name_ = "values_int64s";
  info.GetAttrOrDefault<std::int64_t>("default_int64", &default_value_, static_cast<std::int64_t>(-1));
}

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    LabelEncoder, 2, 3, int64_int64,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_2<std::int64_t, std::int64_t>)

template <>
void LabelEncoder_2<std::int64_t, std::int64_t>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_int64s";
  info.GetAttrOrDefault<std::int64_t>("default_int64", &default_value_, static_cast<std::int64_t>(-1));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, int64_int64,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_4<std::int64_t, std::int64_t>)

template <>
void LabelEncoder_4<std::int64_t, std::int64_t>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_int64s";
  default_value_ = GetDefault(kernel_info, "default_int64", static_cast<int64_t>(-1));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, int64_string,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_4<std::int64_t, std::string>)

template <>
void LabelEncoder_4<std::int64_t, std::string>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_strings";
  default_value_ = GetDefault(kernel_info, "default_string", std::string("_Unused"));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, int64_float,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()}),
    LabelEncoder_4<std::int64_t, float>)

template <>
void LabelEncoder_4<std::int64_t, float>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_floats";
  default_value_ = GetDefault(kernel_info, "default_float", 0.f);
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(LabelEncoder, 4, float_float,
                                  KernelDefBuilder()
                                      .TypeConstraint("T1",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()})
                                      .TypeConstraint("T2",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()}),
                                  LabelEncoder_4<float, float>)

template <>
void LabelEncoder_4<float, float>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_floats";
  default_value_ = GetDefault(kernel_info, "default_float", -0.f);
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, float_string,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_4<float, std::string>)

template <>
void LabelEncoder_4<float, std::string>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_strings";
  default_value_ = GetDefault(kernel_info, "default_string", std::string("_Unused"));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, float_int64,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_4<float, std::int64_t>)

template <>
void LabelEncoder_4<float, std::int64_t>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_int64s";
  default_value_ = GetDefault(kernel_info, "default_int64", static_cast<int64_t>(-1));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, string_int64,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_4<std::string, std::int64_t>)

template <>
void LabelEncoder_4<std::string, std::int64_t>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_strings";
  value_field_name_ = "values_int64s";
  default_value_ = GetDefault(kernel_info, "default_int64", static_cast<int64_t>(-1));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, string_float,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>()}),
    LabelEncoder_4<std::string, float>)

template <>
void LabelEncoder_4<std::string, float>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_strings";
  value_field_name_ = "values_floats";
  default_value_ = GetDefault(kernel_info, "default_float", 0.f);
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, string_string,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_4<std::string, std::string>)

template <>
void LabelEncoder_4<std::string, std::string>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_strings";
  value_field_name_ = "values_strings";
  default_value_ = GetDefault(kernel_info, "default_string", std::string("_Unused"));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, string_int16,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int16_t>()}),
    LabelEncoder_4<std::string, std::int16_t>)

template <>
void LabelEncoder_4<std::string, std::int16_t>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_strings";
  default_value_ = static_cast<std::int16_t>(GetDefault(kernel_info, "", static_cast<std::int16_t>(-1)));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(LabelEncoder, 4, double_double,
                                  KernelDefBuilder()
                                      .TypeConstraint("T1",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<double>()})
                                      .TypeConstraint("T2",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<double>()}),
                                  LabelEncoder_4<double, double>)

template <>
void LabelEncoder_4<double, double>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  default_value_ = GetDefault(kernel_info, "default_float", -0.);
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, double_string,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()}),
    LabelEncoder_4<double, std::string>)

template <>
void LabelEncoder_4<double, std::string>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  value_field_name_ = "values_strings";
  default_value_ = GetDefault(kernel_info, "default_string", std::string("_Unused"));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, string_double,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<double>()}),
    LabelEncoder_4<std::string, double>)

template <>
void LabelEncoder_4<std::string, double>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_strings";
  default_value_ = GetDefault(kernel_info, "default_float", -0.);
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, double_int64,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()}),
    LabelEncoder_4<double, std::int64_t>)

template <>
void LabelEncoder_4<double, std::int64_t>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  value_field_name_ = "values_int64s";
  default_value_ = GetDefault(kernel_info, "default_int64", static_cast<int64_t>(-1));
}

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    LabelEncoder, 4, int64_double,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::int64_t>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<double>()}),
    LabelEncoder_4<std::int64_t, double>)

template <>
void LabelEncoder_4<std::int64_t, double>::InitializeAttrFields(const OpKernelInfo& kernel_info) {
  key_field_name_ = "keys_int64s";
  default_value_ = GetDefault(kernel_info, "default_float", -0.);
}

}  // namespace ml
}  // namespace onnxruntime
