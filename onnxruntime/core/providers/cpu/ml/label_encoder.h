// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/ml/ml_common.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace ml {

class LabelEncoder final : public OpKernel {
 public:
  LabelEncoder(const OpKernelInfo& info) : OpKernel(info) {
    std::vector<std::string> string_classes;

    ORT_THROW_IF_ERROR(info.GetAttrs<std::string>("classes_strings", string_classes));

    ORT_ENFORCE(info.GetAttr<std::string>("default_string", &default_string_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("default_int64", &default_int_).IsOK());

    auto num_entries = string_classes.size();

    string_to_int_map_.reserve(num_entries);
    int_to_string_map_.reserve(num_entries);

    for (size_t i = 0; i < num_entries; ++i) {
      const std::string& str = string_classes[i];

      string_to_int_map_[str] = i;
      int_to_string_map_[i] = str;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::unordered_map<std::string, int64_t> string_to_int_map_;
  std::unordered_map<int64_t, std::string> int_to_string_map_;

  std::string default_string_;
  int64_t default_int_;
};

template <typename TKey, typename TValue>
class LabelEncoder_2 final : public OpKernel {
 public:
  LabelEncoder_2(const OpKernelInfo& info) : OpKernel(info) {
    // Let the specialized member function to tell which fields to load.
    InitializeSomeFields(info);

    std::vector<TKey> keys;
    std::vector<TValue> values;

    ORT_THROW_IF_ERROR(info.GetAttrs<TKey>(_key_field_name, keys));
    ORT_THROW_IF_ERROR(info.GetAttrs<TValue>(_value_field_name, values));

    auto num_keys = keys.size();
    auto num_values = values.size();
    ORT_ENFORCE(num_keys == num_values,
                "The ", _key_field_name, " and ", _value_field_name, " attribtues in LabelEncoder ",
                "(name: ", info.node().Name(), ") must have the same length. ",
                "However, the number of key is ", num_keys, " and the number of ",
                "values is ", num_values, ".");
    _map.reserve(num_keys);
    for (size_t i = 0; i < num_keys; ++i)
      _map.emplace(keys[i], values[i]);
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    const TensorShape& shape = X->Shape();
    auto* Y = context->Output(0, shape);

    auto input = X->template DataAsSpan<TKey>();
    auto output = Y->template MutableDataAsSpan<TValue>();
    auto input_iter = input.begin();
    auto output_iter = output.begin();
    while (input_iter != input.end()) {
      const auto found = _map.find(*input_iter);
      *output_iter = found == _map.end() ? _default_value : found->second;
      output_iter++;
      input_iter++;
    }
    return Status::OK();
  }

 private:
  // Specialize this method to set attribute names. For example, if keys' type
  // is 64-bit integer, _key_field_name should be "keys_int64s". Field names
  // for other types can be found in ONNX spec.
  void InitializeSomeFields(const OpKernelInfo& info);

  // A collection of key-value pairs. Each (a_key, a_value) pair
  // means that the "a_key" in the input would be mapped to "a_value".
  // If _map doesn't contain "a_key", we use _default_value as its output.
  InlinedHashMap<TKey, TValue> _map;
  TValue _default_value;
  // ONNX attribute name to load keys.
  std::string _key_field_name;
  // ONNX attribute name to load values.
  std::string _value_field_name;
};

template <typename T>
std::vector<T> GetAttribute(const OpKernelInfo& info, const std::string& name, const std::string& tensor_name) {
  std::vector<T> attrs;
  auto result = info.GetAttrs<T>(name, attrs);
  if (!result.IsOK()) {
    ONNX_NAMESPACE::TensorProto attr_tensor_proto;
    result = info.GetAttr(tensor_name, &attr_tensor_proto);
    ORT_ENFORCE(result.IsOK(), "LabelEncoder is missing an attribute");
    return utils::ParseData<T>(attr_tensor_proto);
  }
  return attrs;
}

template <typename T>
T GetDefault(const OpKernelInfo& info, const std::string& attr_name, const T& backup) {
  ONNX_NAMESPACE::TensorProto attr_tensor_proto;
  auto result = info.GetAttr("default_tensor", &attr_tensor_proto);
  if (result.IsOK() && utils::HasDataType(attr_tensor_proto)) {
    auto default_value = utils::ParseData<T>(attr_tensor_proto);
    ORT_ENFORCE(default_value.size() == 1, "default_tensor must have exactly one element");
    return default_value[0];
  } else {
    T default_value;
    result = info.GetAttr<T>(attr_name, &default_value);
    if (result.IsOK()) {
      return default_value;
    } else {
      return backup;
    }
  }
}

template <typename TKey, typename TValue>
class LabelEncoder_4 final : public OpKernel {
 public:
  LabelEncoder_4(const OpKernelInfo& kernel_info) : OpKernel(kernel_info) {
    InitializeAttrFields(kernel_info);
    auto keys = GetAttribute<TKey>(kernel_info, _key_field_name, "keys_tensor");
    auto values = GetAttribute<TValue>(kernel_info, _value_field_name, "values_tensor");
    ORT_ENFORCE(keys.size() == values.size(), "Keys and values must have the same length.");
    for (size_t i = 0; i < keys.size(); ++i) {
      _map.emplace(keys[i], values[i]);
    }
  }
  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    const TensorShape& shape = X->Shape();
    auto* Y = context->Output(0, shape);

    auto input = X->template DataAsSpan<TKey>();
    auto output = Y->template MutableDataAsSpan<TValue>();
    auto input_iter = input.begin();
    auto output_iter = output.begin();
    while (input_iter != input.end()) {
      const auto found = _map.find(*input_iter);
      *output_iter = found == _map.end() ? _default_value : found->second;
      output_iter++;
      input_iter++;
    }
    return Status::OK();
  }

 private:
  void InitializeAttrFields(const OpKernelInfo& kernel_info);
  InlinedHashMapNaNSensitive<TKey, TValue> _map;
  TValue _default_value;
  std::string _key_field_name;
  std::string _value_field_name;
};

}  // namespace ml
}  // namespace onnxruntime
