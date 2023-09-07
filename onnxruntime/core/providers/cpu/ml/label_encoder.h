// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/ml/ml_common.h"

namespace onnxruntime {
namespace ml {

class LabelEncoder final : public OpKernel {
 public:
  LabelEncoder(const OpKernelInfo& info) : OpKernel(info) {
    std::vector<std::string> string_classes;

    ORT_ENFORCE(info.GetAttrs<std::string>("classes_strings", string_classes).IsOK());

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

    ORT_ENFORCE(info.GetAttrs<TKey>(_key_field_name, keys).IsOK());
    ORT_ENFORCE(info.GetAttrs<TValue>(_value_field_name, values).IsOK());

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
    const auto* tensor_pointer = context->Input<Tensor>(0);
    if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    const Tensor& X = *tensor_pointer;
    const TensorShape& shape = X.Shape();
    Tensor& Y = *context->Output(0, shape);

    auto input = X.template DataAsSpan<TKey>();
    auto output = Y.template MutableDataAsSpan<TValue>();

    for (int64_t i = 0; i < shape.Size(); ++i) {
      const auto found = _map.find(input[onnxruntime::narrow<size_t>(i)]);
      if (found == _map.end())
        output[onnxruntime::narrow<size_t>(i)] = _default_value;
      else
        output[onnxruntime::narrow<size_t>(i)] = found->second;
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
}  // namespace ml
}  // namespace onnxruntime
