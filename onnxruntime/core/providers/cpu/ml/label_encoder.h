// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/ml/ml_common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/safeint.h"

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

    ORT_THROW_IF_ERROR(info.GetAttrs<TKey>(key_field_name_, keys));
    ORT_THROW_IF_ERROR(info.GetAttrs<TValue>(value_field_name_, values));

    auto num_keys = keys.size();
    auto num_values = values.size();
    ORT_ENFORCE(num_keys == num_values, "The ", key_field_name_, " and ", value_field_name_,
                " attributes in LabelEncoder ", "(name: ", info.node().Name(), ") must have the same length. ",
                "However, the number of key is ", num_keys, " and the number of ", "values is ", num_values, ".");
    map_.reserve(num_keys);
    for (size_t i = 0; i < num_keys; ++i) map_.emplace(keys[i], values[i]);
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
      const auto found = map_.find(*input_iter);
      *output_iter = found == map_.end() ? default_value_ : found->second;
      ++output_iter;
      ++input_iter;
    }
    return Status::OK();
  }

 private:
  // Specialize this method to set attribute names. For example, if keys' type
  // is 64-bit integer, key_field_name_ should be "keys_int64s". Field names
  // for other types can be found in ONNX spec.
  void InitializeSomeFields(const OpKernelInfo& info);

  // A collection of key-value pairs. Each (a_key, a_value) pair
  // means that the "a_key" in the input would be mapped to "a_value".
  // If map_ doesn't contain "a_key", we use default_value_ as its output.
  InlinedHashMap<TKey, TValue> map_;
  TValue default_value_;
  // ONNX attribute name to load keys.
  std::string key_field_name_;
  // ONNX attribute name to load values.
  std::string value_field_name_;
};

template <typename T>
std::vector<T> GetAttribute(const OpKernelInfo& info, const std::string& name, const std::string& tensor_name) {
  if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, float> || std::is_same_v<T, int64_t>) {
    std::vector<T> attrs;
    if (info.GetAttrs<T>(name, attrs).IsOK()) {
      return attrs;
    }
  }
  ONNX_NAMESPACE::TensorProto attr_tensor_proto;
  auto result = info.GetAttr(tensor_name, &attr_tensor_proto);
  if (name.empty()) {
    ORT_ENFORCE(result.IsOK(), "LabelEncoder is missing attribute ", tensor_name);
  } else {
    ORT_ENFORCE(result.IsOK(), "LabelEncoder is missing attribute ", tensor_name, " or ", name);
  }
  SafeInt<int64_t> element_count(1);
  for (auto dim : attr_tensor_proto.dims()) {
    element_count *= dim;
  }
  const SafeInt<size_t> tensor_size(element_count);
  std::vector<T> out(tensor_size);
  result = utils::UnpackTensor<T>(attr_tensor_proto, Path(), out.data(), tensor_size);
  ORT_ENFORCE(result.IsOK(), "LabelEncoder could not unpack tensor attribute ", name);
  return out;
}

template <typename T>
T GetDefault(const OpKernelInfo& info, const std::string& attr_name, const T& backup) {
  ONNX_NAMESPACE::TensorProto attr_tensor_proto;
  auto result = info.GetAttr("default_tensor", &attr_tensor_proto);
  if (result.IsOK() && utils::HasDataType(attr_tensor_proto)) {
    T default_value;
    result = utils::UnpackTensor<T>(attr_tensor_proto, Path(), &default_value, 1);
    ORT_ENFORCE(result.IsOK(), "LabelEncoder could not unpack default tensor ", attr_name);
    return default_value;
  } else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, float> || std::is_same_v<T, int64_t>) {
    T default_value;
    result = info.GetAttr<T>(attr_name, &default_value);
    if (result.IsOK()) {
      return default_value;
    }
  }
  return backup;
}

// We don't make use of InlinedHashMap since we make use of a custom hash and equality function.
// Introducing new template parameters in inlined_containers_fwd.h creates compilation errors
// (see https://github.com/microsoft/onnxruntime/pull/17977#discussion_r1446510961).
#ifndef DISABLE_ABSEIL
template <typename T>
using HashFunc = absl::container_internal::hash_default_hash<T>;

template <typename T>
using EqualFunc = absl::container_internal::hash_default_eq<T>;

template <typename K, typename V, typename Hash, typename Equal>
using HashMap = absl::flat_hash_map<K, V, Hash, Equal>;
#else
template <typename T>
using HashFunc = std::hash<T>;

template <typename T>
using EqualFunc = std::equal_to<T>;

template <typename K, typename V, typename Hash, typename Equal>
using HashMap = std::unordered_map<K, V, Hash, Equal>;
#endif  // DISABLE_ABSEIL

template <typename T>
struct NaNHash {
  size_t operator()(const T& value) const {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(value)) {
        return 0;
      }
    }
    return HashFunc<T>{}(value);
  }
};

template <typename T>
struct NaNEqual {
  bool operator()(const T& lhs, const T& rhs) const {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(lhs) && std::isnan(rhs)) {
        return true;
      }
    }
    return EqualFunc<T>{}(lhs, rhs);
  }
};

template <typename TKey, typename TValue>
class LabelEncoder_4 final : public OpKernel {
 public:
  LabelEncoder_4(const OpKernelInfo& kernel_info) : OpKernel(kernel_info) {
    InitializeAttrFields(kernel_info);
    auto keys = GetAttribute<TKey>(kernel_info, key_field_name_, "keys_tensor");
    auto values = GetAttribute<TValue>(kernel_info, value_field_name_, "values_tensor");
    ORT_ENFORCE(keys.size() == values.size(), "Keys and values must have the same length.");
    for (size_t i = 0; i < keys.size(); ++i) {
      map_.emplace(keys[i], values[i]);
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
      const auto found = map_.find(*input_iter);
      *output_iter = found == map_.end() ? default_value_ : found->second;
      ++output_iter;
      ++input_iter;
    }
    return Status::OK();
  }

 private:
  void InitializeAttrFields(const OpKernelInfo& kernel_info);
  HashMap<TKey, TValue, NaNHash<TKey>, NaNEqual<TKey>> map_;
  TValue default_value_;
  std::string key_field_name_;
  std::string value_field_name_;
};

}  // namespace ml
}  // namespace onnxruntime
