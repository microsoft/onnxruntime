// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/ml/ml_common.h"

namespace onnxruntime {
namespace ml {

class CategoryMapper final : public OpKernel {
 public:
  CategoryMapper(const OpKernelInfo& info) : OpKernel(info) {
    std::vector<std::string> string_categories;
    std::vector<int64_t> int_categories;

    ORT_THROW_IF_ERROR(info.GetAttrs<std::string>("cats_strings", string_categories));
    ORT_THROW_IF_ERROR(info.GetAttrs<int64_t>("cats_int64s", int_categories));

    ORT_THROW_IF_ERROR(info.GetAttr<std::string>("default_string", &default_string_));
    ORT_THROW_IF_ERROR(info.GetAttr<int64_t>("default_int64", &default_int_));

    auto num_entries = string_categories.size();

    ORT_ENFORCE(num_entries == int_categories.size());

    string_to_int_map_.reserve(num_entries);
    int_to_string_map_.reserve(num_entries);

    for (size_t i = 0; i < num_entries; ++i) {
      const std::string& str = string_categories[i];
      int64_t index = int_categories[i];

      string_to_int_map_[str] = index;
      int_to_string_map_[index] = str;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::unordered_map<std::string, int64_t> string_to_int_map_;
  std::unordered_map<int64_t, std::string> int_to_string_map_;

  std::string default_string_;
  int64_t default_int_;
};

}  // namespace ml
}  // namespace onnxruntime
