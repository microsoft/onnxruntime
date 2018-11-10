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

    ONNXRUNTIME_ENFORCE(info.GetAttrs<std::string>("classes_strings", string_classes).IsOK());

    ONNXRUNTIME_ENFORCE(info.GetAttr<std::string>("default_string", &default_string_).IsOK());
    ONNXRUNTIME_ENFORCE(info.GetAttr<int64_t>("default_int64", &default_int_).IsOK());

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

}  // namespace ml
}  // namespace onnxruntime
