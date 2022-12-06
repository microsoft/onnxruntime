// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include "gsl/gsl"
#include "core/framework/tensor.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

using CloudEndPointConfig = std::unordered_map<std::string, std::string>;
using TensorPtr = std::unique_ptr<onnxruntime::Tensor>;
using TensorPtrArray = onnxruntime::InlinedVector<TensorPtr>;
using ConstTensorPtrArray = gsl::span<onnxruntime::Tensor* const>;

class CloudEndPointInvoker {
 public:
  CloudEndPointInvoker(const CloudEndPointConfig& config) : config_(config) {}
  virtual ~CloudEndPointInvoker() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CloudEndPointInvoker);

  static std::unique_ptr<CloudEndPointInvoker> CreateInvoker(const CloudEndPointConfig& config);

  virtual onnxruntime::Status Send(const CloudEndPointConfig& run_options,
                                   const InlinedVector<std::string>& input_names,
                                   gsl::span<const OrtValue> ort_inputs,
                                   const InlinedVector<std::string>& output_names,
                                   std::vector<OrtValue>& ort_outputs) const noexcept = 0;

 protected:
  bool ReadConfig(const char* config_name, bool& config_val, bool required = true);
  bool ReadConfig(const char* config_name, std::string& config_val, bool required = true);
  bool ReadConfig(const char* config_name, onnxruntime::InlinedVector<std::string>& config_vals, bool required = true);

  CloudEndPointConfig config_;
  mutable onnxruntime::Status status_ = onnxruntime::Status::OK();
};
}  // namespace onnxruntime
