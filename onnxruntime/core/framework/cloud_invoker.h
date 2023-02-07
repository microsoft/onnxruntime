// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifdef USE_AZURE
#include <memory>
#include "gsl/gsl"
#include "core/framework/tensor.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

using CloudEndPointConfig = std::unordered_map<std::string, std::string>;
using TensorPtr = std::unique_ptr<onnxruntime::Tensor>;
using TensorPtrArray = onnxruntime::InlinedVector<TensorPtr>;

class CloudEndPointInvoker {
 public:
  CloudEndPointInvoker(const CloudEndPointConfig& config, const AllocatorPtr& allocator);
  virtual ~CloudEndPointInvoker() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CloudEndPointInvoker);

  static Status CreateInvoker(const CloudEndPointConfig& config,
                              const AllocatorPtr& allocator,
                              std::unique_ptr<CloudEndPointInvoker>& invoker);

  virtual onnxruntime::Status Send(const CloudEndPointConfig& run_options,
                                   const InlinedVector<std::string>& input_names,
                                   gsl::span<const OrtValue> ort_inputs,
                                   const InlinedVector<std::string>& output_names,
                                   std::vector<OrtValue>& ort_outputs) const = 0;

 protected:
  void ReadConfig(const char* config_name, std::string& config_val, bool required = true);
  CloudEndPointConfig config_;
  const AllocatorPtr& allocator_;
};
}  // namespace onnxruntime
#endif
