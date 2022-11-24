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

#ifdef USE_CLOUD

class CloudEndPointInvoker {
 public:
  CloudEndPointInvoker(const CloudEndPointConfig& config) : config_(config) {}
  virtual ~CloudEndPointInvoker() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CloudEndPointInvoker);

  static std::unique_ptr<CloudEndPointInvoker> CreateInvoker(const CloudEndPointConfig& config);
  virtual void Send(gsl::span<const OrtValue> ort_inputs, std::vector<OrtValue>& ort_outputs) const noexcept = 0;
  const onnxruntime::Status& GetStaus() const { return status_; }

 protected:
  bool ReadConfig(const char* config_name, bool& config_val, bool required = true);
  bool ReadConfig(const char* config_name, std::string& config_val, bool required = true);
  bool ReadConfig(const char* config_name, onnxruntime::InlinedVector<std::string>& config_vals, bool required = true);

  CloudEndPointConfig config_;
  mutable onnxruntime::Status status_ = onnxruntime::Status::OK();
};

#else

class CloudEndPointInvoker {
 public:
  CloudEndPointInvoker() = delete;
  ~CloudEndPointInvoker() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CloudEndPointInvoker);
  static std::unique_ptr<CloudEndPointInvoker> CreateInvoker(const CloudEndPointConfig&) {
    return {};
  }
  void Send(gsl::span<const OrtValue>, std::vector<OrtValue>&){};
  const onnxruntime::Status& GetStaus() const {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED);
  }
};

#endif

}  // namespace onnxruntime
