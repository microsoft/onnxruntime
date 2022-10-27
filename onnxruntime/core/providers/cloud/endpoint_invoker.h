// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "gsl/gsl"
#include "core/framework/tensor.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace cloud {

using EndPointConfig = std::unordered_map<std::string, std::string>;
using TensorPtr = std::unique_ptr<onnxruntime::Tensor>;
using TensorPtrArray = onnxruntime::InlinedVector<TensorPtr>;
using ConstTensorPtrArray = gsl::span<onnxruntime::Tensor* const>;

class EndPointInvoker {
 public:
  EndPointInvoker(const EndPointConfig& config) : config_(config) {}
  virtual ~EndPointInvoker();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EndPointInvoker);

  static std::unique_ptr<EndPointInvoker> CreateInvoker(const EndPointConfig& config);
  virtual void Send(gsl::span<const OrtValue> ort_inputs, std::vector<OrtValue>& ort_outputs) const = 0;
  const onnxruntime::Status& GetStaus() const { return status_; }
 protected:
  EndPointConfig config_;
  mutable onnxruntime::Status status_ = onnxruntime::Status::OK();
};

}  // namespace cloud
}  // namespace onnxruntime