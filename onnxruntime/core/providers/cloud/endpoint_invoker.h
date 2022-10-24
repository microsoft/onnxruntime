// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "gsl/gsl"
#include "core/framework/tensor.h"
#include "core/common/inlined_containers.h"

namespace triton {
namespace client {
class InferenceServerHttpClient;
}
}  // namespace trition

namespace onnxruntime {
namespace cloud {

//struct Data {
//  char* content{};
//  size_t size_in_byte{};
//};

using EndPointConfig = onnxruntime::InlinedHashMap<std::string, std::string>;

enum class EndPointType {
  triton,
  unknown,
};

using TensorPtr = std::unique_ptr<onnxruntime::Tensor>;
using TensorPtrArray = onnxruntime::InlinedVector<TensorPtr>;
using ConstTensorPtrArray = gsl::span<onnxruntime::Tensor* const>;

class EndPointInvoker {
 public:
  EndPointInvoker(const EndPointConfig& config) : config_(config) {}
  virtual ~EndPointInvoker();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EndPointInvoker);

  virtual TensorPtrArray Send(ConstTensorPtrArray ort_inputs) const = 0;
  const onnxruntime::Status GetStaus() const { return status_; }
 protected:
  EndPointConfig config_;
  mutable onnxruntime::Status status_ = onnxruntime::Status::OK();
};

class TritonInvokder : public EndPointInvoker {
 public:
  TritonInvokder(const EndPointConfig& config);
  TensorPtrArray Send(ConstTensorPtrArray input_tensors) const override;

 private:
  bool ReadConfig(const char* config_name, std::string& config_val);
  bool ReadConfig(const char* config_name, onnxruntime::InlinedVector<std::string>& config_vals);

  std::string uri_;
  std::string key_; // access token for bearer authentication
  std::string model_name_;
  std::string model_ver_;
  onnxruntime::InlinedVector<std::string> input_names_;
  onnxruntime::InlinedVector<std::string> output_names_;
  std::shared_ptr<CPUAllocator> cpu_allocator_;
  std::unique_ptr<triton::client::InferenceServerHttpClient> triton_client_;
};

}  // namespace cloud
}  // namespace onnxruntime