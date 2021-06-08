// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

struct Provider {
  // Takes a pointer to a provider specific structure to create the factory. For example, with OpenVINO it is a pointer to an OrtOpenVINOProviderOptions structure
  virtual std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* /*provider_options*/) { return nullptr; }

  // Old simple device_id API to create provider factories, currently used by DNNL And TensorRT
  virtual std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int /*device_id*/) { return nullptr; }

  virtual void* GetInfo() { return nullptr; }  // Returns a provider specific information interface if it exists
  virtual void Shutdown() = 0;
};

}  // namespace onnxruntime
