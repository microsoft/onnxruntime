// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
#include "core/framework/provider_options.h"
#include <memory>
struct OrtCustomOpDomain;
namespace onnxruntime {
// The suppressed warning is: "The type with a virtual function needs either public virtual or protected nonvirtual destructor."
// However, we do not allocate this type on heap.
// Please do not new or delete this type(and subtypes).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26436)
#endif
struct IExecutionProviderFactory;
struct Provider {
  // Takes a pointer to a provider specific structure to create the factory. For example, with OpenVINO it is a pointer to an OrtOpenVINOProviderOptions structure
  virtual std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* /*provider_options*/) { return nullptr; }

  // Old simple device_id API to create provider factories, currently used by DNNL And TensorRT
  virtual std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int /*device_id*/) { return nullptr; }

  virtual void* GetInfo() { return nullptr; }  // Returns a provider specific information interface if it exists

  // Convert provider options struct to ProviderOptions which is a map
  virtual ProviderOptions GetProviderOptions(const void* /*provider options struct*/) { return {}; }

  // Update provider options from key-value string configuration
  virtual void UpdateProviderOptions(void* /*provider options to be configured*/, const ProviderOptions& /*key-value string provider options*/){};

  // Get provider specific custom op domain list. Provider has the resposibility to release OrtCustomOpDomain instances it creates.
  virtual void GetCustomOpDomainList(IExecutionProviderFactory* /*pointer to factory instance*/, std::vector<OrtCustomOpDomain*>& /*provider custom op domain list*/){};

  virtual void Initialize() = 0;  // Called right after loading the shared library, if this throws any errors Shutdown() will be called and the library unloaded
  virtual void Shutdown() = 0;    // Called right before unloading the shared library

 protected:
  ~Provider() = default;  // Can only be destroyed through a subclass instance
};
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
}  // namespace onnxruntime
