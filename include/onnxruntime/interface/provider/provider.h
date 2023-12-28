// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "interface/graph/graph.h"
#include "interface/framework/kernel.h"
//#include "core/session/onnxruntime_c_api.h"
//#ifdef ORT_API_MANUAL_INIT
//#include "core/session/onnxruntime_cxx_api.h"
//#endif
#include "core/framework/ortdevice.h"
#include "core/framework/stream_handles.h"
#include "core/framework/node_compute_info.h"
#include "core/framework/data_layout.h"
#include <climits>

namespace onnxruntime {

namespace interface {

struct SubGraphDef {
  struct MetaDef {
    std::string name;
    std::string domain;
    int since_version;

    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> constant_initializers;
    std::string doc_string;
  };

  std::vector<size_t> nodes;
  void SetMetaDef(std::unique_ptr<MetaDef>&& meta_def) { meta_def_ = std::move(meta_def); }
  MetaDef* GetMetaDef() { return meta_def_.get(); };

 private:
  std::unique_ptr<MetaDef> meta_def_;
};

enum OrtMemType {   // from onnxruntime_c_api.h
  OrtMemTypeCPUInput = -2,
  OrtMemTypeCPUOutput = -1,
  OrtMemTypeCPU = OrtMemTypeCPUOutput,
  OrtMemTypeDefault = 0,
};

struct Allocator {
  virtual ~Allocator() = default;
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void*) = 0;
  OrtDevice device;
};

using AllocatorPtr = std::unique_ptr<Allocator>;
using AllocatorPtrs = std::vector<AllocatorPtr>;

class ExecutionProvider {
 public:
  ExecutionProvider(std::string type, OrtDevice device = OrtDevice()) : type_{type}, default_device_(device) {
//#ifdef ORT_API_MANUAL_INIT
//    Ort::InitApi();
//#endif
  };
  virtual ~ExecutionProvider() = default;

  std::string& GetType() { return type_; }
  OrtDevice& GetDevice() { return default_device_; }  // only for provider_adapter's constructor. Need to delete once provider_adapter is retired

  virtual bool CanCopy(const OrtDevice&, const OrtDevice&) { return false; }
  // virtual void MemoryCpy(Ort::UnownedValue&, Ort::ConstValue const&) {}
  virtual void RegisterStreamHandlers(IStreamCommandHandleRegistry&, std::map<OrtDevice, OrtAllocator*>&) const {}
  virtual std::vector<std::unique_ptr<SubGraphDef>> GetCapability(GraphViewRef*) { return std::vector<std::unique_ptr<SubGraphDef>>(); }
  virtual common::Status Compile(std::vector<std::unique_ptr<GraphViewRef>>&, std::vector<std::unique_ptr<NodeViewRef>>&, std::vector<NodeComputeInfo>&) { return common::Status::OK(); }

  // latest kernel interface
  virtual void RegisterKernels(interface::IKernelRegistry& kernel_registry) = 0;

  virtual int GetDeviceId() { return default_device_.Id(); }

  virtual DataLayout GetPreferredLayout() const { return DataLayout::NCHW; }
  virtual bool ConcurrentRunSupported() const { return true; }
  virtual AllocatorPtrs CreatePreferredAllocators() { return AllocatorPtrs(); }
  virtual OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const {
    if (mem_type == OrtMemTypeCPUInput || mem_type == OrtMemTypeCPUOutput) {
      return OrtDevice();  // default return CPU device.
    }
    return default_device_;
  };

 protected:
  std::string type_;
  OrtDevice default_device_;
};
}  // namespace interface

}  // namespace onnxruntime
