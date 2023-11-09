// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "interface/graph/graph.h"
#include "interface/framework/kernel.h"
//#include "core/session/onnxruntime_c_api.h"
#include "core/framework/ortdevice.h"
#include "core/framework/stream_handles.h"
#include "core/framework/node_compute_info.h"
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

struct Allocator {
  virtual ~Allocator() = default;
  enum DevType {
    CPU = 0,
    GPU,
    FPGA,
    TPU,
  };
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void*) = 0;
  DevType dev_type = CPU;
};

using AllocatorPtr = std::unique_ptr<Allocator>;
using AllocatorPtrs = std::vector<AllocatorPtr>;

class ExecutionProvider {
 public:
  ExecutionProvider() { default_device_ = OrtDevice(); };
  virtual ~ExecutionProvider() = default;

  AllocatorPtrs& GetAllocators() { return allocators_; }

  std::string& GetType() { return type_; }
  OrtDevice& GetDevice() { return default_device_; }

  virtual bool CanCopy(const OrtDevice&, const OrtDevice&) { return false; }
  //virtual void MemoryCpy(Ort::UnownedValue&, Ort::ConstValue const&) {}
  virtual void RegisterStreamHandlers(IStreamCommandHandleRegistry&, std::map<OrtDevice, OrtAllocator*>&) const {}
  virtual std::vector<std::unique_ptr<SubGraphDef>> GetCapability(GraphViewRef*) { return std::vector<std::unique_ptr<SubGraphDef>>(); }
  virtual common::Status Compile(std::vector<std::unique_ptr<GraphViewRef>>&, std::vector<std::unique_ptr<NodeViewRef>>&, std::vector<NodeComputeInfo>&) { return common::Status::OK(); }

  // latest kernel inteface
  virtual void RegisterKernels(interface::IKernelRegistry& kernel_registry) = 0;

 protected:
  AllocatorPtrs allocators_;
  std::string type_;
  OrtDevice default_device_;
};
}  // namespace interface

}  // namespace onnxruntime
