// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph_transformer.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct CPU execution providers.
struct CPUExecutionProviderInfo {
  bool create_arena{true};

  explicit CPUExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}
  CPUExecutionProviderInfo() = default;
};

using FuseRuleFn = std::function<void(const onnxruntime::GraphViewer&, std::vector<std::unique_ptr<ComputeCapability>>&)>;

// Logical device representation.
class CPUExecutionProvider : public IExecutionProvider {
 public:
  explicit CPUExecutionProvider(const CPUExecutionProviderInfo& info) {
    DeviceAllocatorRegistrationInfo device_info({OrtMemTypeDefault, [](int) {
          return std::make_unique<CPUAllocator>(); }, std::numeric_limits<size_t>::max()});
#ifdef USE_JEMALLOC
    ORT_UNUSED_PARAMETER(info);
    //JEMalloc already has memory pool, so just use device allocator.
    InsertAllocator(
        std::shared_ptr<IArenaAllocator>(
            std::make_unique<DummyArena>(device_info.factory(0))));
#else
    if (info.create_arena)
      InsertAllocator(CreateAllocator(device_info));
    else
      InsertAllocator(
          std::shared_ptr<IArenaAllocator>(
              std::make_unique<DummyArena>(device_info.factory(0))));
#endif
  }

  std::string Type() const override {
    return onnxruntime::kCpuExecutionProvider;
  }

  virtual std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  ///requires src.buffer_deleter_ == nullptr
  Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    ORT_ENFORCE(strcmp(dst.Location().name, CPU) == 0);

    // Todo: support copy with different devices.
    if (strcmp(src.Location().name, CPU) != 0) {
      ORT_NOT_IMPLEMENTED("copy from ", src.Location().name, " is not implemented");
    }

    // no really copy needed if is copy to cpu.
    dst.ShallowCopy(src);

    return Status::OK();
  }

  const void* GetExecutionHandle() const noexcept override {
    // The CPU interface does not return anything interesting.
    return nullptr;
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  void InsertFusedRules(FuseRuleFn rule);
 protected:
  std::vector<FuseRuleFn> fuse_rules_;
};
}  // namespace onnxruntime
