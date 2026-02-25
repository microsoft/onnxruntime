// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include "data_transfer_manager.h"

#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

/// <summary>
/// Wrapper around IExecutionProvider to expose via OrtEp.
/// </summary>
class Ep : public OrtEp {
 protected:
  explicit Ep(std::unique_ptr<IExecutionProvider> impl, AllocatorPtr temp_space_cpu_allocator, AllocatorPtr temp_space_allocator)
      : OrtEp{},
        impl_(std::move(impl)),
        data_transfer_manager_{impl_->GetDataTransfer()},
        profiler_{impl_->GetProfiler()},
        temp_space_cpu_allocator_{temp_space_cpu_allocator},
        temp_space_allocator_{temp_space_allocator} {
  }

 public:
  inline IExecutionProvider* EpImpl() const noexcept {
    return impl_.get();
  }
  inline const DataTransferManager& GetDataTransferManager() const noexcept {
    return data_transfer_manager_;
  }
  Status GetTempSpaceCPUAllocator(AllocatorPtr* output) const {
    *output = temp_space_cpu_allocator_;
    return Status::OK();
  }
  Status GetTempSpaceAllocator(AllocatorPtr* output) const {
    *output = temp_space_allocator_;
    return Status::OK();
  }

 private:
  std::unique_ptr<IExecutionProvider> impl_;
  DataTransferManager data_transfer_manager_;
  std::unique_ptr<profiling::EpProfiler> profiler_;
  AllocatorPtr temp_space_cpu_allocator_;
  AllocatorPtr temp_space_allocator_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
