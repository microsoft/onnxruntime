// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/pch.h instead."
#endif

#include "api.h"
#include "data_transfer_manager.h"

#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace ep {

/// <summary>
/// Wrapper around IExecutionProvider to expose via OrtEp.
/// </summary>
class Ep : public OrtEp {
 protected:
  explicit Ep(IExecutionProvider* impl, AllocatorPtr temp_space_cpu_allocator, AllocatorPtr temp_space_allocator)
      : OrtEp{},
        impl_(impl),
        data_transfer_manager_{impl->GetDataTransfer()},
        profiler_{impl->GetProfiler()},
        temp_space_cpu_allocator_{temp_space_cpu_allocator},
        temp_space_allocator_{temp_space_allocator} {
  }

 public:
  inline IExecutionProvider* EpImpl() const noexcept {
    return impl_.get();
  }
  inline const detail::DataTransferManager& GetDataTransferManager() const noexcept {
    return data_transfer_manager_;
  }
  [[nodiscard]] Status GetTempSpaceCPUAllocator(AllocatorPtr* output) const {
    *output = temp_space_cpu_allocator_;
    return Status::OK();
  }
  [[nodiscard]] Status GetTempSpaceAllocator(AllocatorPtr* output) const {
    *output = temp_space_allocator_;
    return Status::OK();
  }

 private:
  std::unique_ptr<IExecutionProvider> impl_;
  detail::DataTransferManager data_transfer_manager_;
  std::unique_ptr<profiling::EpProfiler> profiler_;
  AllocatorPtr temp_space_cpu_allocator_;
  AllocatorPtr temp_space_allocator_;
};

}  // namespace ep
}  // namespace onnxruntime
