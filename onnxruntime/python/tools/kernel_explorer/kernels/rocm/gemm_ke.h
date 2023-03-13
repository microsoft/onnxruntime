// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace onnxruntime {

template <typename T>
class IBatchedGemmKernelExplorer : public ISelectableKernelExplorer {
 protected:
  void CopyAsBsCsPointersToDevice(const std::vector<DeviceArray>& as,
                                  const std::vector<DeviceArray>& bs,
                                  const std::vector<DeviceArray>& cs,
                                  int64_t batch) {
    ORT_ENFORCE(as.size() == batch);
    ORT_ENFORCE(bs.size() == batch);
    ORT_ENFORCE(cs.size() == batch);
    CopyPointersToDevice(as, dev_as_);
    CopyPointersToDevice(bs, dev_bs_);
    CopyPointersToDevice(cs, dev_cs_);
  }

  static void CopyPointersToDevice(const std::vector<DeviceArray>& src, std::shared_ptr<T*>& dst) {
    // convert pointers in vector<DeviceArray> to continuous for copying
    std::vector<void*> tmp;
    auto cvt_to_raw_ptr = [](const DeviceArray& x) { return static_cast<T*>(x.ptr()); };
    std::transform(src.cbegin(), src.cend(), std::back_inserter(tmp), cvt_to_raw_ptr);

    // create buffer for pointers
    T** ptrs;
    HIP_CALL_THROW(hipMalloc(&ptrs, src.size() * sizeof(T*)));
    dst.reset(ptrs, [](void* addr) { HIP_CALL_THROW(hipFree(addr)); });

    // copy host pointers to buffer
    HIP_CALL_THROW(hipMemcpy(ptrs, tmp.data(), src.size() * sizeof(T*), hipMemcpyHostToDevice));
  }

  std::shared_ptr<T*> dev_as_;
  std::shared_ptr<T*> dev_bs_;
  std::shared_ptr<T*> dev_cs_;
};

}  // namespace onnxruntime
